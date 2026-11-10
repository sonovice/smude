import java.io.File
import java.util.zip.ZipInputStream
import java.util.zip.ZipOutputStream
import java.util.zip.ZipEntry

plugins { id("com.android.application") }

val uiRoot = rootProject.projectDir.parentFile
val sdkRoot = providers.environmentVariable("NSH_HOME").orElse("${System.getProperty("user.home")}/.nsh")
val runtimeAar = file("${sdkRoot.get()}/embed/android/nsh.aar")
val runtimeDir = layout.buildDirectory.dir("nsh-runtime")
val rustRuntime = file("$uiRoot/.nsh/rust/android-rust-release/lib")
val ndkHostTag = when {
    System.getProperty("os.name").startsWith("Mac") -> "darwin-x86_64"
    System.getProperty("os.name").startsWith("Windows") -> "windows-x86_64"
    else -> "linux-x86_64"
}

// Use NSH's Java runtime, never its stock libnsh.so: that library has no Smude root.
val prepareNshRuntime = tasks.register<Sync>("prepareNshRuntime") {
    from(zipTree(runtimeAar)) { include("classes.jar", "proguard.txt") }
    into(runtimeDir)
    // Compile our small ScrollBox fix with the app. Remove the upstream
    // definition and its inner classes so D8 sees exactly one implementation.
    doLast {
        val jar = runtimeDir.get().file("classes.jar").asFile
        val filtered = File(jar.parentFile, "classes-filtered.jar")
        ZipInputStream(jar.inputStream()).use { input ->
            ZipOutputStream(filtered.outputStream()).use { output ->
                var entry = input.nextEntry
                while (entry != null) {
                    if (entry.name != "com/longbridge/nsh/ScrollBox.class" &&
                        !entry.name.startsWith("com/longbridge/nsh/ScrollBox${'$'}")) {
                        output.putNextEntry(ZipEntry(entry.name))
                        input.copyTo(output)
                        output.closeEntry()
                    }
                    entry = input.nextEntry
                }
            }
        }
        check(filtered.renameTo(jar)) { "Could not replace the filtered NSH runtime" }
    }
}
val verifyRustRuntime = tasks.register("verifyRustRuntime") {
    doLast {
        check(file("$rustRuntime/arm64-v8a/libnsh.so").isFile) {
            "Build the Rust UI first: cd .. && nsh package --target android --abi arm64-v8a"
        }
    }
}
val prepareAssets = tasks.register<Sync>("prepareAssets") {
    from(file("$uiRoot/assets")) { into("nsh/assets/@smude-mobile-ui") }
    into(layout.buildDirectory.dir("generated/assets"))
}
// Reuse the original artwork without baking in a launcher-specific mask.
val prepareLauncherIcon = tasks.register<Sync>("prepareLauncherIcon") {
    from(file("$uiRoot/assets/app-icon.png")) {
        into("drawable-nodpi")
        rename { "launcher_artwork.png" }
    }
    into(layout.buildDirectory.dir("generated/launcher-res"))
}

android {
    namespace = "com.gridnotelabs.smude.design"
    compileSdk = 36
    ndkVersion = "29.0.14206865"
    defaultConfig {
        applicationId = "com.gridnotelabs.smude.design"
        minSdk = 28
        targetSdk = 35
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        versionCode = 38
        versionName = "0.1.0"
        ndk { abiFilters += "arm64-v8a" }
    }
    sourceSets["main"].java.srcDir(file("$uiRoot/vendor/nsh-platform-android/java"))
    sourceSets["main"].jniLibs.srcDir(rustRuntime)
    sourceSets["main"].jniLibs.srcDir(layout.buildDirectory.dir("generated/jniLibs").get().asFile)
    sourceSets["main"].assets.srcDir(layout.buildDirectory.dir("generated/assets").get().asFile)
    sourceSets["main"].res.srcDir(layout.buildDirectory.dir("generated/launcher-res").get().asFile)
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    buildTypes {
        release {
            isMinifyEnabled = true
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro", runtimeDir.get().file("proguard.txt"))
        }
    }
}
// ONNX Runtime is linked into libnsh.so, but its C++ standard library is
// deliberately dynamic. Android does not supply libc++_shared.so itself.
val prepareCppRuntime = tasks.register<Sync>("prepareCppRuntime") {
    from(androidComponents.sdkComponents.ndkDirectory.map {
        it.file("toolchains/llvm/prebuilt/$ndkHostTag/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so")
    })
    into(layout.buildDirectory.dir("generated/jniLibs/arm64-v8a"))
}
dependencies {
    implementation("com.google.android.material:material:1.13.0")
    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test:runner:1.6.2")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
    implementation(files(runtimeDir.map { it.file("classes.jar") }).builtBy(prepareNshRuntime))
    implementation("androidx.activity:activity:1.10.1")
    implementation("androidx.camera:camera-camera2:1.5.3")
    implementation("androidx.camera:camera-lifecycle:1.5.3")
    implementation("androidx.camera:camera-view:1.5.3")
}
tasks.named("preBuild") { dependsOn(prepareNshRuntime, prepareAssets, prepareLauncherIcon) }
tasks.matching { it.name.contains("JniLibFolders") }.configureEach { dependsOn(verifyRustRuntime, prepareCppRuntime) }
