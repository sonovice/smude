package com.gridnotelabs.smude.mobile

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.provider.MediaStore
import androidx.activity.result.ActivityResult
import androidx.core.content.FileProvider
import app.tauri.annotation.ActivityCallback
import app.tauri.annotation.Command
import app.tauri.annotation.InvokeArg
import app.tauri.annotation.TauriPlugin
import app.tauri.plugin.Invoke
import app.tauri.plugin.Plugin
import app.tauri.plugin.JSObject
import java.io.File

@InvokeArg
class ExportArgs {
    lateinit var source: String
    lateinit var destination: String
}

@InvokeArg
class ImageArgs {
    lateinit var source: String
    lateinit var destination: String
}

@InvokeArg
class CaptureArgs {
    lateinit var destination: String
}

/** Keep the provider-owned stream alive through close. A detached file
 * descriptor can leave DocumentsProvider/MediaStore reporting a zero size. */
@TauriPlugin
class DevicePlugin(private val activity: Activity) : Plugin(activity) {
    @Command
    fun decodeImage(invoke: Invoke) {
        val args = invoke.parseArgs(ImageArgs::class.java)
        Thread {
            try {
                val source = ImageDecoder.createSource(File(args.source))
                val bitmap = ImageDecoder.decodeBitmap(source) { decoder, info, _ ->
                    val size = info.size
                    require(size.width > 0 && size.height > 0 &&
                        maxOf(size.width, size.height) <= 12000 &&
                        size.width.toLong() * size.height <= 64_000_000) { "Bild ist zu groß." }
                    decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
                }
                try {
                    File(args.destination).outputStream().use {
                        check(bitmap.compress(Bitmap.CompressFormat.PNG, 100, it)) { "PNG konnte nicht geschrieben werden." }
                    }
                } finally { bitmap.recycle() }
                invoke.resolve()
            } catch (error: Exception) { invoke.reject(error.message ?: "Bildformat nicht lesbar.") }
        }.start()
    }

    @Command
    fun capturePhoto(invoke: Invoke) {
        try {
            val args = invoke.parseArgs(CaptureArgs::class.java)
            val uri = FileProvider.getUriForFile(activity, activity.packageName + ".fileprovider", File(args.destination))
            val intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
                .putExtra(MediaStore.EXTRA_OUTPUT, uri)
                .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_WRITE_URI_PERMISSION)
            startActivityForResult(invoke, intent, "cameraResult")
        } catch (error: Exception) { invoke.reject(error.message ?: "Kamera kann nicht geöffnet werden.") }
    }

    @ActivityCallback
    fun cameraResult(invoke: Invoke, result: ActivityResult) {
        val destination = invoke.parseArgs(CaptureArgs::class.java).destination
        val response = JSObject()
        if (result.resultCode == Activity.RESULT_OK && File(destination).length() > 0) {
            response.put("path", destination)
        } else {
            File(destination).delete()
            response.put("path", org.json.JSONObject.NULL)
        }
        invoke.resolve(response)
    }

    @Command
    fun writeExport(invoke: Invoke) {
        val args = invoke.parseArgs(ExportArgs::class.java)
        Thread {
            try {
                val uri = Uri.parse(args.destination)
                val stream = activity.contentResolver.openOutputStream(uri, "wt")
                    ?: error("Exportziel kann nicht geöffnet werden")
                stream.use { output -> File(args.source).inputStream().use { it.copyTo(output) } }
                invoke.resolve()
            } catch (error: Exception) {
                invoke.reject(error.message ?: "Export fehlgeschlagen")
            }
        }.start()
    }
}
