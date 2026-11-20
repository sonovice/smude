package com.notavium.smude;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.net.Uri;
import java.util.List;
import androidx.activity.result.PickVisualMediaRequest;
import androidx.activity.ComponentActivity;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.core.content.ContextCompat;
import com.longbridge.nsh.NshView;
import java.util.function.Consumer;

/** Platform services only; all application UI and navigation are Rust NSH. */
public final class MainActivity extends ComponentActivity {
    private NshView nsh;
    Consumer<java.util.Map<String,Object>> captureEvents;
    void captureEvent(String name,java.util.Map<String,Object> fields){
        java.util.Map<String,Object> payload=new java.util.LinkedHashMap<>(fields);payload.put("event",name);
        runOnUiThread(()->{if(captureEvents!=null)captureEvents.accept(payload);});
    }
    private Consumer<Boolean> permissionResult;
    private Consumer<List<Uri>> imageResult;
    private final ActivityResultLauncher<PickVisualMediaRequest> photos = registerForActivityResult(
        new ActivityResultContracts.PickMultipleVisualMedia(), this::pickedImages);
    private final ActivityResultLauncher<String[]> files = registerForActivityResult(
        new ActivityResultContracts.OpenMultipleDocuments(), this::pickedImages);
    private void pickedImages(List<Uri> uris) {
        Consumer<List<Uri>> callback=imageResult; imageResult=null;
        if(callback!=null) callback.accept(uris);
    }
    void pickImages(boolean fromFiles,Consumer<List<Uri>> callback) {
        if(imageResult!=null) throw new IllegalStateException("An image picker is already open");
        imageResult=callback;
        try {
            if(fromFiles) files.launch(new String[]{"image/*"});
            else photos.launch(new PickVisualMediaRequest.Builder()
                .setMediaType(ActivityResultContracts.PickVisualMedia.ImageOnly.INSTANCE)
                .setOrderedSelection(true).setAccentColor(0xffffcc66L).build());
        } catch(RuntimeException e) { imageResult=null; throw e; }
    }
    private final ActivityResultLauncher<String> permission = registerForActivityResult(
        new ActivityResultContracts.RequestPermission(), granted -> {
            Consumer<Boolean> callback = permissionResult;
            permissionResult = null;
            if (callback != null) callback.accept(granted);
        });

    @Override public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        orientationPolicy();
        nsh = new NshView(this);
        nsh.registerComponent("smude.control", context -> new ControlsComponent(this, context));
        nsh.registerComponent("smude.camera", context -> new CameraComponent(this, context));
        nsh.registerComponent("smude.services", context -> new ServicesComponent(this, context));
        setContentView(nsh);
        nsh.setThemeOverrides("{\"cssVars\":{\"light\":{\"destructive\":\"#ff6b6b\"},\"dark\":{\"destructive\":\"#ff6b6b\"}}}");
        nsh.runRustApp("nsh");
        // NSH owns OnBackAnimationCallback on API 34+ and the older fallbacks.
        // A global AndroidX callback here would steal predictive-back progress.
    }

    private void orientationPolicy(){
        int value=getResources().getConfiguration().smallestScreenWidthDp>=600?android.content.pm.ActivityInfo.SCREEN_ORIENTATION_UNSPECIFIED:android.content.pm.ActivityInfo.SCREEN_ORIENTATION_PORTRAIT;
        if(getRequestedOrientation()!=value)setRequestedOrientation(value);
    }
    @Override public void onConfigurationChanged(android.content.res.Configuration configuration){super.onConfigurationChanged(configuration);orientationPolicy();}

    void requestCamera(Consumer<Boolean> callback) {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            callback.accept(true);
        } else {
            permissionResult = callback;
            permission.launch(Manifest.permission.CAMERA);
        }
    }

    /** Legacy button navigation only; API 33+ stays with NSH/system callbacks. */
    @Override @SuppressWarnings("deprecation") public void onBackPressed() {
        if (android.os.Build.VERSION.SDK_INT < 33 && nsh != null && nsh.handleBack()) return;
        super.onBackPressed();
    }

    @Override protected void onDestroy() {
        permissionResult = null; imageResult = null;
        if (nsh != null) { nsh.destroy(); nsh = null; }
        super.onDestroy();
    }
}
