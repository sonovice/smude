package com.gridnotelabs.smude.design;

import android.content.ClipData;
import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.ColorSpace;
import android.graphics.ImageDecoder;
import android.media.ExifInterface;
import java.io.FileInputStream;
import android.net.Uri;
import android.os.Handler;
import android.os.Looper;
import android.view.View;
import android.view.accessibility.AccessibilityManager;
import android.os.Build;
import androidx.core.content.FileProvider;
import com.longbridge.nsh.NshComponent;
import com.longbridge.nsh.NshComponentContext;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/** Native OS services; Rust owns every application screen and domain operation. */
final class ServicesComponent implements NshComponent {
    private final MainActivity activity;
    private final NshComponentContext context;
    private final View view;
    private final Handler main = new Handler(Looper.getMainLooper());
    private final ExecutorService io = Executors.newSingleThreadExecutor();
    private final File root;
    private volatile boolean destroyed;
    private boolean importing;
    private final java.util.function.Consumer<Map<String,Object>> captureListener=payload->emit((String)payload.get("event"),payload);
    private List<?> dateRequests=List.of();
    private final AccessibilityManager accessibility;
    private final AccessibilityManager.AccessibilityStateChangeListener accessibilityChanged=enabled->feedbackPreferences();
    private final AccessibilityManager.TouchExplorationStateChangeListener explorationChanged=enabled->feedbackPreferences();

    ServicesComponent(MainActivity activity, NshComponentContext context) {
        this.activity=activity; this.context=context;
        activity.captureEvents=captureListener;
        accessibility=(AccessibilityManager)activity.getSystemService(Context.ACCESSIBILITY_SERVICE);
        view=new View(activity) {
            @Override protected void onWindowVisibilityChanged(int visibility) {
                super.onWindowVisibilityChanged(visibility);
                main.post(()->{feedbackPreferences(); if(visibility==View.VISIBLE) localization();});
            }
        };
        accessibility.addAccessibilityStateChangeListener(accessibilityChanged);
        accessibility.addTouchExplorationStateChangeListener(explorationChanged);
        root=new File(activity.getFilesDir(),"SmudeProjects");
        main.post(() -> {
            if (destroyed) return;
            if (!root.isDirectory() && !root.mkdirs()) emit("ready",Map.of("error",Localizer.text(view.getContext(),"Could not open collection storage")));
            else { localization(); emit("ready",Map.of("root",root.getAbsolutePath())); environment(); }
        });
    }
    @Override public View view() { return view; }
    @Override public void setProp(String name,Object value) {}
    private void emit(String name,Map<String,Object> data) { if (!destroyed) context.emit(name,data); }
    @Override public void command(String name,Object payload) {
        if (destroyed) return;
        Map<?,?> args=payload instanceof Map<?,?> map ? map : Map.of();
        if(name.equals("localization")){ dateRequests=args.get("dates") instanceof List<?> dates?dates:List.of(); localization(); return; }
        if(name.equals("open_settings")) { activity.startActivity(new android.content.Intent(android.provider.Settings.ACTION_APPLICATION_DETAILS_SETTINGS, android.net.Uri.parse("package:"+activity.getPackageName()))); return; }
        if(name.equals("feedback_preferences")){feedbackPreferences();return;}
        long id=args.get("request_id") instanceof Number n ? n.longValue() : 0;
        switch(name) {
            case "import", "import_files" -> importImages(id,name.equals("import_files"));
            case "share" -> share(id,args);
            default -> { }
        }
    }
    /** Build and device facts. Android cannot tell test tracks apart at runtime:
     *  test builds are made with SMUDE_TESTING=1, which Rust reads at compile time. */
    private void environment() {
        String version="", build="";
        try {
            android.content.pm.PackageInfo info=activity.getPackageManager().getPackageInfo(activity.getPackageName(),0);
            version=info.versionName==null?"":info.versionName;
            build=Long.toString(info.getLongVersionCode());
        } catch (Exception ignored) { }
        android.util.DisplayMetrics metrics=activity.getResources().getDisplayMetrics();
        emit("environment",Map.of(
            "testing",false,
            "app",Map.of("version",version,"build",build,"bundle",activity.getPackageName()),
            "device",Map.of("model",Build.MODEL,"manufacturer",Build.MANUFACTURER,"system","Android",
                "system_version",Build.VERSION.RELEASE,"screen",List.of(metrics.widthPixels,metrics.heightPixels,metrics.density))));
    }
    private void localization() {
        android.os.LocaleList preferences=activity.getResources().getConfiguration().getLocales();
        String language="en";
        for(int i=0;i<preferences.size();i++) {
            String candidate=preferences.get(i).getLanguage();
            if(candidate.equals("en")||candidate.equals("de")||candidate.equals("pt")){language=candidate;break;}
        }
        Map<String,String> dates=new java.util.HashMap<>();
        for(Object value:dateRequests) if(value instanceof Number n && n.longValue()>0) {
            java.util.Date date=new java.util.Date(n.longValue());
            dates.put(Long.toString(n.longValue()),android.text.format.DateFormat.getMediumDateFormat(activity).format(date)+" · "+android.text.format.DateFormat.getTimeFormat(activity).format(date));
        }
        emit("localization",Map.of("language",language,"dates",dates));
    }
    private void importImages(long id,boolean files) {
        if (importing) { emit("imported",Map.of("request_id",id,"error",Localizer.text(view.getContext(),"An image picker is already open"))); return; }
        importing=true;
        try {
            activity.pickImages(files,uris -> {
                if (destroyed) return;
                if (uris.isEmpty()) { importing=false; emit("imported",Map.of("request_id",id,"cancelled",true)); return; }
                io.execute(() -> copySelection(id,uris));
            });
        } catch (Exception e) { importing=false; emit("imported",Map.of("request_id",id,"error",Localizer.text(view.getContext(),"Could not open the image picker"))); }
    }
    private void copySelection(long id,List<Uri> uris) {
        File directory=new File(root,"staging/"+UUID.randomUUID());
        List<String> result=new ArrayList<>();
        try {
            if (!directory.mkdirs()) throw new java.io.IOException(Localizer.text(view.getContext(),"Could not create the import folder"));
            for (int index=0;index<uris.size();index++) {
                if (destroyed) throw new java.io.IOException(Localizer.text(view.getContext(),"Import interrupted"));
                // Copy while the provider's temporary grant is valid. Cloud-backed
                // images may take time; this never holds the UI thread.
                File original=new File(directory,String.format(java.util.Locale.ROOT,"%04d.original",index+1));
                try (InputStream input=activity.getContentResolver().openInputStream(uris.get(index));
                     FileOutputStream output=new FileOutputStream(original)) {
                    if (input==null) throw new java.io.IOException(Localizer.text(view.getContext(),"Could not read the image"));
                    byte[] chunk=new byte[64*1024];
                    int count; while((count=input.read(chunk))!=-1) output.write(chunk,0,count);
                    output.getFD().sync();
                }
                File image;
                if(isUprightJpeg(original)) {
                    // The original JPEG already has canonical orientation: keep
                    // its compressed bytes exactly and avoid a full PNG encode.
                    image=new File(directory,String.format(java.util.Locale.ROOT,"%04d.jpg",index+1));
                    if(!original.renameTo(image)) throw new java.io.IOException(Localizer.text(view.getContext(),"Could not import the image"));
                } else {
                    image=new File(directory,String.format(java.util.Locale.ROOT,"%04d.png",index+1));
                    normalize(original,image);
                }
                result.add(image.getAbsolutePath());
                int copied=index+1;
                main.post(() -> emit("import_progress",Map.of("request_id",id,"completed",copied,"total",uris.size())));
            }
            if (destroyed) throw new java.io.IOException(Localizer.text(view.getContext(),"Import interrupted"));
            main.post(() -> { importing=false; emit("imported",Map.of("request_id",id,"paths",result)); });
        } catch (Exception | OutOfMemoryError e) {
            removeDirectory(directory);
            String message=e instanceof OutOfMemoryError ? Localizer.text(view.getContext(),"Not enough memory for this image.") : Localizer.text(view.getContext(),"Could not import the selected image. Check the selection.");
            main.post(() -> { importing=false; emit("imported",Map.of("request_id",id,"error",message)); });
        }
    }
    static boolean isUprightJpeg(File source) throws java.io.IOException {
        try(FileInputStream input=new FileInputStream(source)) {
            if(input.read()!=0xff || input.read()!=0xd8) return false;
        }
        int orientation=new ExifInterface(source.getPath()).getAttributeInt(ExifInterface.TAG_ORIENTATION,ExifInterface.ORIENTATION_NORMAL);
        return orientation==ExifInterface.ORIENTATION_NORMAL || orientation==ExifInterface.ORIENTATION_UNDEFINED;
    }
    /** ImageDecoder applies EXIF orientation, including reflected orientations and HEIC. */
    static void normalize(File source,File destination) throws java.io.IOException {
        Bitmap bitmap=ImageDecoder.decodeBitmap(ImageDecoder.createSource(source),(decoder,info,src) -> {
            decoder.setAllocator(ImageDecoder.ALLOCATOR_SOFTWARE);
            decoder.setTargetColorSpace(ColorSpace.get(ColorSpace.Named.SRGB));
        });
        if(bitmap.hasAlpha()) {
            Bitmap opaque=Bitmap.createBitmap(bitmap.getWidth(),bitmap.getHeight(),Bitmap.Config.ARGB_8888);
            Canvas canvas=new Canvas(opaque); canvas.drawColor(Color.WHITE); canvas.drawBitmap(bitmap,0,0,null);
            bitmap.recycle(); bitmap=opaque;
        }
        File temporary=new File(destination.getParentFile(),destination.getName()+".partial");
        try {
            try (FileOutputStream stream=new FileOutputStream(temporary)) {
                if (!bitmap.compress(Bitmap.CompressFormat.PNG,100,stream)) throw new java.io.IOException("Could not save the image");
                stream.getFD().sync();
            }
            if (!temporary.renameTo(destination)) throw new java.io.IOException("Could not finish saving the image");
        } finally { bitmap.recycle(); if (temporary.exists()) temporary.delete(); }
    }
    private void share(long id,Map<?,?> args) {
        try {
            if (!(args.get("paths") instanceof List<?> paths) || paths.isEmpty()) throw new IllegalArgumentException();
            Intent send=shareIntent(activity,root,paths,"application/pdf".equals(args.get("mime")) ? "application/pdf" : "image/jpeg");
            Intent chooser=Intent.createChooser(send,Localizer.text(view.getContext(),"Share"));
            chooser.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
            activity.startActivity(chooser);
            // Android does not report whether the target actually delivered a
            // message/file. Opening the chooser is the only claim made here.
            emit("shared",Map.of("request_id",id,"opened",true));
        } catch (Exception e) { emit("shared",Map.of("request_id",id,"error",Localizer.text(view.getContext(),"Could not open files for sharing"))); }
    }
    static Intent shareIntent(Context context,File root,List<?> paths,String mime) throws java.io.IOException {
        if(paths.isEmpty()) throw new IllegalArgumentException("Keine Exportdateien");
        File exports=new File(root,"exports").getCanonicalFile();
        ArrayList<Uri> uris=new ArrayList<>();
        for(Object value:paths) {
            if(!(value instanceof String path)) throw new IllegalArgumentException("Invalid file path");
            File file=new File(path).getCanonicalFile();
            if(!file.toPath().startsWith(exports.toPath()) || !file.isFile() || file.length()==0) throw new IllegalArgumentException("Keine fertige Exportdatei");
            uris.add(FileProvider.getUriForFile(context,context.getPackageName()+".exports",file));
        }
        Intent send=new Intent(uris.size()==1 ? Intent.ACTION_SEND : Intent.ACTION_SEND_MULTIPLE).setType(mime);
        if(uris.size()==1) send.putExtra(Intent.EXTRA_STREAM,uris.get(0));
        else send.putParcelableArrayListExtra(Intent.EXTRA_STREAM,uris);
        ClipData clip=ClipData.newUri(context.getContentResolver(),"Smude",uris.get(0));
        for(int i=1;i<uris.size();i++) clip.addItem(new ClipData.Item(uris.get(i)));
        send.setClipData(clip); send.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
        return send;
    }
    private static void removeDirectory(File directory) {
        File[] children=directory.listFiles();
        if (children!=null) for(File child:children) { if(child.isDirectory()) removeDirectory(child); else child.delete(); }
        directory.delete();
    }
    private void feedbackPreferences() {
        if(destroyed) return;
        int info=3000, action=6000;
        if(Build.VERSION.SDK_INT>=29) {
            info=accessibility.getRecommendedTimeoutMillis(info,AccessibilityManager.FLAG_CONTENT_TEXT);
            action=accessibility.getRecommendedTimeoutMillis(action,
                AccessibilityManager.FLAG_CONTENT_TEXT|AccessibilityManager.FLAG_CONTENT_CONTROLS);
        }
        emit("feedback_preferences",Map.of("info_ms",info,"action_ms",action,
            "persistent_action",accessibility.isTouchExplorationEnabled(),
            "foreground",view.getWindowVisibility()==View.VISIBLE));
    }
    @Override public void destroy() {
        if(activity.captureEvents==captureListener)activity.captureEvents=null;
        destroyed=true;io.shutdown();
        accessibility.removeAccessibilityStateChangeListener(accessibilityChanged);
        accessibility.removeTouchExplorationStateChangeListener(explorationChanged);
    }
}
