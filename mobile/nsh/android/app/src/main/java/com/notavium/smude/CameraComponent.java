package com.notavium.smude;

import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.hardware.display.DisplayManager;
import android.os.Handler;
import android.os.Looper;
import android.os.SystemClock;
import android.util.Size;
import android.view.Surface;
import android.view.OrientationEventListener;
import android.view.View;
import androidx.annotation.NonNull;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.Camera;
import androidx.camera.core.TorchState;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageCaptureException;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.core.UseCaseGroup;
import androidx.camera.core.ViewPort;
import androidx.camera.core.resolutionselector.ResolutionSelector;
import androidx.camera.core.resolutionselector.ResolutionStrategy;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.DefaultLifecycleObserver;
import androidx.lifecycle.Lifecycle;
import androidx.lifecycle.LifecycleOwner;
import com.longbridge.nsh.NshComponent;
import com.longbridge.nsh.NshComponentContext;
import org.json.JSONArray;
import org.json.JSONObject;
import java.io.File;
import java.io.FileOutputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Owns CameraX resources, never app chrome or contour drawing. Preview and
 * analysis share one ViewPort; inference sees a physically rotated crop of it.
 * Consequently the Rust canvas can use the same normalized aspect-fill mapping
 * on either OS, including after device rotation and for two-page spreads.
 */
final class CameraComponent implements NshComponent, DefaultLifecycleObserver {
    private static final long FRAME_INTERVAL_MS = 140;
    private static final long MAX_RESULT_AGE_MS = 1200;
    private final MainActivity activity;
    private final NshComponentContext context;
    private final PreviewView preview;
    private final Handler main = new Handler(Looper.getMainLooper());
    private final ExecutorService analysisExecutor = Executors.newSingleThreadExecutor();
    private ProcessCameraProvider provider;
    private Preview previewUseCase;
    private ImageAnalysis analysis;
    private ImageCapture capture;
    private Camera boundCamera;
    private boolean torchRequested;
    private final androidx.lifecycle.Observer<Integer> torchObserver = state -> publishTorch();
    private volatile boolean destroyed;
    private volatile int generation;
    private boolean active = true;
    private boolean permission;
    private boolean permissionPending;
    private boolean providerPending;
    private boolean capturing;
    private boolean stopAfterCapture;
    private long captureAfterMs;
    private static final java.util.concurrent.atomic.AtomicInteger pendingPhotos=new java.util.concurrent.atomic.AtomicInteger();
    private static final java.util.concurrent.ExecutorService photoSaves=Executors.newSingleThreadExecutor();
    private boolean ready;
    private int boundRotation = Surface.ROTATION_0;
    private final CameraOrientation orientation = new CameraOrientation();
    private final OrientationEventListener orientationListener;
    private volatile int analysisTurns;
    private volatile int expectedImageRotation;
    private int stableFrames;
    private int detectedPages;
    private long lastValidFrameMs;
    private long lastAnalysisMs;
    private JSONArray previousPolygons;
    private double[] lastGoodScene;
    private Map<String,Object> lastGoodEvent;
    private boolean holding;
    private ByteBuffer inferenceBuffer;
    private final DisplayManager.DisplayListener displayListener = new DisplayManager.DisplayListener() {
        @Override public void onDisplayAdded(int id) {}
        @Override public void onDisplayRemoved(int id) {}
        @Override public void onDisplayChanged(int id) {
            if (preview.getDisplay() != null && preview.getDisplay().getDisplayId() == id
                && preview.getDisplay().getRotation() != boundRotation) {
                // Also catches 180-degree rotations, which do not resize the View.
                stop(); start();
            }
        }
    };
    private final Runnable expiry = new Runnable() {
        @Override public void run() {
            if (destroyed) return;
            if ((ready || stableFrames > 0) && SystemClock.elapsedRealtime() - lastValidFrameMs > (holding ? DetectionHold.GRACE_MS : MAX_RESULT_AGE_MS)) {
                invalidate(Localizer.text(activity,"Place the page in the frame"));
            }
            main.postDelayed(this, 200);
        }
    };

    CameraComponent(MainActivity activity, NshComponentContext context) {
        this.activity = activity;
        this.context = context;
        orientationListener = new OrientationEventListener(activity) {
            @Override public void onOrientationChanged(int degrees) {
                if (destroyed || !orientation.observe(degrees)) return;
                analysisTurns = orientation.analysisTurns(boundRotation);
                generation++;
                if (capture != null && !capturing) capture.setTargetRotation(orientation.captureRotation(boundRotation));
                invalidate(Localizer.text(activity,"Place the page in the frame"));
            }
        };
        preview = new PreviewView(activity);
        preview.setImplementationMode(PreviewView.ImplementationMode.COMPATIBLE);
        preview.setScaleType(PreviewView.ScaleType.FILL_CENTER);
        preview.addOnAttachStateChangeListener(new View.OnAttachStateChangeListener() {
            @Override public void onViewAttachedToWindow(View v) { start(); }
            @Override public void onViewDetachedFromWindow(View v) { stop(); }
        });
        preview.addOnLayoutChangeListener((v,l,t,r,b,ol,ot,or,ob) -> {
            if (r-l != or-ol || b-t != ob-ot) { stop(); start(); }
        });
        activity.getLifecycle().addObserver(this);
        activity.getSystemService(DisplayManager.class).registerDisplayListener(displayListener,main);
        main.post(expiry);
        main.post(this::start);
    }

    @Override public View view() { return preview; }
    @Override public void setProp(String name, Object value) {
        if (name.equals("active")) {
            active = !Boolean.FALSE.equals(value);
            if (active) start(); else { torchRequested = false; stop(); }
        }
    }
    @Override public void command(String name, Object args) {
        if (name.equals("capture")) takePhoto(args);
        else if (name.equals("retry")) start();
        else if (name.equals("torch_on")) setTorch(true);
        else if (name.equals("torch_off")) setTorch(false);
    }
    @Override public void onStart(@NonNull LifecycleOwner owner) { start(); }
    @Override public void onStop(@NonNull LifecycleOwner owner) { torchRequested = false; stop(); }

    private void publishTorch() {
        boolean available = boundCamera != null && boundCamera.getCameraInfo().hasFlashUnit();
        boolean on = available && Integer.valueOf(TorchState.ON).equals(boundCamera.getCameraInfo().getTorchState().getValue());
        if (!destroyed) context.emit("torch", Map.of("available", available, "on", on));
    }
    private void setTorch(boolean on) {
        torchRequested = on && visible() && boundCamera != null && boundCamera.getCameraInfo().hasFlashUnit();
        if (boundCamera == null) { publishTorch(); return; }
        Camera target = boundCamera;
        var result = target.getCameraControl().enableTorch(torchRequested);
        result.addListener(() -> {
            if (destroyed || boundCamera != target) return;
            try { result.get(); }
            catch (Exception e) { torchRequested = false; }
            publishTorch();
        }, ContextCompat.getMainExecutor(activity));
    }

    private boolean visible() {
        return !destroyed && active && preview.isAttachedToWindow()
            && preview.getWidth() > 0 && preview.getHeight() > 0
            && activity.getLifecycle().getCurrentState().isAtLeast(Lifecycle.State.STARTED);
    }
    private void start() {
        if (!visible() || capture != null || permissionPending || providerPending) return;
        orientationListener.enable();
        if (!permission) {
            permissionPending = true;
            activity.requestCamera(granted -> {
                permissionPending = false;
                if (destroyed) return;
                permission = granted;
                if (granted) start(); else invalidate(Localizer.text(activity,"Allow camera access in Settings"),"permission");
            });
            return;
        }
        if (provider == null) {
            providerPending = true;
            var future = ProcessCameraProvider.getInstance(activity);
            future.addListener(() -> {
                providerPending = false;
                if (destroyed) return;
                try { provider = future.get(); start(); }
                catch (Exception e) { invalidate(Localizer.text(activity,"Could not start the camera"),"unavailable"); }
            }, ContextCompat.getMainExecutor(activity));
            return;
        }
        ViewPort viewPort = preview.getViewPort();
        if (viewPort == null) return;
        int rotation = preview.getDisplay() == null ? Surface.ROTATION_0 : preview.getDisplay().getRotation();
        boundRotation = rotation;
        analysisTurns = orientation.analysisTurns(rotation);
        previewUseCase = new Preview.Builder().setTargetRotation(rotation).build();
        previewUseCase.setSurfaceProvider(preview.getSurfaceProvider());
        capture = new ImageCapture.Builder().setTargetRotation(orientation.captureRotation(rotation))
            .setResolutionSelector(new ResolutionSelector.Builder()
                .setAllowedResolutionMode(ResolutionSelector.PREFER_HIGHER_RESOLUTION_OVER_CAPTURE_RATE)
                .setResolutionStrategy(ResolutionStrategy.HIGHEST_AVAILABLE_STRATEGY).build())
            .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
            .setFlashMode(ImageCapture.FLASH_MODE_OFF).setJpegQuality(97).build();
        analysis = new ImageAnalysis.Builder().setTargetRotation(rotation)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setResolutionSelector(new ResolutionSelector.Builder().setResolutionStrategy(
                new ResolutionStrategy(new Size(640,480), ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER)).build())
            .build();
        analysis.setAnalyzer(analysisExecutor, this::analyze);
        try {
            boundCamera = provider.bindToLifecycle(activity, CameraSelector.DEFAULT_BACK_CAMERA,
                new UseCaseGroup.Builder().setViewPort(viewPort).addUseCase(previewUseCase)
                    .addUseCase(capture).addUseCase(analysis).build());
            expectedImageRotation = boundCamera.getCameraInfo().getSensorRotationDegrees(rotation);
            boundCamera.getCameraInfo().getTorchState().observe(activity, torchObserver);
            setTorch(torchRequested);
            invalidate(Localizer.text(activity,"Place the page in the frame"));
        } catch (Exception e) {
            stop();
            invalidate(Localizer.text(activity,"No camera available"),"unavailable");
        }
    }

    private void stop() {
        orientationListener.disable();
        generation++;
        if (analysis != null) analysis.clearAnalyzer();
        if(capturing){stopAfterCapture=true;return;}
        stopAfterCapture=false;
        if (boundCamera != null) {
            boundCamera.getCameraInfo().getTorchState().removeObserver(torchObserver);
            boundCamera.getCameraControl().enableTorch(false);
            boundCamera = null;
        }
        if (provider != null && previewUseCase != null && capture != null && analysis != null) {
            provider.unbind(previewUseCase, capture, analysis);
        }
        previewUseCase = null;
        capture = null;
        analysis = null;
        publishTorch();
        if (!destroyed) invalidate(Localizer.text(activity,"Place the page in the frame"));
    }

    private void analyze(ImageProxy image) {
        long frameMs = SystemClock.elapsedRealtime();
        int revision = generation;
        int turns = analysisTurns;
        Bitmap rgba = null, oriented = null, scaled = null;
        try {
            if (destroyed || frameMs-lastAnalysisMs < FRAME_INTERVAL_MS
                || image.getImageInfo().getRotationDegrees() != expectedImageRotation) return;
            lastAnalysisMs = frameMs;
            ImageProxy.PlaneProxy plane = image.getPlanes()[0];
            ByteBuffer buffer = plane.getBuffer();
            Rect crop = image.getCropRect();
            int width = crop.width(), height = crop.height();
            // Read the crop directly, including row/pixel padding. CameraX RGBA
            // planes expose R,G,B,A; integer ARGB avoids byte-order assumptions.
            int[] pixels = new int[width*height];
            int base = buffer.position();
            for (int y=0; y<height; y++) for (int x=0; x<width; x++) {
                int offset = base+(crop.top+y)*plane.getRowStride()+(crop.left+x)*plane.getPixelStride();
                int r = buffer.get(offset)&255, g = buffer.get(offset+1)&255, b = buffer.get(offset+2)&255;
                pixels[y*width+x] = 0xff000000 | (r<<16) | (g<<8) | b;
            }
            rgba = Bitmap.createBitmap(pixels,width,height,Bitmap.Config.ARGB_8888);
            Matrix rotate = new Matrix();
            rotate.postRotate(image.getImageInfo().getRotationDegrees());
            oriented = Bitmap.createBitmap(rgba,0,0,width,height,rotate,false);
            float ratio = Math.min(1f,640f/Math.max(oriented.getWidth(),oriented.getHeight()));
            scaled = ratio < 1 ? Bitmap.createScaledBitmap(oriented,
                Math.max(2,Math.round(oriented.getWidth()*ratio)), Math.max(2,Math.round(oriented.getHeight()*ratio)),true) : oriented;
            width = scaled.getWidth(); height = scaled.getHeight();
            int size = width*height*4;
            if (inferenceBuffer == null || inferenceBuffer.capacity() != size) inferenceBuffer = ByteBuffer.allocateDirect(size);
            inferenceBuffer.clear();
            // Explicit RGBA packing makes the JNI contract independent of
            // Android Bitmap native endian/premultiplication details.
            pixels = new int[width*height];
            scaled.getPixels(pixels,0,width,0,0,width,height);
            for (int pixel:pixels) inferenceBuffer.put((byte)(pixel>>16)).put((byte)(pixel>>8)).put((byte)pixel).put((byte)255);
            inferenceBuffer.rewind();
            JSONObject result = new JSONObject(CameraInference.detect(inferenceBuffer,width,height,width*4,turns));
            main.post(() -> accept(result,revision,frameMs));
        } catch (Exception | LinkageError e) {
            main.post(() -> { if (!destroyed && generation==revision) invalidate(Localizer.text(activity,"Page detection unavailable"),"detector"); });
        } finally {
            if (scaled != null && scaled != oriented && scaled != rgba) scaled.recycle();
            if (oriented != null && oriented != rgba) oriented.recycle();
            if (rgba != null) rgba.recycle();
            image.close();
        }
    }

    private void accept(JSONObject result, int revision, long frameMs) {
        if (!visible() || revision != generation || capturing) return;
        JSONArray polygons = result.optJSONArray("polygons");
        if (SystemClock.elapsedRealtime()-frameMs > MAX_RESULT_AGE_MS) {
            invalidate(Localizer.text(activity,"Hold the camera steady")); return;
        }
        boolean valid = result.optBoolean("valid",false) && polygons != null && polygons.length()>0;
        double[] scene=scene(result.optJSONArray("scene"));
        double difference=DetectionHold.sceneDifference(lastGoodScene,scene);
        boolean moved=ready && (difference>DetectionHold.SCENE_CHANGE
            || (valid && contourDistance(previousPolygons,polygons)>.06));
        boolean hardFailure=result.has("error") || moved;
        if(!valid && DetectionHold.retain(ready,SystemClock.elapsedRealtime(),lastValidFrameMs,difference,hardFailure)
            && lastGoodEvent!=null) {
            // One bad measurement does not erase a stable page. Keep exactly
            // the last trustworthy contour; never extend its age on a miss.
            holding=true;
            Map<String,Object> retained=new LinkedHashMap<>(lastGoodEvent);
            retained.put("ready",lastValidFrameMs>=captureAfterMs && pendingPhotos.get()<3); retained.put("capturing",false);
            context.emit("status",retained);
            return;
        }
        holding=false;
        if (valid) {
            analysisTurns = result.optInt("analysis_turns",analysisTurns) % 4;
            stableFrames = !moved && (ready || similar(previousPolygons,polygons)) ? stableFrames+1 : 1;
            lastValidFrameMs = frameMs;
            detectedPages = result.optInt("pages",1);
            previousPolygons = polygons;
        } else {
            stableFrames = 0;
            previousPolygons = null;
            detectedPages = 0;
            lastGoodEvent=null; lastGoodScene=null;
        }
        ready = valid && stableFrames >= 2;
        Map<String,Object> event = toMap(result);
        event.remove("scene"); // only the platform latch needs the tiny fingerprint
        event.put("ready",ready && !capturing && lastValidFrameMs>=captureAfterMs && pendingPhotos.get()<3);
        event.put("capturing",capturing);
        if (result.has("error")) { event.put("message",Localizer.text(activity,"Page detection unavailable")); event.put("reason","detector"); }
        else if (valid && !ready) event.put("message",Localizer.text(activity,"Hold the camera steady"));
        if(ready) { lastGoodEvent=new LinkedHashMap<>(event); lastGoodScene=scene; }
        context.emit("status",event);
    }

    private static double[] scene(JSONArray array) {
        if(array==null || array.length()!=64) return null;
        double[] result=new double[64];
        for(int i=0;i<64;i++) result[i]=array.optDouble(i,Double.NaN);
        return result;
    }

    private static boolean similar(JSONArray a, JSONArray b) {
        return contourDistance(a,b)<.025;
    }
    private static double contourDistance(JSONArray a,JSONArray b) {
        if (a == null || b==null || a.length()!=b.length()) return Double.POSITIVE_INFINITY;
        double distance = 0;
        int count = 0;
        try {
            for (int page=0;page<a.length();page++) {
                JSONArray pa=a.getJSONArray(page),pb=b.getJSONArray(page);
                if (pa.length()!=pb.length()) return Double.POSITIVE_INFINITY;
                for (int point=0;point<pa.length();point++) {
                    JSONArray p=pa.getJSONArray(point),q=pb.getJSONArray(point);
                    distance += Math.hypot(p.getDouble(0)-q.getDouble(0),p.getDouble(1)-q.getDouble(1));
                    count++;
                }
            }
        } catch (Exception e) { return Double.POSITIVE_INFINITY; }
        return count>0 ? distance/count : Double.POSITIVE_INFINITY;
    }

    private void takePhoto(Object arguments) {
        Map<?,?> args=arguments instanceof Map<?,?> map?map:Map.of();
        if(!(args.get("request_id") instanceof Number number)||!(args.get("path") instanceof String path))return;
        long request=number.longValue();
        if (!visible() || capture==null || capturing || !ready || pendingPhotos.get()>=3 || lastValidFrameMs<captureAfterMs || (Boolean.TRUE.equals(args.get("single_only")) && detectedPages!=1)
            || SystemClock.elapsedRealtime()-lastValidFrameMs > (holding ? DetectionHold.GRACE_MS : MAX_RESULT_AGE_MS)) {
            activity.captureEvent("capture_failed",Map.of("request_id",request,"reason","not_ready"));
            context.emit("capture_failed",Map.of("request_id",request));
            invalidate(Localizer.text(activity,"Include the entire page in the frame"));return;
        }
        int pages=detectedPages,correction=(analysisTurns%4)*90;
        capture.setTargetRotation(boundRotation);
        capturing=true;pendingPhotos.incrementAndGet();
        activity.captureEvent("capture_started",Map.of("request_id",request,"pages",pages,"rotation",correction));
        Object previewId=lastGoodEvent==null?null:lastGoodEvent.get("preview_frame");
        context.emit("capture_started",Map.of("request_id",request,"pages",pages,"rotation",correction,
            "preview_frame",previewId instanceof Number?((Number)previewId).longValue():0L));
        Map<String,Object> busy=lastGoodEvent==null?new LinkedHashMap<>():new LinkedHashMap<>(lastGoodEvent);
        busy.put("ready",false);busy.put("capturing",true);context.emit("status",busy);
        File destination=new File(path);
        File temporary=new File(destination.getParentFile(),destination.getName()+".pending");
        try {
            capture.takePicture(ContextCompat.getMainExecutor(activity),new ImageCapture.OnImageCapturedCallback(){
                @Override public void onCaptureSuccess(@NonNull ImageProxy photo){
                    byte[] bytes;
                    int rotation=photo.getImageInfo().getRotationDegrees();
                    try {
                        java.nio.ByteBuffer buffer=photo.getPlanes()[0].getBuffer().duplicate();
                        bytes=new byte[buffer.remaining()];buffer.get(bytes);
                    }catch(Exception|OutOfMemoryError e){
                        pendingPhotos.decrementAndGet();
                        activity.captureEvent("capture_failed",Map.of("request_id",request));
                        return;
                    }finally{photo.close();releaseShutter();}
                    photoSaves.execute(()->{
                        boolean saved=false;
                        try {
                            File parent=destination.getParentFile();
                            if(parent==null||(!parent.isDirectory()&&!parent.mkdirs()))throw new java.io.IOException("Capture directory unavailable");
                            FullResolutionPhoto.writeJpeg(java.nio.ByteBuffer.wrap(bytes),rotation,temporary);
                            try(FileOutputStream out=new FileOutputStream(temporary,true)){out.getFD().sync();}
                            saved=temporary.renameTo(destination);
                        }catch(Exception e){saved=false;}
                        finally{temporary.delete();}
                        pendingPhotos.decrementAndGet();
                        activity.captureEvent(saved?"capture_saved":"capture_failed",Map.of("request_id",request,"source_rotation",rotation));
                    });
                }
                @Override public void onError(@NonNull ImageCaptureException error){
                    pendingPhotos.decrementAndGet();releaseShutter();temporary.delete();
                    activity.captureEvent("capture_failed",Map.of("request_id",request));
                }
            });
        }catch(RuntimeException e){pendingPhotos.decrementAndGet();releaseShutter();activity.captureEvent("capture_failed",Map.of("request_id",request));}
    }
    private void releaseShutter(){
        capturing=false;captureAfterMs=SystemClock.elapsedRealtime();
        if(stopAfterCapture||destroyed||!active){stop();if(!destroyed&&active&&visible())start();return;}
        if(!destroyed){
            Map<String,Object> status=lastGoodEvent==null?new LinkedHashMap<>():new LinkedHashMap<>(lastGoodEvent);
            status.put("ready",ready && lastValidFrameMs>=captureAfterMs && pendingPhotos.get()<3);status.put("capturing",false);context.emit("status",status);
        }
    }

    private void invalidate(String message) { invalidate(message, ""); }
    private void invalidate(String message, String reason) {
        ready=false; stableFrames=0; previousPolygons=null; detectedPages=0;
        holding=false; lastGoodScene=null; lastGoodEvent=null;
        if (!destroyed) context.emit("status",Map.of("ready",false,"capturing",capturing,"message",message,"reason",reason,
            "width",0,"height",0,"polygons",List.of()));
    }
    private static Map<String,Object> toMap(JSONObject json) {
        Map<String,Object> values=new LinkedHashMap<>();
        json.keys().forEachRemaining(key -> values.put(key,toValue(json.opt(key))));
        return values;
    }
    private static Object toValue(Object value) {
        if (value instanceof JSONObject object) return toMap(object);
        if (value instanceof JSONArray array) {
            List<Object> list=new ArrayList<>();
            for (int i=0;i<array.length();i++) list.add(toValue(array.opt(i)));
            return list;
        }
        return value==JSONObject.NULL ? null : value;
    }
    @Override public void destroy() {
        if (destroyed) return;
        destroyed=true;
        main.removeCallbacks(expiry);
        activity.getLifecycle().removeObserver(this);
        activity.getSystemService(DisplayManager.class).unregisterDisplayListener(displayListener);
        stop();
        analysisExecutor.shutdown();
    }
}
