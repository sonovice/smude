package com.notavium.smude;

import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.media.ExifInterface;
import android.net.Uri;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.core.app.ActivityScenario;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import androidx.test.platform.app.InstrumentationRegistry;
import org.junit.Test;
import org.junit.runner.RunWith;
import java.io.File;
import java.io.FileOutputStream;
import java.nio.file.Files;
import java.util.List;
import static org.junit.Assert.*;

@RunWith(AndroidJUnit4.class)
public final class ServicesInstrumentedTest {
    private Context context() { return InstrumentationRegistry.getInstrumentation().getTargetContext(); }
    private File scratch(String name) throws Exception {
        File dir=new File(context().getCacheDir(),"service-tests"); assertTrue(dir.isDirectory() || dir.mkdirs());
        return new File(dir,name);
    }
    @Test public void nativePhotoAndFilePickersCancelWithoutSelectingAnything() throws Exception {
        try(ActivityScenario<MainActivity> scenario=ActivityScenario.launch(MainActivity.class)) {
            for(boolean files:new boolean[]{false,true}) {
                CountDownLatch finished=new CountDownLatch(1);
                AtomicReference<List<Uri>> result=new AtomicReference<>();
                scenario.onActivity(activity -> activity.pickImages(files,uris -> {result.set(uris);finished.countDown();}));
                boolean external=false;
                for(int n=0;n<40;n++) {
                    var window=InstrumentationRegistry.getInstrumentation().getUiAutomation().getRootInActiveWindow();
                    if(window!=null && window.getPackageName()!=null && !context().getPackageName().contentEquals(window.getPackageName())) { external=true; break; }
                    Thread.sleep(100);
                }
                assertTrue("Native system picker must actually open",external);
                try(var command=InstrumentationRegistry.getInstrumentation().getUiAutomation().executeShellCommand("input keyevent 4")) {
                    // Closing the picker is a cancellation, never a successful empty import.
                    byte[] buffer=new byte[64];
                    try(var input=new java.io.FileInputStream(command.getFileDescriptor())) { while(input.read(buffer)!=-1) {} }
                }
                assertTrue("Picker must return a result after cancellation",finished.await(5,TimeUnit.SECONDS));
                assertTrue(result.get().isEmpty());
            }
        }
    }
    @Test public void normalizationAppliesExifRotationWithoutReducingResolution() throws Exception {
        File source=scratch("rotated.jpg"),output=scratch("upright.png");
        Bitmap bitmap=Bitmap.createBitmap(80,40,Bitmap.Config.ARGB_8888); bitmap.eraseColor(Color.WHITE);
        for(int x=0;x<40;x++) for(int y=0;y<40;y++) bitmap.setPixel(x,y,Color.RED);
        try(FileOutputStream stream=new FileOutputStream(source)) { assertTrue(bitmap.compress(Bitmap.CompressFormat.JPEG,100,stream)); }
        bitmap.recycle();
        ExifInterface exif=new ExifInterface(source.getPath()); exif.setAttribute(ExifInterface.TAG_ORIENTATION,Integer.toString(ExifInterface.ORIENTATION_ROTATE_90)); exif.saveAttributes();
        byte[] original=Files.readAllBytes(source.toPath());
        assertFalse(ServicesComponent.isUprightJpeg(source));
        ServicesComponent.normalize(source,output);
        Bitmap normalized=BitmapFactory.decodeFile(output.getPath());
        assertEquals(40,normalized.getWidth()); assertEquals(80,normalized.getHeight());
        assertTrue(Color.red(normalized.getPixel(20,10))>240); assertTrue(Color.green(normalized.getPixel(20,10))<20);
        assertTrue(Color.green(normalized.getPixel(20,70))>240);
        assertArrayEquals(original,Files.readAllBytes(source.toPath())); normalized.recycle();
    }
    @Test public void uprightJpegUsesOriginalBytePath() throws Exception {
        File source=scratch("upright.jpg");
        Bitmap bitmap=Bitmap.createBitmap(40,30,Bitmap.Config.ARGB_8888); bitmap.eraseColor(Color.WHITE);
        try(FileOutputStream stream=new FileOutputStream(source)) { assertTrue(bitmap.compress(Bitmap.CompressFormat.JPEG,95,stream)); } bitmap.recycle();
        assertTrue(ServicesComponent.isUprightJpeg(source));
    }
    @Test public void cameraJpegKeepsFullPixelDimensionsAndAllFourOrientations() throws Exception {
        Bitmap original=Bitmap.createBitmap(321,239,Bitmap.Config.ARGB_8888);
        original.eraseColor(Color.WHITE);
        original.setPixel(0,0,Color.BLACK);
        original.setPixel(320,238,Color.BLACK);
        java.io.ByteArrayOutputStream bytes=new java.io.ByteArrayOutputStream();
        assertTrue(original.compress(Bitmap.CompressFormat.JPEG,100,bytes));
        original.recycle();
        for(int rotation:new int[]{0,90,180,270}) {
            File file=scratch("camera-full-"+rotation+".jpg");
            java.nio.ByteBuffer source=java.nio.ByteBuffer.wrap(bytes.toByteArray());
            FullResolutionPhoto.writeJpeg(source,rotation,file);
            assertEquals(0,source.position());
            BitmapFactory.Options metadata=new BitmapFactory.Options();metadata.inJustDecodeBounds=true;
            BitmapFactory.decodeFile(file.getPath(),metadata);
            assertEquals(321,metadata.outWidth);assertEquals(239,metadata.outHeight);
            ExifInterface exif=new ExifInterface(file.getPath());
            int expected=switch(rotation){case 90->ExifInterface.ORIENTATION_ROTATE_90;case 180->ExifInterface.ORIENTATION_ROTATE_180;case 270->ExifInterface.ORIENTATION_ROTATE_270;default->ExifInterface.ORIENTATION_NORMAL;};
            assertEquals(expected,exif.getAttributeInt(ExifInterface.TAG_ORIENTATION,0));
            Bitmap decoded=BitmapFactory.decodeFile(file.getPath());
            Bitmap reference=BitmapFactory.decodeByteArray(bytes.toByteArray(),0,bytes.size());
            assertTrue("EXIF update must not alter JPEG image samples",decoded.sameAs(reference));
            decoded.recycle();reference.recycle();file.delete();
        }
    }
    @Test public void transparentInputsUseWhitePaper() throws Exception {
        File source=scratch("transparent.png"),output=scratch("white.png");
        Bitmap bitmap=Bitmap.createBitmap(20,30,Bitmap.Config.ARGB_8888); bitmap.eraseColor(Color.TRANSPARENT); bitmap.setPixel(10,15,Color.BLACK);
        try(FileOutputStream stream=new FileOutputStream(source)) { assertTrue(bitmap.compress(Bitmap.CompressFormat.PNG,100,stream)); } bitmap.recycle();
        ServicesComponent.normalize(source,output);
        Bitmap normalized=BitmapFactory.decodeFile(output.getPath());
        assertEquals(Color.WHITE,normalized.getPixel(0,0)); assertEquals(Color.BLACK,normalized.getPixel(10,15)); normalized.recycle();
    }
    @Test public void shareUrisExposeOnlyFinishedExportFilesAndPreserveBytes() throws Exception {
        File root=new File(context().getFilesDir(),"SmudeProjects"),exports=new File(root,"exports/service-tests"); assertTrue(exports.isDirectory() || exports.mkdirs());
        File a=new File(exports,"one.pdf"),b=new File(exports,"two.pdf"); byte[] bytes="%PDF-1.7\nfixture\n".getBytes(java.nio.charset.StandardCharsets.US_ASCII);
        Files.write(a.toPath(),bytes); Files.write(b.toPath(),bytes);
        Intent send=ServicesComponent.shareIntent(context(),root,List.of(a.getPath(),b.getPath()),"application/pdf");
        assertEquals(Intent.ACTION_SEND_MULTIPLE,send.getAction()); assertEquals("application/pdf",send.getType());
        assertEquals(2,send.getClipData().getItemCount()); assertTrue((send.getFlags()&Intent.FLAG_GRANT_READ_URI_PERMISSION)!=0);
        Uri uri=send.getClipData().getItemAt(0).getUri(); assertEquals("content",uri.getScheme());
        try(var input=context().getContentResolver().openInputStream(uri)) { java.io.ByteArrayOutputStream buffer=new java.io.ByteArrayOutputStream(); byte[] chunk=new byte[1024]; int count; while((count=input.read(chunk))!=-1) buffer.write(chunk,0,count); assertArrayEquals(bytes,buffer.toByteArray()); }
        File privateFile=new File(root,"private-service-test.json"); Files.write(privateFile.toPath(),new byte[]{1});
        assertThrows(IllegalArgumentException.class,()->ServicesComponent.shareIntent(context(),root,List.of(privateFile.getPath()),"application/pdf"));
        File empty=new File(exports,"empty.pdf"); Files.write(empty.toPath(),new byte[0]);
        assertThrows(IllegalArgumentException.class,()->ServicesComponent.shareIntent(context(),root,List.of(empty.getPath()),"application/pdf"));
        privateFile.delete(); a.delete(); b.delete(); empty.delete();
    }
}
