package com.gridnotelabs.smude.design;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.Rect;
import android.os.SystemClock;
import android.view.InputDevice;
import android.view.MotionEvent;
import android.view.accessibility.AccessibilityNodeInfo;
import android.accessibilityservice.AccessibilityServiceInfo;
import androidx.test.core.app.ActivityScenario;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;
import java.io.File;
import java.io.FileOutputStream;
import java.nio.file.Files;
import org.json.JSONArray;
import org.json.JSONObject;
import org.junit.Test;
import org.junit.runner.RunWith;
import static org.junit.Assert.*;

/** Public Android input injection; run on an emulator, never user collections. */
@RunWith(AndroidJUnit4.class)
public final class PageReorderingInstrumentedTest {
    private final android.app.UiAutomation ui = InstrumentationRegistry.getInstrumentation().getUiAutomation();
    private AccessibilityNodeInfo find(AccessibilityNodeInfo node, String id) {
        if (node == null) return null;
        if (id.equals(node.getViewIdResourceName())) return node;
        for (int i=0;i<node.getChildCount();i++) {
            AccessibilityNodeInfo result=find(node.getChild(i),id);
            if(result!=null)return result;
        }
        return null;
    }
    private AccessibilityNodeInfo waitFor(String id) throws Exception {
        for(int n=0;n<100;n++) {
            AccessibilityNodeInfo node=find(ui.getRootInActiveWindow(),id);
            if(node!=null)return node;
            Thread.sleep(50);
        }
        throw new AssertionError("Missing native view: "+id);
    }
    private void touch(long down,int action,int[] ids,float[] xy) {
        MotionEvent.PointerProperties[] properties=new MotionEvent.PointerProperties[ids.length];
        MotionEvent.PointerCoords[] coords=new MotionEvent.PointerCoords[ids.length];
        for(int i=0;i<ids.length;i++) {
            properties[i]=new MotionEvent.PointerProperties();properties[i].id=ids[i];properties[i].toolType=MotionEvent.TOOL_TYPE_FINGER;
            coords[i]=new MotionEvent.PointerCoords();coords[i].x=xy[2*i];coords[i].y=xy[2*i+1];coords[i].pressure=1;coords[i].size=.1f;
        }
        MotionEvent event=MotionEvent.obtain(down,SystemClock.uptimeMillis(),action,ids.length,properties,coords,0,0,1,1,0,0,InputDevice.SOURCE_TOUCHSCREEN,0);
        try { assertTrue("Touch input must reach the native window",ui.injectInputEvent(event,true)); } finally { event.recycle(); }
    }
    private JSONArray order(File library) throws Exception {
        JSONObject json=new JSONObject(new String(Files.readAllBytes(library.toPath()),java.nio.charset.StandardCharsets.UTF_8));
        JSONArray collections=json.getJSONArray("projects");
        for(int i=0;i<collections.length();i++)if(collections.getJSONObject(i).getLong("id")==26000)return collections.getJSONObject(i).getJSONArray("pages");
        throw new AssertionError("Fixture collection lost");
    }
    @Test public void secondFingerScrollKeepsHeldPageUnderFirstFingerAndPersistsOnRelease() throws Exception {
        Context context=InstrumentationRegistry.getInstrumentation().getTargetContext();
        assertTrue("This test is only authorized on an emulator",android.os.Build.FINGERPRINT.contains("generic") || android.os.Build.MODEL.contains("sdk_gphone"));
        File root=new File(context.getFilesDir(),"SmudeProjects"),renders=new File(root,"renders");
        assertTrue(renders.isDirectory()||renders.mkdirs());
        File library=new File(root,"library.json"),previous=new File(root,"library.previous.json"),image=new File(renders,"drag26-fixture.png");
        byte[] old=library.exists()?Files.readAllBytes(library.toPath()):null;
        byte[] oldPrevious=previous.exists()?Files.readAllBytes(previous.toPath()):null;
        Bitmap bitmap=Bitmap.createBitmap(300,424,Bitmap.Config.ARGB_8888);bitmap.eraseColor(Color.WHITE);
        try(FileOutputStream out=new FileOutputStream(image)){assertTrue(bitmap.compress(Bitmap.CompressFormat.PNG,100,out));}bitmap.recycle();
        JSONArray pages=new JSONArray();
        for(long id=26001;id<=26030;id++) pages.put(new JSONObject().put("id",id).put("capture",id).put("side",0).put("rotation",0).put("model","Mesh").put("review",false).put("status","Ready").put("dimensions",new JSONArray(new int[]{300,424})).put("source","renders/drag26-fixture.png").put("output","renders/drag26-fixture.png"));
        JSONObject fixture=new JSONObject().put("imports",new JSONArray()).put("projects",new JSONArray().put(new JSONObject().put("id",26000).put("title","Reorder26").put("modified_at",System.currentTimeMillis()).put("pages",pages)));
        Files.write(library.toPath(),fixture.toString().getBytes(java.nio.charset.StandardCharsets.UTF_8));
        AccessibilityServiceInfo info=ui.getServiceInfo();info.flags|=AccessibilityServiceInfo.FLAG_REPORT_VIEW_IDS;ui.setServiceInfo(info);
        try(ActivityScenario<MainActivity> scenario=ActivityScenario.launch(MainActivity.class)) {
            assertTrue(waitFor("project-26000").performAction(AccessibilityNodeInfo.ACTION_CLICK));
            Rect page=new Rect(),viewport=new Rect();waitFor("page-26001").getBoundsInScreen(page);waitFor("page-scroll").getBoundsInScreen(viewport);
            Rect third=new Rect();waitFor("page-26003").getBoundsInScreen(third);
            assertEquals("Three columns must fit the actual native viewport",page.top,third.top,2);
            float x=page.exactCenterX(),y=page.exactCenterY(),sx=viewport.exactCenterX(),sy=viewport.top+viewport.height()*.80f;
            long down=SystemClock.uptimeMillis();touch(down,MotionEvent.ACTION_DOWN,new int[]{7},new float[]{x,y});
            Thread.sleep(700);waitFor("selection-done");
            touch(down,MotionEvent.ACTION_POINTER_DOWN|(1<<MotionEvent.ACTION_POINTER_INDEX_SHIFT),new int[]{7,19},new float[]{x,y,sx,sy});
            float distance=viewport.height()*.50f;
            for(int step=1;step<=30;step++) { touch(down,MotionEvent.ACTION_MOVE,new int[]{7,19},new float[]{x,y,sx,sy-distance*step/30});Thread.sleep(12); }
            Thread.sleep(150);
            assertEquals("Preview must not persist before release",26001,order(library).getJSONObject(0).getLong("id"));
            Rect held=new Rect();waitFor("page-26001").getBoundsInScreen(held);
            float tolerance=8*context.getResources().getDisplayMetrics().density;
            assertEquals("The held page must remain under the first finger while scrolling",y,held.exactCenterY(),tolerance);
            touch(down,MotionEvent.ACTION_POINTER_UP,new int[]{7,19},new float[]{x,y,sx,sy-distance});
            touch(down,MotionEvent.ACTION_UP,new int[]{19},new float[]{sx,sy-distance});
            int position=0;
            for(int attempt=0;attempt<50;attempt++) {
                JSONArray reordered=order(library);
                for(int i=0;i<reordered.length();i++)if(reordered.getJSONObject(i).getLong("id")==26001)position=i;
                if(position>=3)break;Thread.sleep(50);
            }
            assertTrue("Drop target must include the second finger's scroll displacement",position>=3);
            assertEquals(30,order(library).length());
        } finally {
            if(old==null)Files.deleteIfExists(library.toPath());else Files.write(library.toPath(),old);
            if(oldPrevious==null)Files.deleteIfExists(previous.toPath());else Files.write(previous.toPath(),oldPrevious);
            Files.deleteIfExists(image.toPath());
        }
    }
}
