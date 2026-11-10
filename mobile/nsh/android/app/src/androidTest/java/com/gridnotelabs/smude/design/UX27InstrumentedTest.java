package com.gridnotelabs.smude.design;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Rect;
import android.os.Bundle;
import android.os.SystemClock;
import android.view.InputDevice;
import android.view.KeyEvent;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.view.accessibility.AccessibilityNodeInfo;
import android.accessibilityservice.AccessibilityServiceInfo;
import androidx.test.core.app.ActivityScenario;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;
import com.longbridge.nsh.NshComponentContext;
import java.io.File;
import java.io.FileOutputStream;
import java.nio.file.Files;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;
import org.json.JSONArray;
import org.json.JSONObject;
import org.junit.Test;
import org.junit.runner.RunWith;
import static org.junit.Assert.*;

/** Emulator-only native behavior and visual regression; original library restored. */
@RunWith(AndroidJUnit4.class)
public final class UX27InstrumentedTest {
 final android.app.Instrumentation instrumentation=InstrumentationRegistry.getInstrumentation();
 final android.app.UiAutomation ui=instrumentation.getUiAutomation();
 File out;
 interface Flow { void run(ActivityScenario<MainActivity> scenario) throws Exception; }
 AccessibilityNodeInfo find(AccessibilityNodeInfo n,String key){
  if(n==null)return null;
  if(key.equals(n.getViewIdResourceName())||key.contentEquals(n.getText()==null?"":n.getText())||key.contentEquals(n.getContentDescription()==null?"":n.getContentDescription()))return n;
  for(int i=0;i<n.getChildCount();i++){AccessibilityNodeInfo r=find(n.getChild(i),key);if(r!=null)return r;}return null;
 }
 AccessibilityNodeInfo node(String key)throws Exception{for(int i=0;i<80;i++){AccessibilityNodeInfo n=find(ui.getRootInActiveWindow(),key);if(n!=null)return n;Thread.sleep(50);}snap("failure-"+key.replace(' ','_'));dump(ui.getRootInActiveWindow(),new StringBuilder(),key);throw new AssertionError("Missing view: "+key);}
 void dump(AccessibilityNodeInfo n,StringBuilder b,String key)throws Exception{if(n==null)return;b.append(n.toString()).append('\n');for(int i=0;i<n.getChildCount();i++)dump(n.getChild(i),b,key);Files.write(new File(out,"tree-"+key.replace(' ','_')+".txt").toPath(),b.toString().getBytes(StandardCharsets.UTF_8));}
 void tap(String key)throws Exception { Rect r=new Rect();node(key).getBoundsInScreen(r);touch(r.centerX(),r.centerY(),80);Thread.sleep(250); }
 void hold(String key)throws Exception {Rect r=new Rect();node(key).getBoundsInScreen(r);touch(r.centerX(),r.centerY(),700);Thread.sleep(200);}
 void touch(float x,float y,long duration)throws Exception{long t=SystemClock.uptimeMillis();MotionEvent a=MotionEvent.obtain(t,t,0,x,y,0);a.setSource(InputDevice.SOURCE_TOUCHSCREEN);assertTrue(ui.injectInputEvent(a,true));a.recycle();Thread.sleep(duration);a=MotionEvent.obtain(t,SystemClock.uptimeMillis(),1,x,y,0);a.setSource(InputDevice.SOURCE_TOUCHSCREEN);assertTrue(ui.injectInputEvent(a,true));a.recycle();}
 void back()throws Exception{assertTrue(ui.performGlobalAction(android.accessibilityservice.AccessibilityService.GLOBAL_ACTION_BACK));Thread.sleep(350);}
 void snap(String name)throws Exception {Bitmap b=ui.takeScreenshot();assertNotNull(b);try(FileOutputStream f=new FileOutputStream(new File(out,name+".png"))){assertTrue(b.compress(Bitmap.CompressFormat.PNG,100,f));}b.recycle();}
 void fixture(Flow flow)throws Exception {
  Context context=instrumentation.getTargetContext();assertTrue("Emulator only",android.os.Build.MODEL.contains("sdk_gphone")||android.os.Build.FINGERPRINT.contains("generic"));
  out=new File(context.getExternalFilesDir(null),"ux27");assertTrue(out.isDirectory()||out.mkdirs());
  File root=new File(context.getFilesDir(),"SmudeProjects"),renders=new File(root,"renders");assertTrue(renders.isDirectory()||renders.mkdirs());
  File library=new File(root,"library.json"),previous=new File(root,"library.previous.json"),image=new File(renders,"ux27-fixture.png");
  byte[] old=library.exists()?Files.readAllBytes(library.toPath()):null,oldPrevious=previous.exists()?Files.readAllBytes(previous.toPath()):null;
  File saved=new File(out,"backup-library.json"),savedPrevious=new File(out,"backup-library.previous.json");
  if(!saved.exists()&&old!=null)Files.write(saved.toPath(),old);
  if(!savedPrevious.exists()&&oldPrevious!=null)Files.write(savedPrevious.toPath(),oldPrevious);
  try(var in=instrumentation.getContext().getAssets().open("audit26-score.png")){Files.copy(in,image.toPath(),java.nio.file.StandardCopyOption.REPLACE_EXISTING);}
  JSONArray pages=new JSONArray();for(int i=1;i<=6;i++)pages.put(new JSONObject().put("id",27000+i).put("capture",27000+i).put("side",0).put("rotation",0).put("model","Mesh").put("review",false).put("status","Ready").put("source","renders/ux27-fixture.png").put("output","renders/ux27-fixture.png"));
  JSONObject failed=new JSONObject(pages.getJSONObject(0).toString()).put("id",27201).put("status","Failed").put("output",JSONObject.NULL);
  JSONArray collections=new JSONArray().put(new JSONObject().put("id",27000).put("title","Evening practice").put("pages",pages).put("modified_at",1794300000000L))
   .put(new JSONObject().put("id",27200).put("title","Failed page").put("pages",new JSONArray().put(failed)))
   .put(new JSONObject().put("id",27300).put("title","New collection").put("pages",new JSONArray()))
   .put(new JSONObject().put("id",27400).put("title","Scans").put("draft",true).put("pages",new JSONArray().put(new JSONObject(pages.getJSONObject(0).toString()).put("id",27401))));
  Files.write(library.toPath(),new JSONObject().put("imports",new JSONArray()).put("projects",collections).toString().getBytes(StandardCharsets.UTF_8));
  AccessibilityServiceInfo info=ui.getServiceInfo();info.flags|=AccessibilityServiceInfo.FLAG_REPORT_VIEW_IDS;ui.setServiceInfo(info);
  ui.grantRuntimePermission(context.getPackageName(),android.Manifest.permission.CAMERA);
  try(ActivityScenario<MainActivity> scenario=ActivityScenario.launch(MainActivity.class)){node("home-camera");flow.run(scenario);}
  finally{if(old==null)Files.deleteIfExists(library.toPath());else Files.write(library.toPath(),old);if(oldPrevious==null)Files.deleteIfExists(previous.toPath());else Files.write(previous.toPath(),oldPrevious);Files.deleteIfExists(image.toPath());}
 }
 @Test public void selectionScopeRenameAndPdfPreparationAreVisible()throws Exception {fixture(s->{
  tap("project-27000");tap("collection-more");node("Rename");snap("android-collection-menu");back();
  hold("page-27001");tap("page-27002");node("Delete 2 pages");snap("android-selection");
  tap("collection-more");node("Share 2 pages");snap("android-selection-menu");tap("Share 2 pages");
  node("share-filename");node("Share 2 pages");snap("android-share-preparing");
  for(int i=0;i<120&&!node("share-pdf").isEnabled();i++)Thread.sleep(100);
  assertTrue(node("share-pdf").isEnabled());snap("android-share-ready");
 });}
 @Test public void missingResultAndEmptyCollectionHaveHonestStates()throws Exception {fixture(s->{
  tap("project-27200");tap("page-27201");assertFalse(node("Processed").isEnabled());assertTrue(node("Original").isEnabled());snap("android-failed-original");
  tap("viewer-more");snap("android-destructive-menu");back();back();back();
  tap("project-27300");node("empty-collection-camera");node("empty-collection-import");snap("android-empty-collection");
 });}
 @Test public void clearSearchWorksWithKeyboardAndDraftAssignmentIsDiscoverable()throws Exception {fixture(s->{
  tap("project-view-list");Rect draft=new Rect();node("project-27400").getBoundsInScreen(draft);assertTrue(draft.height()<100*instrumentation.getTargetContext().getResources().getDisplayMetrics().density);snap("android-home-list");
  tap("search");instrumentation.sendStringSync("zznomatch");node("search-empty");snap("android-search-empty");tap("clear-search");node("project-27400");back();
  tap("project-27400");node("draft-assign");snap("android-draft");tap("draft-assign");node("＋ Create new collection…");snap("android-destination");
 });}
 // App-owned native classes are inspected only in this test; no private OS APIs.
 Object component(View v)throws Exception {
  if(v.getClass().getName().equals("com.longbridge.nsh.ComponentBox")){var field=v.getClass().getDeclaredField("component");field.setAccessible(true);Object c=field.get(v);if(c instanceof CameraComponent)return c;}
  if(v instanceof ViewGroup group)for(int i=0;i<group.getChildCount();i++){Object c=component(group.getChildAt(i));if(c!=null)return c;}return null;
 }
 View named(View v,String id)throws Exception {
  var delegate=v.getAccessibilityDelegate();if(delegate!=null&&delegate.getClass().getName().equals("com.longbridge.nsh.A11y")){var f=delegate.getClass().getDeclaredField("testId");f.setAccessible(true);if(id.equals(f.get(delegate)))return v;}
  if(v instanceof ViewGroup group)for(int i=0;i<group.getChildCount();i++){View r=named(group.getChildAt(i),id);if(r!=null)return r;}return null;
 }
 @Test public void acceptedCaptureFlashesThenClearsWithoutBlockingDone()throws Exception {fixture(s->{
  tap("project-27000");tap("collection-camera");node("shutter");snap("android-camera");
  AtomicReference<View> flash=new AtomicReference<>();
  s.onActivity(activity->{try{
    View root=activity.findViewById(android.R.id.content);Object camera=component(root);assertNotNull(camera);
    var f=CameraComponent.class.getDeclaredField("context");f.setAccessible(true);NshComponentContext context=(NshComponentContext)f.get(camera);
    flash.set(named(root,"capture-flash"));assertNotNull(flash.get());assertEquals(0f,flash.get().getAlpha(),.001f);
    context.emit("capture_started",Map.of("pages",1));
  }catch(Exception e){throw new AssertionError(e);}});
  boolean saw=false;for(int i=0;i<35;i++){AtomicReference<Float> alpha=new AtomicReference<>();s.onActivity(a->alpha.set(flash.get().getAlpha()));if(alpha.get()>.1f){saw=true;break;}Thread.sleep(4);}
  assertTrue("Native animation must visibly darken the preview after accepted capture",saw);snap("android-shutter-flash");
  Thread.sleep(260);s.onActivity(a->assertEquals(0f,flash.get().getAlpha(),.01f));
  node("capture-stack");assertTrue(node("camera-done").isEnabled());snap("android-camera-after-flash");
 });}
 @Test public void cutoutFliesIntoStackAndReviewRemainsImmediate()throws Exception {fixture(s->{
  tap("project-27000");tap("collection-camera");node("capture-stack");node("shutter");
  android.graphics.Bitmap bitmap=android.graphics.BitmapFactory.decodeStream(instrumentation.getContext().getAssets().open("camera-stack-source.jpg"));
  java.nio.ByteBuffer bytes=java.nio.ByteBuffer.allocateDirect(bitmap.getWidth()*bitmap.getHeight()*4);
  for(int y=0;y<bitmap.getHeight();y++)for(int x=0;x<bitmap.getWidth();x++){int pixel=bitmap.getPixel(x,y);bytes.put((byte)(pixel>>16));bytes.put((byte)(pixel>>8));bytes.put((byte)pixel);bytes.put((byte)255);}bytes.rewind();
  JSONObject detection=new JSONObject(CameraInference.detect(bytes,bitmap.getWidth(),bitmap.getHeight(),bitmap.getWidth()*4,0));
  assertTrue("Camera fixture must provide a valid page: "+detection,detection.getBoolean("valid"));
  long token=detection.getLong("preview_frame");assertTrue(token>0);
  AtomicReference<NshComponentContext> ctx=new AtomicReference<>();
  s.onActivity(activity->{try{Object camera=component(activity.findViewById(android.R.id.content));var f=CameraComponent.class.getDeclaredField("context");f.setAccessible(true);ctx.set((NshComponentContext)f.get(camera));ctx.get().emit("capture_started",Map.of("pages",1,"preview_frame",token));}catch(Exception e){throw new AssertionError(e);}});
  AtomicReference<View> flying=new AtomicReference<>();
  for(int i=0;i<50&&flying.get()==null;i++){s.onActivity(a->{try{flying.set(named(a.findViewById(android.R.id.content),"capture-flying-page"));}catch(Exception e){throw new AssertionError(e);}});if(flying.get()==null)Thread.sleep(5);}
  assertNotNull("Native non-interactive cutout must be mounted during flight",flying.get());
  boolean visible=false;for(int i=0;i<35;i++){AtomicReference<Float> alpha=new AtomicReference<>();s.onActivity(a->alpha.set(flying.get().getAlpha()));if(alpha.get()>.4f){visible=true;break;}Thread.sleep(4);}assertTrue("Page must become visible during the native flight",visible);
  Thread.sleep(65);snap("android-page-flight");Thread.sleep(520);snap("android-page-stack");
  // A second flight must not trap navigation or wait for a render worker.
  s.onActivity(a->ctx.get().emit("capture_started",Map.of("pages",1,"preview_frame",token)));
  tap("capture-stack");node("collection-camera");node("page-27001");snap("android-stack-review");
 });}

 @Test public void nativeControlsAndAnchoredMenusMatchActions()throws Exception{fixture(s->{
  tap("project-27000");tap("collection-more");Rect rename=new Rect();node("Rename").getBoundsInScreen(rename);assertTrue("Collection menu must be near the header",rename.centerY()<ui.takeScreenshot().getHeight()*.55);snap("android-29-collection-menu");back();
  hold("page-27001");tap("page-27002");tap("collection-more");node("Share 2 pages");snap("android-29-selection-menu");tap("Share 2 pages");node("share-filename");node("Share 2 pages");snap("android-29-selected-share");back();
  tap("page-27001");tap("page-27002");back();tap("project-27000");tap("page-27001");
  assertTrue(node("Processed").isChecked());tap("Original");assertTrue(node("Original").isChecked());tap("Processed");assertTrue(node("Processed").isChecked());
  s.onActivity(a->{assertTrue("Real Material segmented control must be mounted",hasClass(a.findViewById(android.R.id.content),com.google.android.material.button.MaterialButtonToggleGroup.class));});
  tap("viewer-rotate");snap("android-29-viewer-controls");tap("viewer-more");Rect delete=new Rect();node("Delete page").getBoundsInScreen(delete);assertTrue("Viewer menu must be next to its button",delete.centerY()<ui.takeScreenshot().getHeight()*.5);snap("android-29-viewer-menu");back();
  tap("viewer-retake");node("Replace page 1");
 });}
 boolean hasClass(View root,Class<?> type){if(type.isInstance(root))return true;if(root instanceof ViewGroup group)for(int i=0;i<group.getChildCount();i++)if(hasClass(group.getChildAt(i),type))return true;return false;}

}
