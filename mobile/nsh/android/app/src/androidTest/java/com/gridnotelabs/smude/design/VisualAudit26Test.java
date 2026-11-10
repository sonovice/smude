package com.gridnotelabs.smude.design;
import android.content.Context;
import android.graphics.*;
import android.os.*;
import android.view.*;
import android.view.accessibility.AccessibilityNodeInfo;
import android.accessibilityservice.AccessibilityServiceInfo;
import androidx.test.core.app.ActivityScenario;
import androidx.test.platform.app.InstrumentationRegistry;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import java.io.*;
import java.nio.file.*;
import org.json.*;
import org.junit.Test;
import org.junit.runner.RunWith;
import static org.junit.Assert.*;
@RunWith(AndroidJUnit4.class) public class VisualAudit26Test {
 final android.app.UiAutomation ui=InstrumentationRegistry.getInstrumentation().getUiAutomation();
 File out; Context ctx;
 AccessibilityNodeInfo find(AccessibilityNodeInfo n,String key){if(n==null)return null;if(key.equals(n.getViewIdResourceName())||key.contentEquals(n.getText()==null?"":n.getText())||key.contentEquals(n.getContentDescription()==null?"":n.getContentDescription()))return n;for(int i=0;i<n.getChildCount();i++){AccessibilityNodeInfo r=find(n.getChild(i),key);if(r!=null)return r;}return null;}
 AccessibilityNodeInfo node(String k)throws Exception{for(int i=0;i<40;i++){AccessibilityNodeInfo n=find(ui.getRootInActiveWindow(),k);if(n!=null)return n;Thread.sleep(100);}throw new Exception("Missing: "+k);}
 void click(String k)throws Exception{AccessibilityNodeInfo n=node(k);Rect r=new Rect();n.getBoundsInScreen(r);tap(r.centerX(),r.centerY(),80);Thread.sleep(500);}
 void tap(float x,float y,long duration)throws Exception{long t=SystemClock.uptimeMillis();MotionEvent a=MotionEvent.obtain(t,t,0,x,y,0);a.setSource(InputDevice.SOURCE_TOUCHSCREEN);ui.injectInputEvent(a,true);a.recycle();Thread.sleep(duration);a=MotionEvent.obtain(t,SystemClock.uptimeMillis(),1,x,y,0);a.setSource(InputDevice.SOURCE_TOUCHSCREEN);ui.injectInputEvent(a,true);a.recycle();}
 void hold(String k)throws Exception{Rect r=new Rect();node(k).getBoundsInScreen(r);tap(r.centerX(),r.centerY(),850);Thread.sleep(500);}
 void input(String k,String text)throws Exception{Bundle b=new Bundle();b.putCharSequence(AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE,text);node(k).performAction(AccessibilityNodeInfo.ACTION_SET_TEXT,b);Thread.sleep(500);}
 void back()throws Exception{ui.injectInputEvent(new KeyEvent(KeyEvent.ACTION_DOWN,KeyEvent.KEYCODE_BACK),true);ui.injectInputEvent(new KeyEvent(KeyEvent.ACTION_UP,KeyEvent.KEYCODE_BACK),true);Thread.sleep(500);}
 void snap(String name)throws Exception{Thread.sleep(600);Bitmap b=ui.takeScreenshot();try(FileOutputStream f=new FileOutputStream(new File(out,name+".png"))){b.compress(Bitmap.CompressFormat.PNG,100,f);}b.recycle();}
 static void writeText(java.nio.file.Path p,String s)throws Exception{Files.write(p,s.getBytes(java.nio.charset.StandardCharsets.UTF_8));}
 interface Task{void run()throws Exception;}
 void run(String label,Task task)throws Exception{try(ActivityScenario<MainActivity> s=ActivityScenario.launch(MainActivity.class)){node("home-camera");task.run();}catch(Throwable t){writeText(new File(out,label+"-error.txt").toPath(),t.toString());snap(label+"-failure");}Thread.sleep(500);}
 @Test public void capture()throws Exception{
 ctx=InstrumentationRegistry.getInstrumentation().getTargetContext();assertTrue(android.os.Build.MODEL.contains("sdk_gphone")||android.os.Build.FINGERPRINT.contains("generic"));out=new File(ctx.getExternalFilesDir(null),"audit26");out.mkdirs();
 File root=new File(ctx.getFilesDir(),"SmudeProjects"),lib=new File(root,"library.json"),prev=new File(root,"library.previous.json");root.mkdirs();byte[] old=lib.exists()?Files.readAllBytes(lib.toPath()):null,oldPrev=prev.exists()?Files.readAllBytes(prev.toPath()):null;
 if(old!=null)Files.write(new File(out,"backup-library.json").toPath(),old);if(oldPrev!=null)Files.write(new File(out,"backup-library.previous.json").toPath(),oldPrev);
 File image=new File(root,"renders/audit26-score.png");image.getParentFile().mkdirs();try(InputStream in=InstrumentationRegistry.getInstrumentation().getContext().getAssets().open("audit26-score.png")){Files.copy(in,image.toPath(),StandardCopyOption.REPLACE_EXISTING);}
 JSONArray pages=new JSONArray();for(int i=0;i<12;i++)pages.put(new JSONObject().put("id",27001+i).put("capture",27001+i).put("side",0).put("rotation",0).put("model","Mesh").put("review",false).put("status","Ready").put("dimensions",new JSONArray(new int[]{3029,4134})).put("source","renders/audit26-score.png").put("output","renders/audit26-score.png"));
 JSONArray ps=new JSONArray().put(new JSONObject().put("id",27000).put("title","Evening practice").put("pages",pages).put("modified_at",1794300000000L)).put(new JSONObject().put("id",27300).put("title","New collection").put("pages",new JSONArray())).put(new JSONObject().put("id",27400).put("title","Scans").put("draft",true).put("pages",new JSONArray().put(new JSONObject(pages.getJSONObject(0).toString()).put("id",27401))));
 AccessibilityServiceInfo info=ui.getServiceInfo();info.flags|=AccessibilityServiceInfo.FLAG_REPORT_VIEW_IDS;ui.setServiceInfo(info);
 try{
 writeText(lib.toPath(),new JSONObject().put("imports",new JSONArray()).put("projects",ps).toString());
 run("home",()->{snap("41-android-home");click("project-view-grid");snap("42-android-grid");click("search");input("search","zznomatch");snap("43-android-search-empty");});
 run("name",()->{click("new-project");snap("44-android-name-empty");input("project-name","Morning studies");snap("45-android-name-keyboard");click("Save");snap("46-android-collection-empty");});
 run("collection",()->{click("project-27000");snap("47-android-collection");click("collection-more");snap("48-android-collection-menu");back();hold("project-title");snap("49-android-title-menu");click("Rename");snap("50-android-rename");});
 run("viewer",()->{click("project-27000");click("page-27001");snap("51-android-viewer");click("Original");snap("52-android-original");click("viewer-more");snap("53-android-viewer-menu");back();click("viewer-retake");snap("54-android-retake");});
 run("share",()->{click("project-27000");click("collection-more");click("Share");snap("55-android-share-initial");Thread.sleep(5000);snap("56-android-share-ready");click("share-filename");snap("57-android-share-keyboard");back();click("share-pdf");Thread.sleep(1500);snap("58-android-system-share");});
 run("selection",()->{click("project-27000");hold("page-27001");click("page-27002");snap("59-android-selection");click("collection-more");snap("60-android-selection-menu");click("Share");Thread.sleep(2000);snap("61-android-selection-share");});
 run("camera",()->{click("home-camera");Thread.sleep(2500);snap("62-android-camera");});
 run("draft",()->{click("project-27400");snap("63-android-draft");click("collection-more");snap("64-android-draft-menu");click("Add to collection");snap("65-android-destination");});
 run("files",()->{click("project-27300");click("collection-more");click("Files");Thread.sleep(1000);snap("66-android-files");});
 run("photos",()->{click("project-27300");click("collection-more");click("Import photos");Thread.sleep(1000);snap("67-android-photos");});
 }finally{if(old==null)Files.deleteIfExists(lib.toPath());else Files.write(lib.toPath(),old);if(oldPrev==null)Files.deleteIfExists(prev.toPath());else Files.write(prev.toPath(),oldPrev);Files.deleteIfExists(image.toPath());writeText(new File(out,"restored.txt").toPath(),"Library and previous library restored byte-for-byte");}
 }
}
