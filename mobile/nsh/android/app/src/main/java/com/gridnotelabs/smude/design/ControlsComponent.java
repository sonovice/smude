package com.gridnotelabs.smude.design;

import android.content.res.ColorStateList;
import android.view.ContextThemeWrapper;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import android.widget.ImageButton;
import android.widget.LinearLayout;
import android.widget.PopupMenu;
import com.google.android.material.button.MaterialButton;
import com.google.android.material.color.MaterialColors;
import com.google.android.material.shape.MaterialShapeDrawable;
import com.google.android.material.shape.ShapeAppearanceModel;
import com.google.android.material.shape.RelativeCornerSize;
import com.google.android.material.button.MaterialButtonToggleGroup;
import com.longbridge.nsh.NshComponent;
import com.longbridge.nsh.NshComponentContext;
import java.util.List;
import java.util.Map;

/** Native controls and anchored menus. Domain actions stay in Rust. */
final class ControlsComponent implements NshComponent {
    private final NshComponentContext context;
    private final ContextThemeWrapper theme;
    private final FrameLayout root;
    private ImageButton button;
    private MaterialButtonToggleGroup segments;
    private PieProgressView progress;
    private PopupMenu popup;
    private List<?> choices=List.of();
    private String kind="";
    private boolean updating;
    private boolean destroyed;
    ControlsComponent(MainActivity activity,NshComponentContext context){
        this.context=context;
        theme=new ContextThemeWrapper(activity,R.style.SmudeControlsTheme);
        root=new FrameLayout(theme);
    }
    @Override public View view(){return root;}
    private int dp(int value){return Math.round(value*root.getResources().getDisplayMetrics().density);}
    @Override public void setProp(String name,Object value){
        if(!name.equals("config")||!(value instanceof Map<?,?> config))return;
        String next=config.get("kind") instanceof String s?s:"button";
        if(!next.equals(kind)){
            if(popup!=null)popup.dismiss();root.removeAllViews();button=null;segments=null;progress=null;root.setBackground(null);root.setElevation(0);kind=next;
            if(next.equals("progress")){
                MaterialShapeDrawable surface=new MaterialShapeDrawable(ShapeAppearanceModel.builder()
                    .setAllCornerSizes(new RelativeCornerSize(0.5f)).build());
                surface.initializeElevationOverlay(theme);
                surface.setFillColor(ColorStateList.valueOf(MaterialColors.getColor(root,com.google.android.material.R.attr.colorSurface,0xff242424)));
                surface.setElevation(dp(3));root.setBackground(surface);root.setElevation(dp(3));
                progress=new PieProgressView(theme);
                FrameLayout.LayoutParams params=new FrameLayout.LayoutParams(-1,-1);
                root.addView(progress,params);
            }else if(next.equals("segments")){
                segments=new MaterialButtonToggleGroup(theme);
                segments.setSingleSelection(true);segments.setSelectionRequired(true);
                for(int i=0;i<2;i++){
                    MaterialButton segment=new MaterialButton(theme,null,com.google.android.material.R.attr.materialButtonOutlinedStyle);
                    segment.setId(View.generateViewId());segment.setTextSize(13);segment.setAllCaps(false);
                    segment.setMinWidth(0);segment.setMinimumWidth(0);segment.setMinHeight(dp(48));segment.setMinimumHeight(dp(48));
                    segment.setInsetTop(0);segment.setInsetBottom(0);segment.setPadding(dp(10),0,dp(10),0);
                    segments.addView(segment,new LinearLayout.LayoutParams(0,ViewGroup.LayoutParams.MATCH_PARENT,1));
                }
                segments.addOnButtonCheckedListener((group,id,checked)->{
                    if(!updating&&checked&&!destroyed){for(int i=0;i<2;i++)if(group.getChildAt(i).getId()==id)context.emit("change",Map.of("value",i));}
                });
                root.addView(segments,new FrameLayout.LayoutParams(-1,-1));
            }else{
                button=new ImageButton(theme);
                android.util.TypedValue background=new android.util.TypedValue();
                theme.getTheme().resolveAttribute(android.R.attr.selectableItemBackgroundBorderless,background,true);
                button.setBackgroundResource(background.resourceId);
                button.setScaleType(android.widget.ImageView.ScaleType.CENTER_INSIDE);button.setPadding(dp(12),dp(12),dp(12),dp(12));
                button.setImageTintList(new ColorStateList(new int[][]{new int[]{android.R.attr.state_enabled},new int[]{}},new int[]{0xffffcc66,0x66ffcc66}));
                button.setOnClickListener(v->{if(choices.isEmpty())context.emit("select",Map.of("id","press"));else showMenu();});
                root.addView(button,new FrameLayout.LayoutParams(-1,-1));
            }
        }
        updating=true;
        try{
            if(segments!=null){
                List<?> labels=config.get("labels") instanceof List<?> list?list:List.of();
                List<?> enabled=config.get("enabled") instanceof List<?> list?list:List.of();
                boolean icons=config.get("symbols") instanceof List<?> symbols&&!symbols.isEmpty();
                for(int i=0;i<2;i++){
                    MaterialButton segment=(MaterialButton)segments.getChildAt(i);
                    String label=i<labels.size()?String.valueOf(labels.get(i)):"";
                    segment.setText(icons?"":label);segment.setContentDescription(label);
                    if(icons){segment.setIconResource(i==0?R.drawable.control_list:R.drawable.control_grid);segment.setIconPadding(0);segment.setIconGravity(MaterialButton.ICON_GRAVITY_TEXT_START);}
                    segment.setEnabled(i<enabled.size()&&Boolean.TRUE.equals(enabled.get(i)));
                }
                int selected=config.get("selected") instanceof Number n?n.intValue():0;
                if(selected>=0&&selected<2)segments.check(segments.getChildAt(selected).getId());
            }
            if(progress!=null){
                float completion=config.get("value") instanceof Number n?n.floatValue():0;
                progress.update(completion,Boolean.TRUE.equals(config.get("indeterminate")));
            }
            if(button!=null){
                button.setEnabled(!Boolean.FALSE.equals(config.get("enabled")));
                button.setContentDescription(config.get("label") instanceof String s?s:"");
                String symbol=config.get("symbol") instanceof String s?s:"ellipsis";
                button.setImageResource(symbol.equals("camera")?R.drawable.control_camera:symbol.equals("rotate.right")?R.drawable.control_rotate:R.drawable.control_more);
                choices=config.get("choices") instanceof List<?> list?List.copyOf(list):List.of();
            }
        }finally{updating=false;}
    }
    private void showMenu(){
        if(destroyed||button==null||!button.isAttachedToWindow())return;
        // Freeze the actions and payloads at presentation, not selection time.
        List<?> entries=List.copyOf(choices);
        popup=new PopupMenu(theme,button,Gravity.END);
        int index=0;
        for(Object value:entries)if(value instanceof Map<?,?> entry){
            android.view.MenuItem item=popup.getMenu().add(0,index++,0,String.valueOf(entry.get("title")));
            if(Boolean.TRUE.equals(entry.get("destructive"))){android.text.SpannableString text=new android.text.SpannableString(item.getTitle());text.setSpan(new android.text.style.ForegroundColorSpan(0xffff6b6b),0,text.length(),0);item.setTitle(text);}
        }
        popup.setOnMenuItemClickListener(item->{
            if(!destroyed&&entries.get(item.getItemId()) instanceof Map<?,?> entry){
                java.util.HashMap<String,Object> event=new java.util.HashMap<>();event.put("id",entry.get("id"));event.put("payload",entry.get("payload"));context.emit("select",event);
            }return true;
        });
        popup.setOnDismissListener(p->popup=null);popup.show();
    }
    @Override public void command(String name,Object args){}
    @Override public void destroy(){destroyed=true;if(popup!=null)popup.dismiss();button=null;segments=null;}
}
