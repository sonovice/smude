package com.gridnotelabs.smude.design;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.drawable.AdaptiveIconDrawable;
import android.graphics.drawable.ColorDrawable;
import android.graphics.drawable.Drawable;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;
import java.io.File;
import java.io.FileOutputStream;
import org.junit.Test;
import org.junit.runner.RunWith;
import static org.junit.Assert.*;

@RunWith(AndroidJUnit4.class)
public final class LauncherIconInstrumentedTest {
    @Test public void launcherUsesAnAdaptiveDarkIconAndKeepsTheGoldInsideACircle() throws Exception {
        Context context = InstrumentationRegistry.getInstrumentation().getTargetContext();
        Drawable drawable = context.getPackageManager().getApplicationIcon(context.getApplicationInfo());
        assertTrue("Legacy square icons receive a launcher-generated background", drawable instanceof AdaptiveIconDrawable);
        AdaptiveIconDrawable icon = (AdaptiveIconDrawable) drawable;
        assertTrue(icon.getBackground() instanceof ColorDrawable);
        int background = ((ColorDrawable) icon.getBackground()).getColor();
        assertTrue("The native launcher background must be dark", Color.red(background) < 40 && Color.green(background) < 40 && Color.blue(background) < 40);
        int size = 512;
        icon.setBounds(0, 0, size, size);
        Bitmap foreground = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888);
        icon.getForeground().draw(new Canvas(foreground));
        int gold = 0;
        int clipped = 0;
        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
                int pixel = foreground.getPixel(x, y);
                if (Color.alpha(pixel) > 240 && Color.red(pixel) > 200 && Color.green(pixel) > 140 && Color.blue(pixel) < 145) {
                    gold++;
                    double dx = x + 0.5 - size / 2.0;
                    double dy = y + 0.5 - size / 2.0;
                    if (dx * dx + dy * dy > size * size / 4.0) clipped++;
                }
            }
        }
        assertTrue("The gold scanner corners and scan beam must remain visible", gold > 4000);
        assertEquals("A circular launcher mask must not cut off gold artwork", 0, clipped);
        Bitmap rendered = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888);
        icon.draw(new Canvas(rendered));
        int rim = rendered.getPixel(size / 2, 8);
        assertTrue("No white launcher rim", Color.red(rim) < 50 && Color.green(rim) < 50 && Color.blue(rim) < 50);
        try (FileOutputStream out = new FileOutputStream(new File(context.getCacheDir(), "launcher-icon.png"))) {
            assertTrue(rendered.compress(Bitmap.CompressFormat.PNG, 100, out));
        }
        rendered.recycle();
        foreground.recycle();
    }
}
