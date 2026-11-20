package com.notavium.smude;

import android.animation.ValueAnimator;
import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.RectF;
import android.view.View;

/** Solid clockwise completion sector; no track, outline, backdrop or text. */
final class PieProgressView extends View {
    private final Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final RectF circle = new RectF();
    private float fraction;
    private float target = -1;
    private ValueAnimator completion;
    PieProgressView(Context context) {
        super(context);
        paint.setColor(0xffffcc66);
        setImportantForAccessibility(IMPORTANT_FOR_ACCESSIBILITY_NO);
    }
    void update(float percent, boolean indeterminate) {
        float next = indeterminate ? 0 : Math.max(0, Math.min(100, percent)) / 100;
        if (next == target && !indeterminate) return;
        target = next;
        if (completion != null) completion.cancel();
        if (isAttachedToWindow() && !indeterminate && ValueAnimator.areAnimatorsEnabled()) {
            completion = ValueAnimator.ofFloat(fraction, next);
            completion.setDuration(200);
            completion.addUpdateListener(a -> { fraction = (float) a.getAnimatedValue(); invalidate(); });
            completion.start();
        } else { fraction = next; invalidate(); }
    }
    @Override protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        float diameter = Math.min(getWidth(), getHeight());
        float left = (getWidth() - diameter) / 2, top = (getHeight() - diameter) / 2;
        circle.set(left, top, left + diameter, top + diameter);
        canvas.drawArc(circle, -90, fraction * 360, true, paint);
    }
    @Override protected void onAttachedToWindow() {
        super.onAttachedToWindow();
        if (target >= 0) { fraction = target; invalidate(); }
    }
    @Override protected void onDetachedFromWindow() {
        if (completion != null) completion.cancel();
        super.onDetachedFromWindow();
    }
}
