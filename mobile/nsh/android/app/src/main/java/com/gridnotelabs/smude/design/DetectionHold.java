package com.gridnotelabs.smude.design;

/** Time and scene evidence for the capture latch; no Android dependency. */
final class DetectionHold {
    static final long GRACE_MS=600;
    static final double SCENE_CHANGE=.08;
    static boolean retain(boolean wasReady,long now,long lastGood,double sceneDifference,boolean hardFailure) {
        return wasReady && !hardFailure && now>=lastGood && now-lastGood<=GRACE_MS
            && Double.isFinite(sceneDifference) && sceneDifference<=SCENE_CHANGE;
    }
    /** Ignore uniform exposure changes, retain spatial image changes. */
    static double sceneDifference(double[] before,double[] after) {
        if(before==null || after==null || before.length!=64 || after.length!=64) return Double.POSITIVE_INFINITY;
        double shift=0;
        for(int i=0;i<64;i++) {
            if(!Double.isFinite(before[i]) || !Double.isFinite(after[i])) return Double.POSITIVE_INFINITY;
            shift+=after[i]-before[i];
        }
        shift/=64;
        double absolute=0;
        for(int i=0;i<64;i++) absolute+=Math.abs(after[i]-before[i]-shift);
        return absolute/64;
    }
}
