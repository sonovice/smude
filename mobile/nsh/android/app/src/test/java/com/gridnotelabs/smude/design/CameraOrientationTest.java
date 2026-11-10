package com.gridnotelabs.smude.design;

import org.junit.Test;
import static org.junit.Assert.*;

public final class CameraOrientationTest {
    @Test public void lockedDisplayDoesNotLockCaptureOrInference() {
        CameraOrientation pose = new CameraOrientation();
        assertEquals(1, pose.captureRotation(1));
        assertEquals(0, pose.analysisTurns(1));
        for (int degrees : new int[]{0,90,180,270}) {
            pose.observe(degrees);
            int target = ((360 - degrees) % 360) / 90;
            assertEquals(target, pose.captureRotation(0));
            assertEquals((4-target)%4, pose.analysisTurns(0));
            assertEquals(0, pose.analysisTurns(target));
        }
    }
    @Test public void diagonalJitterAndUnknownPoseKeepLastOrientation() {
        CameraOrientation pose = new CameraOrientation();
        assertTrue(pose.observe(0));
        for (int angle : new int[]{44,46,58,40,-1,359}) assertFalse(pose.observe(angle));
        assertTrue(pose.observe(70));
        for (int angle : new int[]{44,46,60,-1}) assertFalse(pose.observe(angle));
        assertEquals(3, pose.captureRotation(0));
        assertEquals(1, pose.analysisTurns(0));
    }
}
