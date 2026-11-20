package com.notavium.smude;

/** Physical pose independently of the display/rotation lock. Surface rotations are 0..3. */
final class CameraOrientation {
    private int deviceRotation = -1;

    boolean observe(int clockwiseDegrees) {
        if (clockwiseDegrees < 0) return false;
        int degrees = clockwiseDegrees % 360;
        if (deviceRotation >= 0) {
            int previous = (360 - deviceRotation * 90) % 360;
            int delta = Math.abs(degrees - previous);
            // Hysteresis avoids repeatedly switching at a diagonal boundary.
            if (Math.min(delta, 360 - delta) <= 60) return false;
        }
        int next = ((360 - ((degrees + 45) / 90 * 90)) % 360) / 90;
        if (next == deviceRotation) return false;
        deviceRotation = next;
        return true;
    }

    int captureRotation(int displayRotation) {
        return deviceRotation < 0 ? displayRotation : deviceRotation;
    }

    int analysisTurns(int displayRotation) {
        return (displayRotation - captureRotation(displayRotation) + 4) % 4;
    }
}
