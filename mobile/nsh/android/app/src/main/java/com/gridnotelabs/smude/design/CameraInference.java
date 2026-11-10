package com.gridnotelabs.smude.design;

import java.nio.ByteBuffer;

/** RGBA remains in native/direct memory. Only small contour JSON crosses JNI. */
final class CameraInference {
    static { System.loadLibrary("nsh"); }
    static native String detect(ByteBuffer rgba, int width, int height, int stride, int clockwiseTurns);
    private CameraInference() {}
}
