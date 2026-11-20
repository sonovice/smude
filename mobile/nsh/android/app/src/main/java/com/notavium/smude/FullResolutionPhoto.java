package com.notavium.smude;

import android.media.ExifInterface;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;

/** Write complete JPEG bytes without applying the analysis/preview crop or decoding pixels. */
final class FullResolutionPhoto {
    static void writeJpeg(ByteBuffer source, int clockwiseDegrees, File file) throws IOException {
        ByteBuffer jpeg = source.duplicate();
        try (FileOutputStream output = new FileOutputStream(file)) {
            byte[] chunk = new byte[Math.min(65536,jpeg.remaining())];
            while (jpeg.hasRemaining()) {
                int n = Math.min(chunk.length,jpeg.remaining());
                jpeg.get(chunk,0,n); output.write(chunk,0,n);
            }
        }
        ExifInterface exif = new ExifInterface(file.getPath());
        int orientation = switch ((clockwiseDegrees % 360 + 360) % 360) {
            case 90 -> ExifInterface.ORIENTATION_ROTATE_90;
            case 180 -> ExifInterface.ORIENTATION_ROTATE_180;
            case 270 -> ExifInterface.ORIENTATION_ROTATE_270;
            default -> ExifInterface.ORIENTATION_NORMAL;
        };
        exif.setAttribute(ExifInterface.TAG_ORIENTATION,Integer.toString(orientation));
        exif.saveAttributes();
    }
    private FullResolutionPhoto() {}
}
