package com.edgeml.edgemlinsight;

import android.content.res.AssetManager;
import android.graphics.Bitmap;

public class NativeClassifier {
    static {
        try {
            System.loadLibrary("opencv_java3");
        } catch (UnsatisfiedLinkError e) {
            // Some example apps (e.g. template matching) require OpenCV 4.
            System.loadLibrary("opencv_java4");
        }
        System.loadLibrary("imageclassifier");
    }


    public NativeClassifier() {

    }

    public long init(AssetManager assetManager,
                     String nativeLogFile,
                     String model_name,
                     String m_accelerator,
                     int num_threads,
                     String m_logging,
                     String m_resizing_func,
                     String m_scale_range,
                     String m_channel,
                     int m_rotation,
                     boolean playback) {
        return initClassifier(
                assetManager,
                nativeLogFile,
                model_name,
                m_accelerator,
                num_threads,
                m_logging,
                m_resizing_func,
                m_scale_range,
                m_channel,
                m_rotation,
                playback);
    }

    public float[] run_classify(long detectorAddr, byte[] srcAddr, int width, int height, int rotation) {
        return classify(detectorAddr, srcAddr, width, height, rotation);
    }

    public float[] run_classify_playback(long detectorAddr, Bitmap obj_bitmap, int rotation) {
        return playbackclassify(detectorAddr, obj_bitmap, rotation);
    }

    /**
     * A native method that is implemented by the 'native-lib' native library,
     * which is packaged with this application.
     */
    private static native long initClassifier(AssetManager assetManager,
                                              String nativeLogFile,
                                              String model_name,
                                              String m_accelerator,
                                              int num_threads,
                                              String m_logging,
                                              String m_resizing_func,
                                              String m_scale_range,
                                              String m_channel,
                                              int m_rotation,
                                              boolean playback);

    private static native float[] classify(long detectorAddr,
                                           byte[] srcAddr,
                                           int width,
                                           int height,
                                           int rotation);

    private static native float[] playbackclassify(long detectorAddr,
                                                   Bitmap obj_bitmap,
                                                   int rotation);

}
