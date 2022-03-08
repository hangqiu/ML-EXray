package com.edgeml.edgemlinsight;

import android.content.res.AssetManager;
import android.graphics.Bitmap;

public class NativeDetector {
    static {
        try {
            System.loadLibrary("opencv_java3");
        } catch (java.lang.UnsatisfiedLinkError e) {
            // Some example apps (e.g. template matching) require OpenCV 4.
            System.loadLibrary("opencv_java4");
        }
        System.loadLibrary("objectdetector");
    }


    public NativeDetector(){

    }

    public long init(
            AssetManager assetManager,
            String nativeLogFile,
            String model_name,
            String m_accelerator,
            int num_threads,
            String m_logging,
            String m_resizing_func,
            String m_scale_range,
            String m_channel,
            int m_rotation,
            boolean playback){
        return initDetector(
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

    public float[] run_detect(long detectorAddr,byte[] srcAddr,int width,int height,int rotation){
        return detect(detectorAddr, srcAddr, width, height, rotation);
    }

    public float[] run_detect_playback(long detectorAddr, Bitmap obj_bitmap){
        return playbackdetect(detectorAddr, obj_bitmap);
    }

    /**
     * A native method that is implemented by the 'native-lib' native library,
     * which is packaged with this application.
     */
    private static native long initDetector(AssetManager assetManager,
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
    private static native float[] detect(long detectorAddr,
                                         byte[] srcAddr,
                                         int width,
                                         int height,
                                         int rotation);
    private static native float[] playbackdetect(long detectorAddr,
                                         Bitmap obj_bitmap);
}
