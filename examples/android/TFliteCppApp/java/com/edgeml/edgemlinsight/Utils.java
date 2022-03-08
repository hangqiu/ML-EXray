package com.edgeml.edgemlinsight;

import android.content.Context;
import android.util.Log;

import org.tensorflow.lite.support.common.FileUtil;

import java.io.IOException;
import java.util.List;

public class Utils {
    public static String TAG = "Utils";
    public static final String Label_imagenet = "classification/imagenet_label.txt";
    public static final String Label_coco = "detection/coco_label.txt";

    public static List<String> loadLabels(Context context, String model_name) throws IOException {

        if (model_name.contains("coco")){
            return FileUtil.loadLabels(context, Utils.Label_coco);
        }
        if (model_name.contains("imagenet")){
            return FileUtil.loadLabels(context, Utils.Label_imagenet);
        }

        Log.e(TAG, "Data label not supported!");
        return null;
    }

    public enum AcceleratorChoice{
        CPU,
        GPU,
        NNAPI
    }

    public enum LoggingChoice{
        NONE,
        IO,
        EMBEDDING,
        PERLAYER
    }

    public enum ResizingChoice{
        AVG_AREA,
        BILINEAR,
    }

    public enum ChannelChoice{
        RGB,
        BGR,
    }
    public enum ScaleRangeChoice{
        MINUSONE_ONE,
        ZERO_ONE,
    }

    public enum ExtraIntentVar{
        EXTRA_TASK_NAME,
        EXTRA_MODEL_NAME,
        EXTRA_DATA_NAME,
        EXTRA_ACCELERATOR_CHOICE,
        EXTRA_NUM_THREAD,
        EXTRA_LOGGING_CHOICE,
        EXTRA_RESIZING_FUNC,
        EXTRA_SCALE_RANGE,
        EXTRA_CHANNEL_CHOICE,
        EXTRA_ROTATION
    }
}


