package com.edgeml.edgemlinsight;

import android.graphics.*;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;

//import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import com.edgeml.monitor.monitor.EdgeMLMonitor;
//import com.google.android.gms.tasks.OnFailureListener;
//import com.google.android.gms.tasks.OnSuccessListener;
import com.otaliastudios.cameraview.CameraView;
import com.otaliastudios.cameraview.frame.Frame;
import com.otaliastudios.cameraview.frame.FrameProcessor;

import org.tensorflow.lite.support.common.FileUtil;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;


public class DetectorActivity extends AppCompatActivity {

    private String TAG = "DetectorActivity";
    private long detectorAddr = 0L;
    private int frameWidth = 0;
    private int frameHeight = 0;
    private Paint _paint = new Paint();
    private List<String> labelsMap;
    //    private AssetManager assetManager;
    private SurfaceView surfaceView;
    private SurfaceHolder surfaceHolder;
    private CameraView cameraView;

    private EdgeMLMonitor mlMonitor;
    private int num_threads = 4;
    private int rotation = 0;

    private String model_name = "";
    private String data_name = "";


    //CONFIGS
    private String m_accelerator = "";
    private String m_num_threads = "";
    private String m_logging = "";
    private String m_resizing_func = "";
    private String m_scale_range = "";
    private String m_channel = "";
    private String m_rotation = "";

    private NativeDetector mDetector = new NativeDetector();

    static {
        try {
            System.loadLibrary("opencv_java3");
        } catch (java.lang.UnsatisfiedLinkError e) {
            // Some example apps (e.g. template matching) require OpenCV 4.
            System.loadLibrary("opencv_java4");
        }
    }

    @Override
    protected void onCreate(final Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_detector);

        model_name = getIntent().getStringExtra(Utils.ExtraIntentVar.EXTRA_MODEL_NAME.toString());
        data_name = getIntent().getStringExtra(Utils.ExtraIntentVar.EXTRA_DATA_NAME.toString());

        m_accelerator = getIntent().getStringExtra(Utils.ExtraIntentVar.EXTRA_ACCELERATOR_CHOICE.toString());
        m_num_threads = getIntent().getStringExtra(Utils.ExtraIntentVar.EXTRA_NUM_THREAD.toString());
        m_logging = getIntent().getStringExtra(Utils.ExtraIntentVar.EXTRA_LOGGING_CHOICE.toString());
        m_resizing_func = getIntent().getStringExtra(Utils.ExtraIntentVar.EXTRA_RESIZING_FUNC.toString());
        m_scale_range = getIntent().getStringExtra(Utils.ExtraIntentVar.EXTRA_SCALE_RANGE.toString());
        m_channel = getIntent().getStringExtra(Utils.ExtraIntentVar.EXTRA_CHANNEL_CHOICE.toString());
        m_rotation = getIntent().getStringExtra(Utils.ExtraIntentVar.EXTRA_ROTATION.toString());

        num_threads = Integer.parseInt(m_num_threads);
        rotation = Integer.parseInt(m_rotation);

        Log.d(TAG, "Initializing model: " + model_name + " on " + data_name);


        try {
            mlMonitor = new EdgeMLMonitor(this, "Detection", model_name, data_name);
        } catch (IOException e) {
            e.printStackTrace();
        }

        cameraView = findViewById(R.id.DetectorCamera);
        cameraView.setLifecycleOwner(this);
        surfaceView = findViewById(R.id.DetectorSurfaceView);
        surfaceHolder = surfaceView.getHolder();
        Log.d(TAG, "Initializaing native edgeml monitor: " + mlMonitor.nativeLogDir);
        detectorAddr = mDetector.init(
                getAssets(),
                mlMonitor.baseLogDir,
                model_name,
                m_accelerator,
                num_threads,
                m_logging,
                m_resizing_func,
                m_scale_range,
                m_channel,
                rotation,
                false);

        FrameProcessor processor = new FrameProcessor() {
            @Override
            public void process(Frame frame) {
                detectObjectNative(frame);
            }
        };
        cameraView.addFrameProcessor(processor);

        // init the paint for drawing the detections
        _paint.setColor(Color.RED);
        _paint.setStyle(Paint.Style.STROKE);
        _paint.setStrokeWidth(3f);
        _paint.setTextSize(50f);
        _paint.setTextAlign(Paint.Align.LEFT);

        // Set the detections drawings surface transparent
        surfaceView.setZOrderOnTop(true);
        surfaceHolder.setFormat(PixelFormat.TRANSPARENT);

        try {
            labelsMap = Utils.loadLabels(this, model_name);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void detectObjectNative(Frame frame) {
        if (!m_logging.equals(Utils.LoggingChoice.NONE.toString())) {
            mlMonitor.onSensorStop();
        }

        if (frameWidth == 0) {
            frameWidth = frame.getSize().getWidth();
            frameHeight = frame.getSize().getHeight();
        }

        long start = System.currentTimeMillis();
        if (!m_logging.equals(Utils.LoggingChoice.NONE.toString())) {
            mlMonitor.onInferenceStart();
        }
        float[] res = mDetector.run_detect(
                detectorAddr,
                frame.getData(),
                frame.getSize().getWidth(),
                frame.getSize().getHeight(),
                frame.getRotationToUser()
        );

        String InfResutls = Arrays.toString(res);
        if (!m_logging.equals(Utils.LoggingChoice.NONE.toString())) {
            mlMonitor.onInferenceStop(InfResutls);
        }

        long span = System.currentTimeMillis() - start;
        Log.i(TAG, String.format("Detection span: %d ms", span));

        Canvas canvas = surfaceHolder.lockCanvas();
        if (canvas != null) {
            canvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.MULTIPLY);
            // Draw the detections, in our case there are only 3
            drawDetection(canvas, frame.getRotationToUser(), res, 0);
            drawDetection(canvas, frame.getRotationToUser(), res, 1);
            drawDetection(canvas, frame.getRotationToUser(), res, 2);
            surfaceHolder.unlockCanvasAndPost(canvas);

        }
        // new frame will not be recorded until this function returns
        if (!m_logging.equals(Utils.LoggingChoice.NONE.toString())) {
            mlMonitor.onSensorStart();
        }
    }

    private void drawDetection(
            Canvas canvas,
            int rotation,
            float[] detectionsArr,
            int detectionIdx
    ) {
        // Filter by score
        float score = detectionsArr[detectionIdx * 6 + 1];
        if (score < 0.6) return;

        // Get the frame dimensions
        int w = 0;
        int h = 0;
        if (rotation == 0 || rotation == 180) {

            w = frameWidth;
            h = frameHeight;
        } else {
            w = frameHeight;
            h = frameWidth;
        }

        // detection coords are in frame coord system, convert to screen coords
        float scaleX = (float) cameraView.getWidth() / w;
        float scaleY = (float) cameraView.getHeight() / h;

        // The camera view offset on screen
        float xoff = (float) cameraView.getLeft();
        float yoff = (float) cameraView.getTop();

        float classId = detectionsArr[detectionIdx * 6 + 0];
        float xmin = xoff + detectionsArr[detectionIdx * 6 + 2] * scaleX;
        float xmax = xoff + detectionsArr[detectionIdx * 6 + 3] * scaleX;
        float ymin = yoff + detectionsArr[detectionIdx * 6 + 4] * scaleY;
        float ymax = yoff + detectionsArr[detectionIdx * 6 + 5] * scaleY;


        // Draw the rect
        Path p = new Path();
        p.moveTo(xmin, ymin);
        p.lineTo(xmax, ymin);
        p.lineTo(xmax, ymax);
        p.lineTo(xmin, ymax);
        p.lineTo(xmin, ymin);

        canvas.drawPath(p, _paint);

        // SSD Mobilenet Model assumes class 0 is background class and detection result class
        // are zero-based (meaning class id 0 is class 1)
        String label = labelsMap.get((int) classId + 1);

        String txt = String.format("%s (%.2f)", label, score);
        canvas.drawText(txt, xmin, ymin, _paint);
    }
}
