package com.edgeml.edgemlinsight;

import android.graphics.Color;
import android.graphics.Paint;
import android.os.Bundle;
import android.util.Log;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import com.edgeml.monitor.monitor.EdgeMLMonitor;
import com.otaliastudios.cameraview.CameraView;
import com.otaliastudios.cameraview.frame.Frame;
import com.otaliastudios.cameraview.frame.FrameProcessor;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

public class ClassifierActivity extends AppCompatActivity {

    private String TAG = "ClassifierActivity";
    private long detectorAddr = 0L;
    private int frameWidth = 0;
    private int frameHeight = 0;
    private Paint _paint = new Paint();
    private List<String> labelsMap;

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


    private NativeClassifier mClassifier = new NativeClassifier();

    protected TextView recognition0TextView,
            recognition1TextView,
            recognition2TextView,
            recognition0ValueTextView,
            recognition1ValueTextView,
            recognition2ValueTextView;

    static {
        try {
            System.loadLibrary("opencv_java3");
        } catch (UnsatisfiedLinkError e) {
            // Some example apps (e.g. template matching) require OpenCV 4.
            System.loadLibrary("opencv_java4");
        }
    }

    @Override
    protected void onCreate(final Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_classifier);

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
            mlMonitor = new EdgeMLMonitor(this, "Classification", model_name, data_name);
        } catch (IOException e) {
            e.printStackTrace();
        }

        cameraView = findViewById(R.id.ClassifierCamera);
        cameraView.setLifecycleOwner(this);
        recognition0TextView = findViewById(R.id.detected_item0);
        recognition0ValueTextView = findViewById(R.id.detected_item0_value);
        recognition1TextView = findViewById(R.id.detected_item1);
        recognition1ValueTextView = findViewById(R.id.detected_item1_value);
        recognition2TextView = findViewById(R.id.detected_item2);
        recognition2ValueTextView = findViewById(R.id.detected_item2_value);

        Log.d(TAG, "Initializaing native edgeml monitor: " + mlMonitor.nativeLogDir);
        detectorAddr = mClassifier.init(
                getAssets(),
                mlMonitor.baseLogDir, model_name,
                m_accelerator,
                num_threads,
                m_logging,
                m_resizing_func,
                m_scale_range,
                m_channel,
                rotation, false);

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
        float[] res = mClassifier.run_classify(
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


        // Draw the detections, in our case there are only 3
        drawDetection(res, 0);
        drawDetection(res, 1);
        drawDetection(res, 2);

        // new frame will not be recorded until this function returns
        if (!m_logging.equals(Utils.LoggingChoice.NONE.toString())) {
            mlMonitor.onSensorStart();
        }
    }

    private void drawDetection(
            float[] detectionsArr,
            int detectionIdx
    ) {
        // Filter by score
        float classId = detectionsArr[detectionIdx * 2 + 0];
        float score = detectionsArr[detectionIdx * 2 + 1];

        String label = labelsMap.get((int) classId);

        String label_txt = String.format("%d %s", (int) classId, label);
        String score_txt = String.format("%.2f", score * 100) + "%";
        if (detectionIdx == 0) {
            recognition0TextView.setText(label_txt);
            recognition0ValueTextView.setText(score_txt);
        }
        if (detectionIdx == 1) {
            recognition1TextView.setText(label_txt);
            recognition1ValueTextView.setText(score_txt);
        }
        if (detectionIdx == 2) {
            recognition2TextView.setText(label_txt);
            recognition2ValueTextView.setText(score_txt);
        }

    }
}
