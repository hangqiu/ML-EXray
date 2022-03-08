package com.edgeml.edgemlinsight;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.PixelFormat;
import android.graphics.PorterDuff;
import android.os.Bundle;
import android.os.Handler;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import com.edgeml.monitor.monitor.EdgeMLMonitor;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

public class PlaybackActivity extends AppCompatActivity {

    private String TAG = "PlaybackActivity";
    private String task_name = "";
    private String data_name = "";
    private String model_name = "";
    private int num_threads = 4;
    private int rotation = 0;
    private EdgeMLMonitor mlMonitor;
    private List<String> labelsMap;
    private int frameWidth = 0;
    private int frameHeight = 0;
    private int RotationToUser = 0;

    //CONFIGS
    private String m_accelerator = "";
    private String m_num_threads = "";
    private String m_logging = "";
    private String m_resizing_func = "";
    private String m_scale_range = "";
    private String m_channel = "";
    private String m_rotation = "";

    private SurfaceView surfaceView;
    private SurfaceHolder surfaceHolder;
    private Paint _paint = new Paint();
    private ImageView playback_imageview;
    private TextView playback_textview;
    private long mlModelAddr = 0L;

    private NativeDetector mDetector;
    private NativeClassifier mClassifier;

    private int idx = 0;

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
        task_name = getIntent().getStringExtra(Utils.ExtraIntentVar.EXTRA_TASK_NAME.toString());
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

        setContentView(R.layout.activity_playback);
        playback_imageview = (ImageView) findViewById(R.id.playback);
        playback_textview = (TextView) findViewById(R.id.playback_text);

        try {
            mlMonitor = new EdgeMLMonitor(this, task_name, model_name, data_name);
        } catch (IOException e) {
            e.printStackTrace();
        }
        Log.d(TAG, "Initialized native edgeml monitor: " + mlMonitor.nativeLogDir);
        if (task_name.equals("detection")) {
            mDetector = new NativeDetector();
            mlModelAddr = mDetector.init(
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
                    true);
        }

        if (task_name.equals("classification")) {
            mClassifier = new NativeClassifier();
            mlModelAddr = mClassifier.init(getAssets(), mlMonitor.baseLogDir, model_name,
                    m_accelerator,
                    num_threads,
                    m_logging,
                    m_resizing_func,
                    m_scale_range,
                    m_channel,
                    rotation,
                    true);
        }

        Log.d(TAG, "Initialized ml Model");
        surfaceView = findViewById(R.id.PlaybackSurfaceView);
        surfaceHolder = surfaceView.getHolder();

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

        String inputDir = data_name + "/nativeInput/";
        File img_dir = new File(inputDir);
        String[] imgs = img_dir.list();

        if (imgs == null) {
            Log.d(TAG, "Img loading error! ");
            return;
        }

        Arrays.sort(imgs);
        idx = 0;
        Handler h = new Handler();
        Runnable r = new Runnable() {
            @Override
            public void run() {
                try {
//                    String image_name = String.format("%s.png", idx);
                    String image_name = imgs[idx];
                    run_models(image_name, inputDir);
                } catch (Exception e) {
                    e.printStackTrace();
                } finally {
                    idx++;
                    if (idx < imgs.length) {
                        h.post(this);
//                        h.postDelayed(this, 100);
                    }
                }
            }
        };


        h.post(r);

    }

    private void run_models(String img, String inputDir) {

        Log.d(TAG, "Loading image " + img);
        Bitmap img_bitmap = BitmapFactory.decodeFile(inputDir + img);

        /// Display
        playback_imageview.setImageBitmap(img_bitmap);

        if (frameWidth == 0) {
            frameWidth = img_bitmap.getWidth();
            frameHeight = img_bitmap.getHeight();
        }

        if (task_name.equals("detection")) {
            float[] res;
            if (!m_logging.equals(Utils.LoggingChoice.NONE.toString())) {
                mlMonitor.onInferenceStart();
            }
            res = mDetector.run_detect_playback(
                    mlModelAddr,
                    img_bitmap
            );
            String InfResutls = Arrays.toString(res);
            if (!m_logging.equals(Utils.LoggingChoice.NONE.toString())) {
                mlMonitor.onInferenceStop(InfResutls);
            }
            playback_textview.setText("Detection Result:\n" + InfResutls);


            Canvas canvas = surfaceHolder.lockCanvas();
            if (canvas != null) {
                canvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.MULTIPLY);
                // Draw the detections, in our case there are only 3
                drawDetection(canvas, RotationToUser, res, 0);
                drawDetection(canvas, RotationToUser, res, 1);
                drawDetection(canvas, RotationToUser, res, 2);
                surfaceHolder.unlockCanvasAndPost(canvas);

            } else {
                Log.d(TAG, "Canvas is null");
            }

        }
        // TODO: make abstract class for native detector and classifier
        if (task_name.equals("classification")) {
            float[] res;
            if (!m_logging.equals(Utils.LoggingChoice.NONE.toString())) {
                mlMonitor.onInferenceStart();
            }
            res = mClassifier.run_classify_playback(
                    mlModelAddr,
                    img_bitmap,
                    0
            );
            String InfResutls = Arrays.toString(res);
            if (!m_logging.equals(Utils.LoggingChoice.NONE.toString())) {
                mlMonitor.onInferenceStop(InfResutls);
            }
            playback_textview.setText("Classification Result:\n" + InfResutls);
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
        float scaleX = (float) playback_imageview.getWidth() / w;
        // for square image playback
//        float scaleY = (float) playback_imageview.getHeight() / h;
        float scaleY = scaleX;

        // The camera view offset on screen
        float xoff = (float) playback_imageview.getLeft();
        float yoff = (float) ((float) playback_imageview.getTop() + (float) playback_imageview.getHeight() / 2.0 - (float) playback_imageview.getWidth() / 2.0);

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
