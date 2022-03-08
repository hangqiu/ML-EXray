package com.edgeml.monitor.monitor;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.Context;
import android.graphics.Bitmap;
import android.hardware.SensorManager;
import android.os.Environment;
import android.os.Handler;
import android.os.HardwarePropertiesManager;
import android.os.SystemClock;
import android.util.Log;
import android.util.Pair;

import com.edgeml.monitor.hwinfo.CpuListener;
import com.edgeml.monitor.hwinfo.CpuStatListener;
import com.edgeml.monitor.hwinfo.Hwinfo;
import com.edgeml.monitor.hwinfo.SensorListener;
import com.edgeml.monitor.env.Logger;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.time.Clock;
import java.util.Calendar;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;
import java.util.Arrays;

//import io.opentelemetry.api.OpenTelemetry;
//import io.opentelemetry.api.trace.Span;
//import io.opentelemetry.api.trace.Tracer;
//import io.opentelemetry.context.Scope;


public class EdgeMLMonitor {

    // variables
    private long InferenceLatencyStart;
    private long InferenceLatencyStop;
    private Context mContext;
    private Runnable sensorLogger;
    private Handler SensorLoggerHandler = new Handler();
//    private Runnable cpuLogger;
    private Handler CpuLoggerHandler = new Handler();
    private int SensorLoggingInterval = 30; // ms
    private int CpuLoggingInterval = 10; // ms
    private int InferenceLogID = 0;
    private int sensorLogID = 0;
    private static final Logger LOGGER = new Logger();
    private Boolean isSensorLogging = false;
    private Boolean isInferenceLogging = false;
    private Bitmap mInputImage = null;
    private Boolean isInitialized = false;
    private static final String PERMISSION_CAMERA = Manifest.permission.CAMERA;
    private static final String PERMISSION_STORAGE_WRITE = Manifest.permission.WRITE_EXTERNAL_STORAGE;
    public String baseLogDir;
    private String logDir;
    private String inputDir;
    public String nativeLogDir;
    public String nativeInputDir;
    public String nativeLogFile;
    private String data_dir;

    //    Open Telemetry
//    private OpenTelemetry mOpenTelemetry;
//    private Tracer tracer;

    // hwinfo
    private SensorListener mSensor;
    private CpuStatListener mCpu;
    private SensorManager mSensorManager;
//    private HardwarePropertiesManager mHWManager;


    // Metrics
    private long InfernceLatency_ms = 0;
    private String mInferenceResults = "";
    private Map<String, Map<Long, float[]>> SensorLog = new HashMap<String, Map<Long, float[]>>();
    private Map<Long, String> CpuLog = new HashMap<Long, String>();
    private String LatestSensorLogString = "";
    private String LatestInferenceLogString = "";

    public EdgeMLMonitor(Context ctx, String task_name, String model_name) throws IOException {
        this(ctx, task_name, model_name,"");
    }

    public EdgeMLMonitor(Context ctx, String task_name, String model_name, String data_name) throws IOException {
        mContext = ctx;
        data_dir = data_name;
//        mOpenTelemetry = otl;
//        tracer = mOpenTelemetry.getTracer("edgeml-insight");
        // check permission
        mSensorManager = (SensorManager)mContext.getSystemService(Context.SENSOR_SERVICE);
//        mHWManager = (HardwarePropertiesManager)mContext.getSystemService(Context.HARDWARE_PROPERTIES_SERVICE);
        mSensor = new SensorListener(mSensorManager);
        mCpu = new CpuStatListener();
        init_runnables();
        isInitialized = true;

        Date currentTime = Calendar.getInstance().getTime();
        String initTime = currentTime.toString().replaceAll(" ", "_").replaceAll(":", "-");
        if (data_name.equals("")){
            baseLogDir = "/sdcard/edgeml/" + initTime + "_" + task_name;
        }else{
            baseLogDir = data_name + "_playback_at_" + initTime + "_" + task_name;
        }
        logDir = baseLogDir + "/log/";
        inputDir = baseLogDir  + "/input/";
        nativeLogDir = baseLogDir + "/nativeLog/";
        nativeInputDir = baseLogDir + "/nativeInput/";
        File logDirFile = new File(logDir);
        File inputDirFile = new File(inputDir);
        File nativeLogDirFile = new File(nativeLogDir);
        File nativeInputDirFile = new File(nativeInputDir);
        if (!logDirFile.exists()) {
            LOGGER.d("Creating " + logDir);
            logDirFile.mkdirs();
            LOGGER.d("Creating " + inputDir);
            inputDirFile.mkdirs();
            LOGGER.d("Creating " + nativeLogDir);
            nativeLogDirFile.mkdirs();
            LOGGER.d("Creating " + nativeInputDir);
            nativeInputDirFile.mkdirs();
        }
        nativeLogFile = nativeLogDir + "/init.log";

        // log all meta data
        File metalog = new File(logDir + "/meta.log");
        if (metalog.createNewFile()){
            BufferedWriter buf = new BufferedWriter(new FileWriter(metalog, true));
            buf.append(task_name);
            buf.newLine();
            buf.append(model_name);
            buf.newLine();
            buf.append(data_name);
            buf.newLine();
            buf.close();
        }

    }

    public void init_runnables(){
        sensorLogger = new Runnable() {
            @Override
            public void run() {
//                Span SensorSpan = tracer.spanBuilder("SensorSpan").startSpan();
//                try (Scope scope = SensorSpan.makeCurrent()){
                try{
                    Map<String, Pair<Long, float[]>> sensor = mSensor.getSensor();
                    for (Map.Entry<String, Pair<Long, float[]>> entry : sensor.entrySet()){
                        if (!SensorLog.containsKey(entry.getKey())){
                            SensorLog.put(entry.getKey(), new HashMap<Long, float[]>());
                        }

                        if (!(SensorLog.get(entry.getKey()).containsKey(entry.getValue().first))){
                            SensorLog.get(entry.getKey()).put(entry.getValue().first, entry.getValue().second);
                        }
                    }
                    // opentelemetry
//                    String sensorString = mSensor.getSensorString();
//                    SensorSpan.setAttribute("sensor", sensorString);

                }finally {
//                    SensorSpan.end();
                    SensorLoggerHandler.postDelayed(sensorLogger, SensorLoggingInterval);
                }
            }
        };

//        cpuLogger = new Runnable() {
//            @Override
//            public void run() {
////                Span InferenceSpan = tracer.spanBuilder("InferenceSpan").startSpan();
////                try (Scope scope = InferenceSpan.makeCurrent()){
//                try{
//                    long time = SystemClock.uptimeMillis();
//                    String tmp = mCpu.getCpuUsageString();
//                    CpuLog.put(time,tmp);
//                }catch (IOException e){
//                    e.printStackTrace();
//                }
//                finally {
////                    InferenceSpan.end();
//                    CpuLoggerHandler.postDelayed(cpuLogger, CpuLoggingInterval);
//                }
//            }
//        };
    }

    public void onSensorStart(){
        if (!isInitialized) return;
        if (isSensorLogging)return;

        // reinitialize relevant metrics
        SensorLog = new HashMap<String, Map<Long, float[]>>();

        // Logging start
        LOGGER.d("onSensorStart!");
        isSensorLogging = true;
        sensorLogger.run();

    }

    public void onSensorStop(){
        if (!isSensorLogging)return;
        LOGGER.d("onSensorStop!");
        isSensorLogging = false;
        SensorLoggerHandler.removeCallbacks(sensorLogger);
        LatestSensorLogString = getSensorLogString();
        appendSensorLog();

    }

    public void onInferenceStart(){
        if (!isInitialized) return;
        if (isInferenceLogging)return;

        // reinitialize private metrics
        InfernceLatency_ms = 0;
        mInputImage = null;
        CpuLog = new HashMap<Long, String>();

        // logging begins
        LOGGER.d("onInferenceStart!");
        isInferenceLogging = true;
//        cpuLogger.run();
        InferenceLatencyStart = SystemClock.uptimeMillis();

    }

    public void onInferenceStart(Bitmap inputImage){
        onInferenceStart();
        mInputImage = inputImage;
    }

    public void onInferenceStop(){
        if (!isInferenceLogging)return;
        LOGGER.d("onInferenceStop!");
        isInferenceLogging = false;
        InferenceLatencyStop = SystemClock.uptimeMillis();
//        CpuLoggerHandler.removeCallbacks(cpuLogger);
        LatestInferenceLogString = getInferenceLogString();
        appendInferenceLog();

    }

    public void onInferenceStop(String InferenceResults){
        mInferenceResults = InferenceResults;
        onInferenceStop();
    }

    public String getSensorLogString(){
        Map<String, Map<Long, float[]>> tmpSensorLog = SensorLog;
        String ret = "";
        for (Map.Entry<String, Map<Long, float[]>> entry : tmpSensorLog.entrySet()){
            ret = ret + entry.getKey() + "\n";
            for (Map.Entry<Long, float[]> data : entry.getValue().entrySet()){
                ret = ret + "\t" + data.getKey().toString() + ": " + Arrays.toString(data.getValue()) + "\n";
            }
        }
        return ret;
    }

    public String getInferenceLogString(){
        InfernceLatency_ms = InferenceLatencyStop - InferenceLatencyStart;
        @SuppressLint("DefaultLocale")
        String ret = String.format("Inference Start Time: %d ms\n" +
                "Inference Time: %d ms\n" +
                "Inference Result: %s\n",
                InferenceLatencyStart, InfernceLatency_ms, mInferenceResults);
//        ret += "Cpu Usage\n";
//        for (Map.Entry<Long, String> data : CpuLog.entrySet()){
//            ret += data.getKey().toString() + ": " + data.getValue() + "\n";
//        }
        return ret;
    }

    public void saveInputImage() {
        if (mInputImage!=null){
            LOGGER.d("Saving Input Image");
            File imageFile = new File(String.format("%s/%d.png", inputDir, InferenceLogID));
            try{
                FileOutputStream fOut = new FileOutputStream(imageFile);
                mInputImage.compress(Bitmap.CompressFormat.PNG,100, fOut);
                fOut.flush();
                fOut.close();
            }
            catch (IOException e){
                e.printStackTrace();
            }
        }else{
            LOGGER.d("Input Image NULL");
        }
    }

    public void saveInferenceLog(){
        saveLog(LatestInferenceLogString, InferenceLogID);
    }

    public void saveSensorLog(){
        saveLog(LatestSensorLogString, sensorLogID);
    }

    public void saveLog(String log, int LogID){
        String fileDir = String.format("%s/%d.log", logDir, LogID);
        File logFile = new File(fileDir);
        if (!logFile.exists())
        {
            try
            {
                LOGGER.d("Creating " + fileDir);
                logFile.createNewFile();
            }
            catch (IOException e)
            {
                e.printStackTrace();
            }
        }
        try
        {
            //BufferedWriter for performance, true to set append to file flag
            BufferedWriter buf = new BufferedWriter(new FileWriter(logFile, true));
            buf.append(log);
            buf.newLine();
            buf.close();
        }
        catch (IOException e)
        {
            e.printStackTrace();
        }
    }

    public void appendInferenceLog() {
        if (data_dir.equals("")){
            // save image for non-playback traces
            saveInputImage();
        }
        saveInferenceLog();
        // increment log id only after each inference
        InferenceLogID = InferenceLogID + 1;
    }

    public void appendSensorLog(){
        saveSensorLog();
        // increment log id only after each inference
        sensorLogID = sensorLogID + 1;
    }
}
