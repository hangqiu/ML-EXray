package com.edgeml.monitor.hwinfo;

import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.util.Pair;

import com.edgeml.monitor.env.Logger;

import java.util.HashMap;
import java.util.Map;

public class SensorListener implements SensorEventListener {
    private final SensorManager mSensorManager;
    private static final Logger LOGGER = new Logger();
//    private final Sensor mAccelerometer;
//    private final Sensor mGyroscope;

    Map<String, Pair<Long, float[]>> sensorCache = new HashMap<>();
    Map<String, String> sensorStringCache = new HashMap<>();

    public SensorListener(SensorManager sensor_manager) {
        mSensorManager = sensor_manager;

        for (Sensor sensor : mSensorManager.getSensorList(Sensor.TYPE_ALL)) {
            LOGGER.d("Registering " + sensor.getName());
            mSensorManager.registerListener(this, sensor, 0);
        }
    }

    public Map<String, Pair<Long, float[]>> getSensor(){
        return sensorCache;
    }

    public Map<String, String> getSensorStringCache(){
        return sensorStringCache;
    }

    public String getSensorString(){
        Map<String, String> sensorString = getSensorStringCache();
        String ret1 = "";
        for (Map.Entry<String, String> entry : sensorString.entrySet()){
            ret1 = ret1 + entry.getKey() + ": " + entry.getValue() + "\n";
        }
        return ret1;
    }

    @Override
    public void onSensorChanged(SensorEvent sensorEvent) {
//        LOGGER.d("Logging " + sensorEvent.sensor.getName());
        sensorCache.put(sensorEvent.sensor.getName(), new Pair<Long, float[]>(sensorEvent.timestamp ,sensorEvent.values));
        sensorStringCache.put(sensorEvent.sensor.getName(), sensorString(sensorEvent));
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int i) {

    }

    public String sensorString(SensorEvent evt){
        String sensorContent ="";
        switch(evt.sensor.getType()) {
            case Sensor.TYPE_ACCELEROMETER:
                sensorContent = String.format("%d; ACC; %f; %f; %f; %f; %f; %f\n", evt.timestamp, evt.values[0], evt.values[1], evt.values[2], 0.f, 0.f, 0.f);
                break;
            case Sensor.TYPE_GYROSCOPE_UNCALIBRATED:
                sensorContent = String.format("%d; GYRO_UN; %f; %f; %f; %f; %f; %f\n", evt.timestamp, evt.values[0], evt.values[1], evt.values[2], evt.values[3], evt.values[4], evt.values[5]);
                break;
            case Sensor.TYPE_GYROSCOPE:
                sensorContent = String.format("%d; GYRO; %f; %f; %f; %f; %f; %f\n", evt.timestamp, evt.values[0], evt.values[1], evt.values[2], 0.f, 0.f, 0.f);
                break;
            case Sensor.TYPE_MAGNETIC_FIELD:
                sensorContent = String.format("%d; MAG; %f; %f; %f; %f; %f; %f\n", evt.timestamp, evt.values[0], evt.values[1], evt.values[2], 0.f, 0.f, 0.f);
                break;
            case Sensor.TYPE_MAGNETIC_FIELD_UNCALIBRATED:
                sensorContent = String.format("%d; MAG_UN; %f; %f; %f; %f; %f; %f\n", evt.timestamp, evt.values[0], evt.values[1], evt.values[2], 0.f, 0.f, 0.f);
                break;
            case Sensor.TYPE_ROTATION_VECTOR:
                sensorContent = String.format("%d; ROT; %f; %f; %f; %f; %f; %f\n", evt.timestamp, evt.values[0], evt.values[1], evt.values[2], evt.values[3], 0.f, 0.f);
                break;
            case Sensor.TYPE_GAME_ROTATION_VECTOR:
                sensorContent = String.format("%d; GAME_ROT; %f; %f; %f; %f; %f; %f\n", evt.timestamp, evt.values[0], evt.values[1], evt.values[2], evt.values[3], 0.f, 0.f);
                break;
                // environment sensors
            case Sensor.TYPE_AMBIENT_TEMPERATURE:
                sensorContent = String.format("%d; TEMP; %f\n", evt.timestamp, evt.values[0]);
            case Sensor.TYPE_LIGHT:
                sensorContent = String.format("%d; LIGHT; %f\n", evt.timestamp, evt.values[0]);
                break;
            case Sensor.TYPE_RELATIVE_HUMIDITY:
                sensorContent = String.format("%d; HUMID; %f\n", evt.timestamp, evt.values[0]);
                break;
            case Sensor.TYPE_PRESSURE:
                sensorContent = String.format("%d; Pressure; %f\n", evt.timestamp, evt.values[0]);
                break;
            default:
                sensorContent = "N/A";
        }
        return sensorContent;

    }

}

