package com.edgeml.monitor.hwinfo;

import android.hardware.SensorManager;
import android.os.HardwarePropertiesManager;

import com.edgeml.monitor.env.Logger;

import java.util.HashMap;
import java.util.Map;

public class Hwinfo {

    private Map<String, String> mHWInfo = new HashMap<>();

    private SensorListener mSensor;
    private CpuListener mCpu;

    private SensorManager mSensorMgr;
    private static final Logger LOGGER = new Logger();

    public Hwinfo(SensorManager sensor_mgr, HardwarePropertiesManager hw_mgr){
        mSensorMgr = sensor_mgr;
        mSensor = new SensorListener(sensor_mgr);
        mCpu = new CpuListener(hw_mgr);

    }

    public String getHWInfoString(){

        String sensorString = mSensor.getSensorString();
        String cpuString = mCpu.getCpuInfoString();

        return cpuString + "\n" + sensorString;
    }


}
