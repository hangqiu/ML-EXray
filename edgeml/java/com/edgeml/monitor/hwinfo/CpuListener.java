package com.edgeml.monitor.hwinfo;

import android.os.Build;
import android.os.CpuUsageInfo;
import android.os.HardwarePropertiesManager;

//import androidx.annotation.RequiresApi;

import java.util.HashMap;
import java.util.Map;

import static android.os.HardwarePropertiesManager.DEVICE_TEMPERATURE_CPU;
import static android.os.HardwarePropertiesManager.TEMPERATURE_CURRENT;

public class CpuListener {
    private Map<String, String> cpuinfo = new HashMap<>();

    private HardwarePropertiesManager mHWMgr;

    private long lastActive = 0;
    private long lastTotal = 0;

    Map<Integer, Float> CpuUsage = new HashMap<>();
    Map<Integer, Float> CpuTemp = new HashMap<>();

    public CpuListener(HardwarePropertiesManager mHWMgr_){
        mHWMgr = mHWMgr_;

    }

//    @RequiresApi(api = Build.VERSION_CODES.N)
    public Map<String, String> getCpuInfo(){
        if (!(cpuinfo.containsKey("abi"))){
            cpuinfo.put("abi", getAbi());
        }
        if (!(cpuinfo.containsKey("cores"))){
            cpuinfo.put("cores", getNumberOfCores().toString());
        }
        return cpuinfo;
    }

//    @RequiresApi(api = Build.VERSION_CODES.N)
    public String getCpuInfoString(){
        Map<String, String> mCpuInfo = getCpuInfo();
        String ret = "";
        for (Map.Entry<String, String> entry : mCpuInfo.entrySet()){
            ret = ret + entry.getKey() + ": " + entry.getValue() + "\n";
        }
        return ret;
    }

    private String getAbi(){
        if (Build.VERSION.SDK_INT >= 21) {
            return  Build.SUPPORTED_ABIS[0];
        } else return Build.CPU_ABI;
    }

    private Integer getNumberOfCores() {
        return Runtime.getRuntime().availableProcessors();
    }

//    @RequiresApi(api = Build.VERSION_CODES.N)
    private void getCpuUsage(){
        String ret = "";
        CpuUsageInfo[] cpuinfo = mHWMgr.getCpuUsages();
        for (int i=0; i<cpuinfo.length; i++)
        {
            CpuUsageInfo coreUsage = cpuinfo[i];
//            String cpu_tmp = String.format("Core %d: %d / %d \n", i, coreUsage.getActive(), coreUsage.getTotal());
            Float active = (float)coreUsage.getActive();
            Float total = (float)coreUsage.getTotal();
            CpuUsage.put(i, (( active / total )*100 ));
        }
    }



//    @RequiresApi(api = Build.VERSION_CODES.N)
    private void getCpuTemp(){
        float[] cputemp = mHWMgr.getDeviceTemperatures(DEVICE_TEMPERATURE_CPU, TEMPERATURE_CURRENT);
        for (int i=0; i<cputemp.length; i++){
            CpuTemp.put(i, cputemp[i]);
        }
    }

    public String getCpuUsageString(){
        getCpuUsage();
        getCpuTemp();
        String ret="";
        for (Map.Entry<Integer, Float> entry : CpuUsage.entrySet()){
            Integer coreId = entry.getKey();
            ret = ret + coreId.toString() + ": " + entry.getValue();
            if (CpuTemp.containsKey(coreId)){
                ret = ret + String.format(", %f C", CpuTemp.get(coreId));
            }
            ret = ret + "\n";
        }
        return ret;
    }
}
