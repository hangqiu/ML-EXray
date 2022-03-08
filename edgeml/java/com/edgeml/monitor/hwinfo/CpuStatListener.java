package com.edgeml.monitor.hwinfo;

import android.os.Build;
import android.os.CpuUsageInfo;
import android.os.HardwarePropertiesManager;

import com.edgeml.monitor.env.Logger;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.Map;




public class CpuStatListener {

    public class CpuStat{
        int cpuid = -1;
        long usertime=0;
        long nicetime=0;
        long systemtime=0;
        long idletime=0;
        long ioWait=0;
        long irq=0;
        long softIrq=0;
        long steal=0;
        long guest=0;
        long guestnice=0;

        // abstract
        long idlealltime = 0;
        long systemalltime = 0;
        long virtalltime = 0;
        long totaltime = 0;

        private long wrap_subtract(long a, long b){
            if (a>b) return a-b;
            else return 0;
        }

        public CpuStat subtract(CpuStat b){
            CpuStat ret = new CpuStat();
            ret.usertime = wrap_subtract(usertime, b.usertime);
            ret.nicetime = wrap_subtract(nicetime, b.nicetime);
            ret.systemtime = wrap_subtract(systemtime, b.systemtime);
            ret.idletime = wrap_subtract(idletime, b.idletime);
            ret.ioWait = wrap_subtract(ioWait, b.ioWait);
            ret.irq = wrap_subtract(irq, b.irq);
            ret.softIrq = wrap_subtract(softIrq, b.softIrq);
            ret.steal = wrap_subtract(steal, b.steal);
            ret.guest = wrap_subtract(guest, b.guest);
            ret.guestnice = wrap_subtract(guestnice, b.guestnice);
            ret.idlealltime = wrap_subtract(idlealltime, b.idlealltime);
            ret.systemalltime = wrap_subtract(systemalltime, b.systemalltime);
            ret.virtalltime = wrap_subtract(virtalltime, b.virtalltime);
            ret.totaltime = wrap_subtract(totaltime, b.totaltime);
            return ret;
        }

        public Boolean readFromStat(String line){

            String[] data = line.split(" ");
            if (!data[0].contains("cpu")) return false;
            if (data[0] == "cpu") return false;
            if (data.length<11) return false;

            cpuid = Integer.parseInt(String.valueOf(data[0].charAt(3)));
            usertime=Long.parseLong(data[1]);
            nicetime=Long.parseLong(data[2]);
            systemtime=Long.parseLong(data[3]);
            idletime=Long.parseLong(data[4]);
            ioWait=Long.parseLong(data[5]);
            irq=Long.parseLong(data[6]);
            softIrq=Long.parseLong(data[7]);
            steal=Long.parseLong(data[8]);
            guest=Long.parseLong(data[9]);
            guestnice=Long.parseLong(data[10]);
            // internalize
            usertime = usertime - guest; // Guest time is already accounted in usertime
            nicetime = nicetime - guestnice;
            idlealltime = idletime + ioWait;
            systemalltime = systemtime + irq + softIrq;
            virtalltime = guest + guestnice;
            totaltime = usertime + nicetime + systemalltime + idlealltime + steal + virtalltime;
            return true;
        }
    }
    private Map<String, Long> cpuinfo = new HashMap<String, Long>();
    private Map<Integer, Float> CpuUsage = new HashMap<Integer, Float>();
    private Map<Integer, CpuStat> cpuStats = new HashMap<Integer, CpuStat>();
    private Logger LOGGER = new Logger();




    public Map<Integer, Float> getCpuUsage() {
//        LOGGER.d("Getting Cpu Usage");
        CpuUsage = new HashMap<Integer, Float>();
        String[] command = {"/system/bin/cat", "/proc/stat"};
        ProcessBuilder processBuilder = new ProcessBuilder(command);
        try{
            Process process = processBuilder.start();
            InputStream inputStream = process.getInputStream();

//            byte[] byteArry = new byte[1024];
//            String Holder = "";
//            while(inputStream.read(byteArry) != -1){
//
//                Holder = Holder + new String(byteArry);
//            }

            BufferedReader r = new BufferedReader(new InputStreamReader(inputStream));

//        File stat = new File("/proc/stat");
//        BufferedReader br = new BufferedReader(new FileReader(stat));

//            LOGGER.d("Getting Cpu Usage");
            for (String line; (line = r.readLine()) != null; ) {
                LOGGER.e(line);
                CpuStat c = new CpuStat();
                if (!c.readFromStat(line)) continue;
                calculateCpuUsage(c);
                cpuStats.put(c.cpuid, c);
            }
//        br.close();
        }catch (IOException e){
            e.printStackTrace();
        }

        return CpuUsage;
    }

    public void calculateCpuUsage(CpuStat c){
        if (cpuStats.containsKey(c.cpuid)){
            CpuStat cpuUsagePeriod = c.subtract(cpuStats.get(c.cpuid));
            float usage_percent = (float) (100.0 - (double)cpuUsagePeriod.idlealltime / (double)cpuUsagePeriod.totaltime * 100.0);
            CpuUsage.put(c.cpuid, usage_percent);
        }
    }

    public String getCpuUsageString() throws IOException {

        getCpuUsage();
        String ret = "";
        for (Map.Entry<Integer, Float> entry : CpuUsage.entrySet()){
            ret += entry.getKey() + ": " + entry.getValue() + "\n";
        }
        return ret;
    }
}
