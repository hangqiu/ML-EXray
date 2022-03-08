package com.edgeml.edgemlinsight;

import android.content.Context;
import android.content.Intent;
import android.net.ConnectivityManager;
import android.net.NetworkInfo;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Spinner;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.android.gms.tasks.Task;
import com.google.firebase.storage.FileDownloadTask;
import com.google.firebase.storage.FirebaseStorage;
import com.google.firebase.storage.ListResult;
import com.google.firebase.storage.StorageReference;
import com.google.firebase.storage.UploadTask;
import com.google.android.gms.tasks.OnCompleteListener;


import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.UUID;


public class MainActivity extends AppCompatActivity {
    private String TAG = "MainActivity";
    private StorageReference mStorageRef;
    private String uniqueID = UUID.randomUUID().toString();
    private String rootfile = "/sdcard/edgeml/";
    private String DeviceIDPath = "/sdcard/edgeml/DeviceID.txt";
    private String[] DatasetPath = new String[]{"/sdcard/edgeml/imagenet2012", "/sdcard/edgeml/imagenet2012_10", "/sdcard/edgeml/imagenet2012_100"};
    private List<String> DatasetPathArray = new ArrayList<>(Arrays.asList(DatasetPath));
    private String DeviceID;
    private String notice;
    private String task_name = "";
    private String model_name = "";
    private String data_src_name = "";
    private String data_name = "";

    // Configs
    private String m_accelerator = "";
    private String m_num_threads = "";
    private String m_logging = "";
    private String m_resizing_func = "";
    private String m_scale_range = "";
    private String m_channel = "";
    private String m_rotation = "";

    private Spinner model_choice;
    private ArrayAdapter<String> ModelArrayAdapter;
    private Spinner data_choice;
    private ArrayAdapter<String> DataArrayAdapter;
    private ArrayList<String> local_traces = new ArrayList<String>();
    private ArrayList<String> cloud_traces = new ArrayList<String>();
    private ArrayList<String> classification_model_names = new ArrayList<>();
    private ArrayList<String> detection_model_names = new ArrayList<>();
    private ArrayList<StorageReference> cloud_trace_refs = new ArrayList<StorageReference>();
    private StorageReference cloud_trace_ref;

    private ConnectivityManager cm;
    boolean isConnected;
    boolean isMetered;

    TextView noticeTextView;


    @Override
    protected void onCreate(final Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        // Firebase: need to happen before mlmonitor is init
        mStorageRef = FirebaseStorage.getInstance().getReference();
        DeviceID = getDeviceID();

        cm = (ConnectivityManager) getSystemService(CONNECTIVITY_SERVICE);
        noticeTextView = (TextView) findViewById(R.id.MainActivityText);

        try {
            initModelList(this);
            loadTaskList(this);
        } catch (IOException e) {
            e.printStackTrace();
        }
        initDataList(this);
        loadDataSrcList(this);

        initConfigList(this);
    }

    @Override
    protected void onResume() {
        super.onResume();
        loadLocalDataList();
        loadCloudDataList();
        loadDataSrcList(this);
    }

    /// menu functions
    public void initModelList(Context context) throws IOException {
        loadDetectionModelList();
        loadClassificationModelList();
        model_choice = findViewById(R.id.model_spinner);
        ModelArrayAdapter = new ArrayAdapter<String>(context, android.R.layout.simple_spinner_dropdown_item);
        ModelArrayAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        model_choice.setAdapter(ModelArrayAdapter);
        model_choice.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                model_name = task_name + '/' + parent.getItemAtPosition(position).toString();
                Log.d(TAG, "Choosing Model: " + model_name);
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
                model_name = "";
                Log.d(TAG, "Model not selected");
            }
        });
    }

    public void loadTaskList(Context context) {

        ArrayList<String> task_names = new ArrayList<>();
        task_names.add("classification");
        task_names.add("detection");

        Spinner task_choice = findViewById(R.id.task_spinner);
        ArrayAdapter<String> TaskArrayAdapter = new ArrayAdapter<String>(context, android.R.layout.simple_spinner_dropdown_item, task_names);
        TaskArrayAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        task_choice.setAdapter(TaskArrayAdapter);

        task_choice.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                task_name = parent.getItemAtPosition(position).toString();
                if (task_name.equals("detection")) {
                    ModelArrayAdapter.clear();
                    ModelArrayAdapter.addAll(detection_model_names);
                }
                if (task_name.equals("classification")) {
                    ModelArrayAdapter.clear();
                    ModelArrayAdapter.addAll(classification_model_names);
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
                task_name = "";
            }
        });
    }

    public void loadDetectionModelList() throws IOException {
        detection_model_names.clear();
        detection_model_names.add("");
        String[] model_names = getAssets().list("detection");
        for (String model_name : model_names) {
            if (model_name.contains(".tflite")) {
                detection_model_names.add(model_name);
            }
        }
    }

    public void loadClassificationModelList() throws IOException {
        classification_model_names.clear();
        classification_model_names.add("");
        String[] model_names = getAssets().list("classification");
        for (String model_name : model_names) {
            if (model_name.contains(".tflite")) {
                classification_model_names.add(model_name);
            }
        }
    }

    public void loadDataSrcList(Context context) {
        ArrayList<String> data_src_names = new ArrayList<>();
        data_src_names.add("Local");
        data_src_names.add("Cloud");

        Spinner data_src_choice = findViewById(R.id.data_src_spinner);
        ArrayAdapter<String> DataSrcArrayAdapter = new ArrayAdapter<String>(context, android.R.layout.simple_spinner_dropdown_item, data_src_names);
        DataSrcArrayAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        data_src_choice.setAdapter(DataSrcArrayAdapter);

        data_src_choice.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                data_src_name = parent.getItemAtPosition(position).toString();
                if (data_src_name.equals("Local")) {
                    DataArrayAdapter.clear();
                    DataArrayAdapter.addAll(local_traces);
                }
                if (data_src_name.equals("Cloud")) {
                    DataArrayAdapter.clear();
                    DataArrayAdapter.addAll(cloud_traces);
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
                data_src_name = "";
            }
        });
    }

    public void initDataList(Context context) {
        loadLocalDataList();
        loadCloudDataList();
        data_choice = findViewById(R.id.data_spinner);
        DataArrayAdapter = new ArrayAdapter<String>(context, android.R.layout.simple_spinner_dropdown_item);
        DataArrayAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        data_choice.setAdapter(DataArrayAdapter);

        data_choice.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                data_name = parent.getItemAtPosition(position).toString();
                Log.d(TAG, "Selecting data trace " + data_name);
                if (data_src_name.equals("Cloud")) {
                    cloud_trace_ref = cloud_trace_refs.get(position);
                    Log.d(TAG, "cloud ref " + cloud_trace_ref.getName());
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
                data_name = "";
                Log.d(TAG, "Dataset not selected!");
            }
        });
    }

    public void loadCloudDataList() {
        // find all traces from cloud
        cloud_traces.clear();
        cloud_traces.add("");
        cloud_trace_refs.clear();
        cloud_trace_refs.add(null);
        ArrayList<String> users = new ArrayList<>();

        mStorageRef.listAll().addOnSuccessListener(new OnSuccessListener<ListResult>() {
            @Override
            public void onSuccess(ListResult listResult) {
                for (StorageReference user_prefix : listResult.getPrefixes()) {
                    // All the prefixes under listRef.
                    // You may call listAll() recursively on them.
                    String user = user_prefix.getName();
                    users.add(user);
                    Log.d(TAG, "Loading user " + user);
                    user_prefix.child("sdcard").child("edgeml").listAll().addOnSuccessListener(new OnSuccessListener<ListResult>() {
                        @Override
                        public void onSuccess(ListResult listResult) {
                            for (StorageReference trace_prefix : listResult.getPrefixes()) {
                                String trace = trace_prefix.getName();
                                cloud_traces.add(trace);
                                cloud_trace_refs.add(trace_prefix);
                                Log.d(TAG, "Loading trace " + trace);
                            }
                        }
                    });
                }
            }
        });
    }

    public void loadLocalDataList() {
        local_traces.clear();
        local_traces.add("");
        File rootFile = new File(rootfile);
        File[] subDirs = rootFile.listFiles();
        if (subDirs == null) return;
        for (File subDir : subDirs) {
            // check if deviceid text file
            if (subDir.isDirectory()) {
                String subDir_path = subDir.getPath();
                if (subDir_path.contains("playback")) {
                    continue;
                }
                local_traces.add(subDir_path);
            }
        }
    }

    public void initConfigList(Context context) {
        // accelerator configuration
        initAcceleratorChoiceList(context);
        initThreadingChoiceList(context);
        initLoggingChoiceList(context);
        initResizingFuncChoiceList(context);
        initChannelChoiceList(context);
        initScaleRnageChoiceList(context);
        initRotationChoiceList(context);

    }

    public void initAcceleratorChoiceList(Context context) {
        Spinner mSpinner = findViewById(R.id.aceelerator_spinner);
        ArrayAdapter<Utils.AcceleratorChoice> mArrayAdapter = new ArrayAdapter<Utils.AcceleratorChoice>(context, android.R.layout.simple_spinner_dropdown_item);
        mArrayAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);

        ArrayList<Utils.AcceleratorChoice> choice_list = new ArrayList<>();
        choice_list.add(Utils.AcceleratorChoice.CPU);
        choice_list.add(Utils.AcceleratorChoice.GPU);
        choice_list.add(Utils.AcceleratorChoice.NNAPI);
        mArrayAdapter.addAll(choice_list);

        mSpinner.setAdapter(mArrayAdapter);
        mSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                m_accelerator = parent.getItemAtPosition(position).toString();
                Log.d(TAG, "Choosing Accelerator: " + m_accelerator);
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
                m_accelerator = Utils.AcceleratorChoice.CPU.toString();
                Log.d(TAG, "Choosing Accelerator: " + m_accelerator);
            }
        });
    }

    public void initThreadingChoiceList(Context context) {
        Spinner mSpinner = findViewById(R.id.thread_spinner);
        ArrayAdapter<Integer> mArrayAdapter = new ArrayAdapter<Integer>(context, android.R.layout.simple_spinner_dropdown_item);
        mArrayAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);

        ArrayList<Integer> choice_list = new ArrayList<>();
        choice_list.add(1);
        choice_list.add(2);
        choice_list.add(4);
        choice_list.add(8);
        mArrayAdapter.addAll(choice_list);

        mSpinner.setAdapter(mArrayAdapter);
        mSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                m_num_threads = parent.getItemAtPosition(position).toString();
                Log.d(TAG, "Choosing Number of Threads: " + m_num_threads);
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
                m_num_threads = "1";
                Log.d(TAG, "Choosing Number of Threads: " + m_num_threads);
            }
        });
    }

    public void initLoggingChoiceList(Context context) {
        Spinner mSpinner = findViewById(R.id.logging_spinner);
        ArrayAdapter<Utils.LoggingChoice> mArrayAdapter = new ArrayAdapter<Utils.LoggingChoice>(context, android.R.layout.simple_spinner_dropdown_item);
        mArrayAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);

        ArrayList<Utils.LoggingChoice> choice_list = new ArrayList<>();
        choice_list.add(Utils.LoggingChoice.NONE);
        choice_list.add(Utils.LoggingChoice.IO);
        choice_list.add(Utils.LoggingChoice.EMBEDDING);
        choice_list.add(Utils.LoggingChoice.PERLAYER);
        mArrayAdapter.addAll(choice_list);

        mSpinner.setAdapter(mArrayAdapter);
        mSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                m_logging = parent.getItemAtPosition(position).toString();
                Log.d(TAG, "Choosing Logging Levels: " + m_logging);
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
                m_logging = Utils.LoggingChoice.NONE.toString();
                Log.d(TAG, "Choosing Logging Levels: " + m_logging);
            }
        });
    }

    public void initResizingFuncChoiceList(Context context) {
        Spinner mSpinner = findViewById(R.id.resize_func_spinner);
        ArrayAdapter<Utils.ResizingChoice> mArrayAdapter = new ArrayAdapter<Utils.ResizingChoice>(context, android.R.layout.simple_spinner_dropdown_item);
        mArrayAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);

        ArrayList<Utils.ResizingChoice> choice_list = new ArrayList<>();
        choice_list.add(Utils.ResizingChoice.AVG_AREA);
        choice_list.add(Utils.ResizingChoice.BILINEAR);
        mArrayAdapter.addAll(choice_list);

        mSpinner.setAdapter(mArrayAdapter);
        mSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                m_resizing_func = parent.getItemAtPosition(position).toString();
                Log.d(TAG, "Choosing Resizing Function: " + m_resizing_func);
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
                m_resizing_func = Utils.ResizingChoice.AVG_AREA.toString();
                Log.d(TAG, "Choosing Resizing Function: " + m_resizing_func);
            }
        });
    }

    public void initChannelChoiceList(Context context) {
        Spinner mSpinner = findViewById(R.id.channel_spinner);
        ArrayAdapter<Utils.ChannelChoice> mArrayAdapter = new ArrayAdapter<>(context, android.R.layout.simple_spinner_dropdown_item);
        mArrayAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);

        ArrayList<Utils.ChannelChoice> choice_list = new ArrayList<>();
        choice_list.add(Utils.ChannelChoice.RGB);
        choice_list.add(Utils.ChannelChoice.BGR);
        mArrayAdapter.addAll(choice_list);

        mSpinner.setAdapter(mArrayAdapter);
        mSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                m_channel = parent.getItemAtPosition(position).toString();
                Log.d(TAG, "Choosing Channel: " + m_channel);
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
                m_channel = Utils.ChannelChoice.RGB.toString();
                Log.d(TAG, "Choosing Channel: " + m_channel);
            }
        });
    }

    public void initScaleRnageChoiceList(Context context) {
        Spinner mSpinner = findViewById(R.id.scale_spinner);
        ArrayAdapter<Utils.ScaleRangeChoice> mArrayAdapter = new ArrayAdapter<>(context, android.R.layout.simple_spinner_dropdown_item);
        mArrayAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);

        ArrayList<Utils.ScaleRangeChoice> choice_list = new ArrayList<>();
        choice_list.add(Utils.ScaleRangeChoice.MINUSONE_ONE);
        choice_list.add(Utils.ScaleRangeChoice.ZERO_ONE);
        mArrayAdapter.addAll(choice_list);

        mSpinner.setAdapter(mArrayAdapter);
        mSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                m_scale_range = parent.getItemAtPosition(position).toString();
                Log.d(TAG, "Choosing Input Scale: " + m_scale_range);
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
                m_scale_range = Utils.ScaleRangeChoice.MINUSONE_ONE.toString();
                Log.d(TAG, "Choosing Input Scale: " + m_scale_range);
            }
        });
    }

    public void initRotationChoiceList(Context context) {
        Spinner mSpinner = findViewById(R.id.rotation_spinner);
        ArrayAdapter<Integer> mArrayAdapter = new ArrayAdapter<>(context, android.R.layout.simple_spinner_dropdown_item);
        mArrayAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);

        ArrayList<Integer> choice_list = new ArrayList<>();
        choice_list.add(0);
        choice_list.add(90);
        choice_list.add(180);
        choice_list.add(270);
        mArrayAdapter.addAll(choice_list);

        mSpinner.setAdapter(mArrayAdapter);
        mSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                m_rotation = parent.getItemAtPosition(position).toString();
                Log.d(TAG, "Choosing Rotation: " + m_rotation);
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
                m_rotation = "0";
                Log.d(TAG, "Choosing Rotation: " + m_rotation);
            }
        });
    }

    public void log_configs(){
        Log.d(TAG, "Configuration in Java UI:");
        Log.d(TAG, "Model Name: " + model_name);
        Log.d(TAG, "Dataset: " + data_name);
        Log.d(TAG, "Task: " + task_name);
        Log.d(TAG, "Accelerator: " + m_accelerator);
        Log.d(TAG, "Num Threads: " + m_num_threads);
        Log.d(TAG, "Logging Level: " + m_logging);
        Log.d(TAG, "Resising Function: " + m_resizing_func);
        Log.d(TAG, "Input Scale: " + m_scale_range);
        Log.d(TAG, "Channel: " + m_channel);
        Log.d(TAG, "Rotation: " + m_rotation);
    }

    /// button functions
    public void onClickStart(View view) {
        if (task_name.equals("")) {
            return;
        }
        log_configs();
        Intent intent;
        switch (task_name){
            case "detection":
                intent = new Intent(this, DetectorActivity.class);
                break;
            case "classification":
                intent = new Intent(this, ClassifierActivity.class);
                break;
            default:
                throw new IllegalStateException("Unexpected task name value: " + task_name);
        }

        intent.putExtra(Utils.ExtraIntentVar.EXTRA_MODEL_NAME.toString(), model_name);
        intent.putExtra(Utils.ExtraIntentVar.EXTRA_DATA_NAME.toString(), "");

        intent.putExtra(Utils.ExtraIntentVar.EXTRA_ACCELERATOR_CHOICE.toString(), m_accelerator);
        intent.putExtra(Utils.ExtraIntentVar.EXTRA_NUM_THREAD.toString(), m_num_threads);
        intent.putExtra(Utils.ExtraIntentVar.EXTRA_LOGGING_CHOICE.toString(), m_logging);
        intent.putExtra(Utils.ExtraIntentVar.EXTRA_RESIZING_FUNC.toString(), m_resizing_func);
        intent.putExtra(Utils.ExtraIntentVar.EXTRA_SCALE_RANGE.toString(), m_scale_range);
        intent.putExtra(Utils.ExtraIntentVar.EXTRA_CHANNEL_CHOICE.toString(), m_channel);
        intent.putExtra(Utils.ExtraIntentVar.EXTRA_ROTATION.toString(), m_rotation);
        startActivity(intent);

    }

    public void onClickPlayback(View view) {
        if (data_name.equals("")) {
            return;
        }
        log_configs();
        if (data_src_name.equals("Cloud")) {
            if (!checkPathExist(rootfile + data_name)) {
                data_name = "/sdcard/edgeml/" + data_name + "_Download";
            }
        }
        if (!checkPathExist(data_name)) {
            Log.d(TAG, "Datafolder " + data_name + " doesn't exist!");
            return;
        }

        Intent intent = new Intent(this, PlaybackActivity.class);
        intent.putExtra(Utils.ExtraIntentVar.EXTRA_TASK_NAME.toString(), task_name);
        intent.putExtra(Utils.ExtraIntentVar.EXTRA_MODEL_NAME.toString(), model_name);
        intent.putExtra(Utils.ExtraIntentVar.EXTRA_DATA_NAME.toString(), data_name);

        intent.putExtra(Utils.ExtraIntentVar.EXTRA_ACCELERATOR_CHOICE.toString(), m_accelerator);
        intent.putExtra(Utils.ExtraIntentVar.EXTRA_NUM_THREAD.toString(), m_num_threads);
        intent.putExtra(Utils.ExtraIntentVar.EXTRA_LOGGING_CHOICE.toString(), m_logging);
        intent.putExtra(Utils.ExtraIntentVar.EXTRA_RESIZING_FUNC.toString(), m_resizing_func);
        intent.putExtra(Utils.ExtraIntentVar.EXTRA_SCALE_RANGE.toString(), m_scale_range);
        intent.putExtra(Utils.ExtraIntentVar.EXTRA_CHANNEL_CHOICE.toString(), m_channel);
        intent.putExtra(Utils.ExtraIntentVar.EXTRA_ROTATION.toString(), m_rotation);

        startActivity(intent);
    }

    public void onClickUpload(View view) {

        checkNetwork();
        notice = "Checking network...\n";
        noticeTextView.setText(notice);
        if (isConnected) {
            Log.d(TAG, "Network Connectivity: OK");
            notice = notice + "Network Connectivity: OK\n";
            noticeTextView.setText(notice);
            if (!isMetered) {
                Log.d(TAG, "None-Metered Connectivity:  uploading logs...");
                notice = notice + "None-Metered Connectivity:  uploading logs\n";
                noticeTextView.setText(notice);
                final String curNotice = notice;
                UploadLogs(DeviceID, curNotice);
            } else {
                notice = notice + "Metered Connectivity: not uploading logs for now\n";
                noticeTextView.setText(notice);
                Log.d(TAG, "Metered Connectivity: not uploading logs");
            }
        } else {
            notice = notice + "No Network Connectivity, not uploading logs for now\n";
            noticeTextView.setText(notice);
            Log.d(TAG, "No Network Connectivity");
        }
    }

    public void onClickDownload(View view) {
        checkNetwork();
        notice = "Checking network...\n";
        noticeTextView.setText(notice);
        if (isConnected) {
            Log.d(TAG, "Network Connectivity: OK");
            notice = notice + "Network Connectivity: OK\n";
            noticeTextView.setText(notice);
            if (!isMetered) {
                Log.d(TAG, "None-Metered Connectivity:  Downloading logs...");
                notice = notice + "None-Metered Connectivity:  Downloading logs\n";
                noticeTextView.setText(notice);
                final String curNotice = notice;
                DownloadCloudTrace(curNotice);
            } else {
                notice = notice + "Metered Connectivity: not Downloading logs for now\n";
                noticeTextView.setText(notice);
                Log.d(TAG, "Metered Connectivity: not Downloading logs");
            }
        } else {
            notice = notice + "No Network Connectivity, not Downloading logs for now\n";
            noticeTextView.setText(notice);
            Log.d(TAG, "No Network Connectivity");
        }
    }

    public void onClickClear(View view) {
        noticeTextView.setText("Removing all local logs...");
        File rootFile = new File(rootfile);
        File[] subDirs = rootFile.listFiles();
        if (subDirs == null) return;
        for (File subDir : subDirs) {
            // check if deviceid text file
            if (subDir.getAbsolutePath().equals(DeviceIDPath)) {
                continue;
            }
            // filter benchmark dataset paths
            if (DatasetPathArray.contains(subDir.getAbsolutePath())) {
                continue;
            }
            // delete this folder
            noticeTextView.setText("Deleting " + subDir.getAbsolutePath());
            deleteDirectory(subDir);
        }
        loadLocalDataList();
        loadDataSrcList(this);
        noticeTextView.setText("Local logs removed!");
    }

    // TODO: move all these irrelavant code below to a separate library
    private String getDeviceID() {
        File IDFile = new File(DeviceIDPath);
        if (IDFile.exists()) {
            try (BufferedReader br = new BufferedReader(new FileReader(IDFile))) {
                uniqueID = br.readLine();
            } catch (Exception e) {
                e.printStackTrace();
            }
        } else {
            try (BufferedWriter bw = new BufferedWriter(new FileWriter(IDFile))) {
                bw.write(uniqueID);
                bw.newLine();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        return uniqueID;
    }

    public void checkNetwork() {
        // check network

        NetworkInfo activeNetwork = cm.getActiveNetworkInfo();
        isConnected = activeNetwork != null &&
                activeNetwork.isConnectedOrConnecting();
        isMetered = cm.isActiveNetworkMetered();
    }

    private void uploadOneFile(String DeviceID, String filePath, String CurNotice) {

        final StorageReference uploadRef = mStorageRef.child(DeviceID).child(filePath);

//        Log.d(TAG, "uploadFromUri:src:" + filePath);
//        Log.d(TAG, "uploadFromUri:dst:" + uploadRef.getPath());
        InputStream stream;
        try {
            stream = new FileInputStream(filePath);
            UploadTask uploadTask = uploadRef.putStream(stream);
            // sync upload
            uploadTask.addOnCompleteListener(new OnCompleteListener<UploadTask.TaskSnapshot>() {
                @Override
                public void onComplete(Task<UploadTask.TaskSnapshot> task) {
                    notice = CurNotice + "Uploaded: " + filePath;
                    noticeTextView.setText(notice);
                }
            });
        } catch (Exception e) {
            e.printStackTrace();
        }
//        uploadTask.addOnFailureListener(new OnFailureListener() {
//            @Override
//            public void onFailure(@NonNull Exception exception) {
//                // Handle unsuccessful uploads
//            }
//        }).addOnSuccessListener(new OnSuccessListener<UploadTask.TaskSnapshot>() {
//            @Override
//            public void onSuccess(UploadTask.TaskSnapshot taskSnapshot) {
//                // taskSnapshot.getMetadata() contains file metadata such as size, content-type, etc.
//                // ...
//            }
//        });
    }

    public boolean deleteDirectory(File path) {
        Log.d(TAG, "Deleting " + path.getAbsolutePath());
        if (!path.exists()) {
            return true;
        }
        if (path.isDirectory()) {
            File[] files = path.listFiles();
            if (files != null) {
                for (int i = 0; i < files.length; i++) {
                    if (!deleteDirectory(files[i])) {
                        return false;
                    }
                }
            }
        }
        return path.delete();
    }

    private void UploadLogs(String DeviceID, String CurNotice) {

        File rootFile = new File(rootfile);
        File[] subDirs = rootFile.listFiles();
        if (subDirs == null) return;
        for (File subDir : subDirs) {
            // check if deviceid text file
            if (subDir.getAbsolutePath().equals(DeviceIDPath)) {
                continue;
            }
            if (subDir.getAbsolutePath().contains("summary.log")) {
                uploadOneFile(DeviceID, subDir.getAbsolutePath(), CurNotice);
                continue;
            }
            // check if already uploaded
            String uploadFlagFilePath = subDir.getAbsolutePath() + "/UPLOADED";
            File uploadFlag = new File(uploadFlagFilePath);
            if (uploadFlag.exists()) {
                // already uploaded
                // delete this folder
                deleteDirectory(subDir);
                continue;
            }
            // upload the folder
            File[] subFiles = subDir.listFiles();
            if (subFiles != null) {
                for (File subFilePath : subFiles) {
                    File[] logs = subFilePath.listFiles();
                    if (logs == null) continue;
                    for (File log : logs) {
                        uploadOneFile(DeviceID, log.getAbsolutePath(), CurNotice);
                    }
                }
            }
            // leave an uploaded flag
            try {
                uploadFlag.createNewFile();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    public void DownloadCloudTrace(String CurNotice) {
        if (data_src_name.equals("Local") || cloud_trace_ref == null) {
            Log.d(TAG, "illegal cloud ref ");
            return;
        }
        // search if exist first
        String FolderName = rootfile + data_name;
        File FolderFile = new File(FolderName);
        if (FolderFile.exists()) {
            // It's my data, no need to download, assuming each trace has unique start time
            Log.d(TAG, "Data is local already");
            return;
        }

        // download it
        String localDir = rootfile + data_name + "_Download/nativeInput/";
        File localFile = new File(localDir);
        if (localFile.exists()) {
            // already downloaded
            Log.d(TAG, "Data is already downloaded");
            return;
        } else {
            localFile.mkdirs();
        }

        // download data from firebase to local
        cloud_trace_ref.child("nativeInput").listAll().addOnSuccessListener(new OnSuccessListener<ListResult>() {
            @Override
            public void onSuccess(ListResult listResult) {
                for (StorageReference item : listResult.getItems()) {
                    // All the items under listRef.
                    String localfilename = localDir + item.getName();
                    File local_cp = new File(localfilename);
                    item.getFile(local_cp).addOnSuccessListener(new OnSuccessListener<FileDownloadTask.TaskSnapshot>() {
                        @Override
                        public void onSuccess(FileDownloadTask.TaskSnapshot taskSnapshot) {
                            Log.d(TAG, "Downloaded " + localfilename);
                            notice = CurNotice + "Downloaded: " + localfilename;
                            noticeTextView.setText(notice);
                        }
                    }).addOnFailureListener(new OnFailureListener() {
                        @Override
                        public void onFailure(Exception e) {
                            e.printStackTrace();
                        }
                    });
                }
            }
        });
    }

    public boolean checkPathExist(String path) {
        File FolderFile = new File(path);
        return FolderFile.exists();
    }
}
