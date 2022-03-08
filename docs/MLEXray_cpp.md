
### Using ML-EXray C++ API
For a full example, please see android app in the examples folder.

The following example will produce the EXray logs including the inference results,
inference latency, per-layer output, per-layer latency.

For inertial sensors during image capture, please refer to [Java APIs](docs/MLEXray_java.md).

* Create an instance of EdgeMLMonitor
```
mlMonitorNative = new EdgeMLMonitor_Native(nativeLogDir, playback_,no_logging_,input_logging_,output_logging_,embedding_logging_,per_layer_logging_);
``` 
* Initialize with TFLite interpreter
```
mlMonitorNative->onTFliteInitStart(modelSize);
// initiazlize TFLite interpreter
mlMonitorNative->onTFliteInitStop(&m_interpreter);
```

* Logging metadata around ML inference
```
ClassificationResult *ImageClassifier::classify(Mat src) {
    mlMonitorNative->startCurrentFrame();
    ...
    mlMonitorNative->onTFliteInferenceStart(image);
    TfLiteStatus status = m_interpreter->Invoke();
    mlMonitorNative->onTFliteInferenceStop(&m_interpreter, m_input_tensor, m_last_tensor,
                                           m_output_scores, m_modelQuantized);
    ...
    mlMonitorNative->endCurrentFrame();
}                                       
```

* Logging customized Ops
```
    mlMonitorNative->onRandomOpsStart("Resizing");
    Mat image;
    if (m_resizing_func_choice == "AVG_AREA") {
        resize(src, image, Size(DETECTION_MODEL_SIZE, DETECTION_MODEL_SIZE), 0, 0, INTER_AREA);
    } else if (m_resizing_func_choice == "BILINEAR") {

        resize(src, image, Size(DETECTION_MODEL_SIZE, DETECTION_MODEL_SIZE), 0, 0, INTER_LINEAR);
    } else {
        __android_log_print(ANDROID_LOG_ERROR, TAG, "Resizing function not supported");
    }
    mlMonitorNative->onRandomOpsStop("Resizing");
```

