
### Using ML-EXray Java API
For a full example, please see android app in the examples folder.

The following example will produce the EXray logs including the inference results,
inference latency, inertial sensors during image capture.

For per-layer output and latency, please refer to [C++](docs/MLEXray_cpp.md) or [Python](edgeml/python/README.md) APIs.

* Create an instance of EdgeMLMonitor
```
public EdgeMLMonitor mlMonitor;
``` 
* Pass the context to constructor function
```
protected void onCreate(...){
    ...
    mlMonitor = new EdgeMLMonitor(this);
    ...
}
```
* Logging sensor around image capture

```
/** Callback for Camera2 API */
public void onImageAvailable(final ImageReader reader) {
    mlMonitor.onSensorStop();
    ... process image
    mlMonitor.onSensorStart();
}
/** Callback for android.hardware.Camera API */
public void onPreviewFrame(final byte[] bytes, final Camera camera) {
    mlMonitor.onSensorStop();
    ... process image
    mlMonitor.onSensorStart();
}
```
* Logging metadata around ML inference
```
mlMonitor.onInferenceStart();
final List<Classifier.Recognition> results =
  classifier.recognizeImage(rgbFrameBitmap, sensorOrientation);
mlMonitor.onInferenceStop();
```

