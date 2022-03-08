#include <android/log.h>
#include "ObjectDetector.h"
#include <opencv2/imgproc.hpp>
//#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/register_ref.h"
#include "tensorflow/lite/model.h"

using namespace cv;

ObjectDetector::ObjectDetector(const char *tfliteModel,
							   string nativeLogDir,
							   long modelSize,
							   bool playback_,
							   bool quantized,
							   int model_input_size,
							   string acceleartor_choice,
							   int numThreads,
							   string logging_choice,
							   string resizing_func_choice,
							   string scale_range_choice,
							   string channel_choice
							   ) {
	m_modelQuantized = quantized;
	m_numThreads = numThreads;
	DETECTION_MODEL_SIZE = model_input_size;
	m_resizing_func_choice = resizing_func_choice;
	m_channel_choice = channel_choice;
	m_input_scale_range = scale_range_choice;
	//////////////////////////////////////////
	if (acceleartor_choice == "GPU") {
		use_GPUDelegate = true;
	}
	//////////////////////////////////////////
	if (logging_choice=="NONE"){
		no_logging_ = true;
	}
	if (logging_choice == "IO") {
//		input_logging_ = true;
		output_logging_ = true;
	}
	if (logging_choice == "EMBEDDING") {
		input_logging_ = true;
		output_logging_ = true;
		embedding_logging_ = true;
	}
	if (logging_choice == "PERLAYER") {
		input_logging_ = true;
		output_logging_ = true;
		embedding_logging_ = true;
		per_layer_logging_ = true;
	}
	mlMonitorNative = new EdgeMLMonitor_Native(nativeLogDir, playback_,
											   no_logging_,
											   input_logging_,
											   output_logging_,
											   embedding_logging_,
											   per_layer_logging_);
	//////////////////////////////////////////
	if (modelSize > 0) {
		mlMonitorNative->onTFliteInitStart(modelSize);
		initDetectionModel(tfliteModel, modelSize);
		mlMonitorNative->initTFLiteBenchmarkModel(&m_interpreter);
		mlMonitorNative->onTFliteInitStop(&m_interpreter);
	}
}

ObjectDetector::~ObjectDetector() {
	if (per_layer_logging_){
		mlMonitorNative->endTFLiteBenchmarkModel();
	}
	free(mlMonitorNative);
	if (m_modelBytes != nullptr) {
		free(m_modelBytes);
		m_modelBytes = nullptr;
	}
	m_hasDetectionModel = false;
}

// Credit: https://github.com/YijinLiu/tf-cpu/blob/master/benchmark/obj_detect_lite.cc
void ObjectDetector::initDetectionModel(const char *tfliteModel, long modelSize) {
	if (modelSize < 1) { return; }

	// Copy to model bytes as the caller might release this memory while we need it (EXC_BAD_ACCESS error on ios)
	m_modelBytes = (char *) malloc(sizeof(char) * modelSize);
	memcpy(m_modelBytes, tfliteModel, sizeof(char) * modelSize);
	m_model = tflite::FlatBufferModel::BuildFromBuffer(m_modelBytes, modelSize);

	if (m_model == nullptr) {
		__android_log_print(ANDROID_LOG_ERROR, TAG,"Failed to load model");
		return;
	}

	// Build the interpreter
	tflite::ops::builtin::BuiltinOpResolver resolver;
//	tflite::ops::builtin::BuiltinRefOpResolver resolver;
	tflite::InterpreterBuilder builder(*m_model, resolver);
	if (per_layer_logging_){
		builder.PreserveAllTensorsExperimental();
	}
	builder(&m_interpreter);
	if (m_interpreter == nullptr) {
		__android_log_print(ANDROID_LOG_ERROR, TAG, "Failed to create interpreter");
		return;
	}

	// Allocate tensor buffers.
	if (m_interpreter->AllocateTensors() != kTfLiteOk) {
		__android_log_print(ANDROID_LOG_ERROR, TAG, "Failed to allocate tensors!");
		return;
	}

	m_interpreter->SetNumThreads(m_numThreads);

	// Find input tensors.
	if (m_interpreter->inputs().size() != 1) {
		__android_log_print(ANDROID_LOG_ERROR, TAG, "Detection model graph needs to have 1 and only 1 input!");
		return;
	}

	m_input_tensor = m_interpreter->tensor(m_interpreter->inputs()[0]);
	// ignoring input sanity check for detector model
//	if (m_modelQuantized && m_input_tensor->type != kTfLiteUInt8) {
//		__android_log_print(ANDROID_LOG_ERROR, TAG, "Detection model input should be kTfLiteUInt8!");
//		return;
//	}
//
//	if (!m_modelQuantized && m_input_tensor->type != kTfLiteFloat32) {
//		__android_log_print(ANDROID_LOG_ERROR, TAG, "Detection model input should be kTfLiteFloat32!");
//		return;
//	}

	if (m_input_tensor->dims->data[0] != 1 ||
		m_input_tensor->dims->data[1] != DETECTION_MODEL_SIZE ||
		m_input_tensor->dims->data[2] != DETECTION_MODEL_SIZE ||
		m_input_tensor->dims->data[3] != DETECTION_MODEL_CNLS) {
		__android_log_print(ANDROID_LOG_ERROR, TAG, "Detection model must have input dims of 1x%ix%ix%i", DETECTION_MODEL_SIZE,
			   DETECTION_MODEL_SIZE, DETECTION_MODEL_CNLS);
		return;
	}

	// Find output tensors.
	if (m_interpreter->outputs().size() != 4) {
		__android_log_print(ANDROID_LOG_ERROR, TAG, "Detection model graph needs to have 4 and only 4 outputs!");
		__android_log_print(ANDROID_LOG_ERROR, TAG, std::to_string(m_interpreter->outputs().size()).c_str());
		return;
	}

	m_output_locations = m_interpreter->tensor(m_interpreter->outputs()[0]);
	m_output_classes = m_interpreter->tensor(m_interpreter->outputs()[1]);
	m_output_scores = m_interpreter->tensor(m_interpreter->outputs()[2]);
	m_num_detections = m_interpreter->tensor(m_interpreter->outputs()[3]);


	// add intermediate layer outputs
	// visualize all tensors
	string embedding_layer_name = "FeatureExtractor/MobilenetV1/Conv2d_13_pointwise_2_Conv2d_5_3x3_s2_96/Relu6";
	int tensor_size = m_interpreter->tensors_size();
	__android_log_print(ANDROID_LOG_ERROR, TAG, "Model tensor size %d",tensor_size);
	for (int j=0;j<tensor_size;j++){
		// last tensor is null tensor
        if (!m_interpreter->tensor(j)->name)continue;
		__android_log_print(ANDROID_LOG_ERROR, TAG, "%d:%s",j, m_interpreter->tensor(j)->name);
		if (m_interpreter->tensor(j)->name == embedding_layer_name){
			m_last_tensor = m_interpreter->tensor(j);
			TfLiteIntArray *last_tensor_dim = m_last_tensor->dims;
			for (int i=0;i<last_tensor_dim->size;i++){
				__android_log_print(ANDROID_LOG_ERROR, TAG, "%d",last_tensor_dim->data[i]);
			}
		}
	}
	m_hasDetectionModel = true;
	__android_log_print(ANDROID_LOG_ERROR, TAG, "Finished tflite Initialization");
}

DetectResult *ObjectDetector::detect(Mat src) {
    mlMonitorNative->startCurrentFrame();

	DetectResult res[DETECT_NUM];
	if (!m_hasDetectionModel) {
		return res;
	}

	mlMonitorNative->onRandomOpsStart("PreprocessingTotal");
    mlMonitorNative->onRandomOpsStart("Resizing");
    Mat image;
    resize(src, image, Size(DETECTION_MODEL_SIZE, DETECTION_MODEL_SIZE), 0, 0, INTER_AREA);
    mlMonitorNative->onRandomOpsStop("Resizing");

    mlMonitorNative->onRandomOpsStart("MemCopy");
	int cnls = image.type();
	if (cnls == CV_8UC1) {
		cvtColor(image, image, COLOR_GRAY2RGB);
	} else if (cnls == CV_8UC4) {
		cvtColor(image, image, COLOR_BGRA2RGB);
	}

//	if (m_modelQuantized) {
    // Copy image into input tensor
    // Ignoring quantization flag for now, cuz supported detector ssd model use uint8 input
    uchar *dst = m_input_tensor->data.uint8;
    memcpy(dst, image.data,
           sizeof(uchar) * DETECTION_MODEL_SIZE * DETECTION_MODEL_SIZE * DETECTION_MODEL_CNLS);
//	} else {
//		// Normalize the image based on std and mean (p' = (p-mean)/std)
//		// this is scaling to [-1, 1]
//		Mat fimage;
//		image.convertTo(fimage, CV_32FC3, 1 / IMAGE_STD, -IMAGE_MEAN / IMAGE_STD);
//		image = fimage;
//
//		// Copy image into input tensor
//		float *dst = m_input_tensor->data.f;
//		memcpy(dst, fimage.data,
//			   sizeof(float) * DETECTION_MODEL_SIZE * DETECTION_MODEL_SIZE * DETECTION_MODEL_CNLS);
//	}
	mlMonitorNative->onRandomOpsStop("MemCopy");
	mlMonitorNative->onRandomOpsStop("PreprocessingTotal");

//	mlMonitorNative->onTFliteInferenceStart();
	mlMonitorNative->onTFliteInferenceStart(image);
	TfLiteStatus status = m_interpreter->Invoke();
	mlMonitorNative->onTFliteInferenceStop(&m_interpreter, m_input_tensor, m_last_tensor, nullptr, m_modelQuantized);
	if (status != kTfLiteOk) {
		__android_log_print(ANDROID_LOG_ERROR, TAG, "Error invoking detection model");
		return res;
	}

	const float *detection_locations = m_output_locations->data.f;
	const float *detection_classes = m_output_classes->data.f;
	const float *detection_scores = m_output_scores->data.f;
	const int num_detections = (int) *m_num_detections->data.f;

	for (int i = 0; i < num_detections && i < DETECT_NUM; ++i) {
		res[i].score = detection_scores[i];
		res[i].label = (int) detection_classes[i];

		// Get the bbox, make sure its not out of the image bounds, and scale up to src image size
		res[i].ymin = std::fmax(0.0f, detection_locations[4 * i] * src.rows);
		res[i].xmin = std::fmax(0.0f, detection_locations[4 * i + 1] * src.cols);
		res[i].ymax = std::fmin(float(src.rows - 1), detection_locations[4 * i + 2] * src.rows);
		res[i].xmax = std::fmin(float(src.cols - 1), detection_locations[4 * i + 3] * src.cols);
	}

	mlMonitorNative->endCurrentFrame();
	return res;
}
