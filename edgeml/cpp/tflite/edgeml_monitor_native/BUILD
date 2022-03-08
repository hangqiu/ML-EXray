load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_android//android:rules.bzl", "android_binary", "android_library")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_cc_binary")
load("@org_tensorflow//tensorflow/lite:special_rules.bzl", "tflite_portable_test_suite")
load("@org_tensorflow//tensorflow/lite:build_def.bzl", "tflite_copts", "tflite_copts_warnings", "tflite_linkopts")

common_copts = tflite_copts() + tflite_copts_warnings()

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "edgeml_monitor_native_lib",
    srcs = ["edgeml_monitor_native.cpp"],
    hdrs = ["edgeml_monitor_native.h"],
    copts = common_copts,
    deps = [
        #        "//edgeml/cpp/tfbenchmark:benchmark_model_lib",
        "//third_party:opencv_binary",
        "//edgeml/cpp/tfbenchmark:benchmark_tflite_model_lib",
        "@org_tensorflow//tensorflow/lite/profiling:profiler",
        #        "@org_tensorflow//tensorflow/lite:framework",
        #        "@org_tensorflow//tensorflow/lite:string_util",
        #        "@org_tensorflow//tensorflow/lite/c:common",
        #        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
        #        "@org_tensorflow//tensorflow/lite/kernels:cpu_backend_context",
    ],
)
