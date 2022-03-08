load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_android//android:rules.bzl", "android_binary", "android_library")
load("@org_tensorflow//tensorflow/lite:build_def.bzl", "tflite_jni_binary")

[tflite_jni_binary(
    name = "libtensorflowlite_benchmark.so",
    srcs = glob([
        "**/*.cc",
        "**/*.h",
    ]),
    deps = [
        "//edgeml/cpp/tfbenchmark:benchmark_tflite_model_lib",
        "@org_tensorflow//tensorflow/lite/java/jni",
    ],
)]

[cc_library(
    name = "tensorflowlite_benchmark_native",
    srcs = ["libtensorflowlite_benchmark.so"],
    visibility = ["//visibility:public"],
)]
