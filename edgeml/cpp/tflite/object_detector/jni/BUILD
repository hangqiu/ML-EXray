load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_android//android:rules.bzl", "android_binary", "android_library")
load("@org_tensorflow//tensorflow/lite:build_def.bzl", "tflite_jni_binary")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

[tflite_jni_binary(
    name = "libobjectdetector.so",
    srcs = glob([
        "**/*.cpp",
        "**/*.h",
    ]),
    deps = [
        "//edgeml/cpp/tflite/object_detector:tflite_object_detector_lib",
        "@org_tensorflow//tensorflow/lite/java/jni",
    ],
)]

[cc_library(
    name = "object_detector_native",
    srcs = ["libobjectdetector.so"],
    visibility = ["//visibility:public"],
)]

#cc_library(
#    name = "object_detector_native",
#    srcs = ["object_detector_jni.cpp"],
#    deps = ["//edgeml/cpp/tflite/object_detector:tflite_object_detector_lib",],
#)
