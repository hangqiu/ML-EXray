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
    name = "libimageclassifier.so",
    srcs = glob([
        "**/*.cpp",
        "**/*.h",
    ]),
    deps = [
        "//edgeml/cpp/tflite/image_classification:tflite_image_classifier_lib",
        "@org_tensorflow//tensorflow/lite/java/jni",
    ],
)]

[cc_library(
    name = "image_classifier_native",
    srcs = ["libimageclassifier.so"],
    visibility = ["//visibility:public"],
)]
