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
    name = "benchmark_tflite_model_lib",
    srcs = ["benchmark_tflite_model.cc"],
    hdrs = ["benchmark_tflite_model.h"],
    copts = common_copts + select({
        "@org_tensorflow//tensorflow:ios": [
            "-xobjective-c++",
        ],
        "//conditions:default": [],
    }),
    deps = [
        ":benchmark_model_lib",
        ":benchmark_utils",
        ":profiling_listener",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite:string_util",
        "@org_tensorflow//tensorflow/lite/c:common",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
        "@org_tensorflow//tensorflow/lite/kernels:cpu_backend_context",
        "@org_tensorflow//tensorflow/lite/profiling:profile_summary_formatter",
        "@org_tensorflow//tensorflow/lite/profiling:profiler",
        "@org_tensorflow//tensorflow/lite/tools:logging",
        "@org_tensorflow//tensorflow/lite/tools/delegates:delegate_provider_hdr",
        "@org_tensorflow//tensorflow/lite/tools/delegates:tflite_execution_providers",
        "@ruy//ruy/profiler",
    ],
)

cc_library(
    name = "benchmark_model_lib",
    srcs = [
        "benchmark_model.cc",
    ],
    hdrs = ["benchmark_model.h"],
    copts = common_copts,
    deps = [
        ":benchmark_params",
        ":benchmark_utils",
        ":stats_calculator_portable",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/c:common",
        "@org_tensorflow//tensorflow/lite/profiling:memory_info",
        "@org_tensorflow//tensorflow/lite/profiling:time",
        "@org_tensorflow//tensorflow/lite/tools:command_line_flags",
        "@org_tensorflow//tensorflow/lite/tools:logging",
    ],
)

cc_library(
    name = "profiling_listener",
    srcs = ["profiling_listener.cc"],
    hdrs = ["profiling_listener.h"],
    copts = common_copts,
    deps = [
        ":benchmark_model_lib",
        "@org_tensorflow//tensorflow/lite/profiling:profile_summarizer",
        "@org_tensorflow//tensorflow/lite/profiling:profile_summary_formatter",
        "@org_tensorflow//tensorflow/lite/profiling:profiler",
        "@org_tensorflow//tensorflow/lite/tools:logging",
    ],
)

cc_library(
    name = "benchmark_utils",
    srcs = [
        "benchmark_utils.cc",
    ],
    hdrs = ["benchmark_utils.h"],
    copts = common_copts,
    deps = ["@org_tensorflow//tensorflow/lite/profiling:time"],
)

cc_library(
    name = "benchmark_params",
    hdrs = ["benchmark_params.h"],
    copts = common_copts,
    deps = ["@org_tensorflow//tensorflow/lite/tools:tool_params"],
)

cc_library(
    name = "stats_calculator_portable",
    srcs = [
        "stat_summarizer_options.h",
        "stats_calculator.cc",
    ],
    hdrs = [
        "stats_calculator.h",
    ],
    copts = common_copts,
)
