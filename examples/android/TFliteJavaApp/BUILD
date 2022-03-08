load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_android//android:rules.bzl", "android_binary", "android_library")
load("@org_tensorflow//tensorflow/lite:build_def.bzl", "tflite_jni_binary")

android_binary(
    name = "edgeml_insight_java",
    srcs = glob([
        "**/*.java",
    ]),
    assets = glob(["assets/**/*"]),
    assets_dir = "assets/",
    custom_package = "com.edgeml.insight",
    manifest = "AndroidManifest.xml",
    nocompress_extensions = ["tflite"],
    resource_files = glob(["res/**/*"]),
    deps = [
        "//edgeml/java/com/edgeml/monitor/env",
        "//edgeml/java/com/edgeml/monitor/framework",
        "//edgeml/java/com/edgeml/monitor/hwinfo",
        "//edgeml/java/com/edgeml/monitor/monitor:edgemlmonitor",
        "@maven//:androidx_annotation_annotation",
        "@maven//:androidx_appcompat_appcompat",
        "@maven//:androidx_constraintlayout_constraintlayout",
        "@maven//:com_google_android_material_material",
        "@maven//:org_tensorflow_tensorflow_lite",
        "@maven//:org_tensorflow_tensorflow_lite_gpu",
        "@maven//:org_tensorflow_tensorflow_lite_support",
        #        "@maven//:io_opentelemetry_opentelemetry_sdk",
        #        "@maven//:io_opentelemetry_opentelemetry_sdk_common",
        #        "@maven//:io_opentelemetry_opentelemetry_sdk_trace",
        #        "@maven//:io_opentelemetry_opentelemetry_exporter_otlp",
        #        "@maven//:io_opentelemetry_opentelemetry_exporter_otlp_trace",
        #        "@maven//:io_opentelemetry_opentelemetry_context",
        #        "@maven//:io_opentelemetry_opentelemetry_api",
    ],
)
