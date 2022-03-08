load("@rules_android//android:rules.bzl", "android_binary", "android_library")
#load("@tools_android//tools/googleservices:defs.bzl", "google_services_xml")
#
#GOOGLE_SERVICES_XML = google_services_xml(
#    package_name = "com.edgeml.monitor.monitor",
#    google_services_json = "google-services.json",
#)

android_binary(
    name = "edgemlinsight",
    srcs = glob([
        "**/*.java",
    ]),
    assets = glob(["assets/**/*"]),
    assets_dir = "assets/",
    custom_package = "com.edgeml.edgemlinsight",
    manifest = "AndroidManifest.xml",
    manifest_values = {
        "applicationId": "com.edgeml.edgemlinsight",
    },
    nocompress_extensions = ["tflite"],
    resource_files = glob(["res/**/*"]),
    deps = [
        "//edgeml/cpp/tflite/object_detector/jni:object_detector_native",
        "//edgeml/cpp/tflite/image_classification/jni:image_classifier_native",
        "//edgeml/java/com/edgeml/monitor/monitor:edgemlmonitor",
        "//third_party:opencv",
        "@maven//:androidx_appcompat_appcompat",
        "@maven//:androidx_constraintlayout_constraintlayout",
        "//third_party:androidx_constraint_layout",
        "@maven//:androidx_core_core_ktx",
        "@maven//:com_google_auto_value_auto_value_annotations",
        #        "@maven//:com_google_firebase_firebase_auth",
        "@maven//:com_google_firebase_firebase_storage",
        "@maven//:com_otaliastudios_cameraview",
        "@maven//:junit_junit",
        "@maven//:org_tensorflow_tensorflow_lite_support",
        "@maven//:com_google_android_gms_play_services_tasks",
    ],
)
