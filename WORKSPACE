#workspace(name = "mediapipe")
workspace(name = "edgeml_insight")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

#android rules
http_archive(
    name = "rules_android",
    sha256 = "cd06d15dd8bb59926e4d65f9003bfc20f9da4b2519985c27e190cddc8b7a7806",
    strip_prefix = "rules_android-0.1.1",
    urls = ["https://github.com/bazelbuild/rules_android/archive/v0.1.1.zip"],
)

## google GCP firebase
#TOOLS_ANDROID_VERSION = "0.1"
#
#http_archive(
#    name = "tools_android",
#    strip_prefix = "tools_android-" + TOOLS_ANDROID_VERSION,
#    url = "https://github.com/bazelbuild/tools_android/archive/%s.tar.gz" % TOOLS_ANDROID_VERSION,
#)

#load("@tools_android//tools/googleservices:defs.bzl", "google_services_workspace_dependencies")
#
#google_services_workspace_dependencies()

http_archive(
    name = "bazel_skylib",
    sha256 = "1c531376ac7e5a180e0237938a2536de0c54d93f5c278634818e0efc952dd56c",
    type = "tar.gz",
    urls = [
        "https://github.com/bazelbuild/bazel-skylib/releases/download/1.0.3/bazel-skylib-1.0.3.tar.gz",
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.0.3/bazel-skylib-1.0.3.tar.gz",
    ],
)

load("@bazel_skylib//:workspace.bzl", "bazel_skylib_workspace")

bazel_skylib_workspace()

load("@bazel_skylib//lib:versions.bzl", "versions")

versions.check(minimum_bazel_version = "3.7.2")

# ABSL cpp library lts_2020_09_23
http_archive(
    name = "com_google_absl",
    patch_args = [
        "-p1",
    ],
    # Remove after https://github.com/abseil/abseil-cpp/issues/326 is solved.
    patches = [
        "@//third_party:com_google_absl_f863b622fe13612433fdf43f76547d5edda0c93001.diff",
    ],
    sha256 = "b3744a4f7a249d5eaf2309daad597631ce77ea62e0fc6abffbab4b4c3dc0fc08",
    strip_prefix = "abseil-cpp-20200923",
    urls = [
        "https://github.com/abseil/abseil-cpp/archive/20200923.tar.gz",
    ],
)

http_archive(
    name = "rules_cc",
    strip_prefix = "rules_cc-main",
    urls = ["https://github.com/bazelbuild/rules_cc/archive/main.zip"],
)

http_archive(
    name = "rules_foreign_cc",
    strip_prefix = "rules_foreign_cc-0.1.0",
    url = "https://github.com/bazelbuild/rules_foreign_cc/archive/0.1.0.zip",
)

load("@rules_foreign_cc//:workspace_definitions.bzl", "rules_foreign_cc_dependencies")

rules_foreign_cc_dependencies()

# This is used to select all contents of the archives for CMake-based packages to give CMake access to them.
all_content = """filegroup(name = "all", srcs = glob(["**"]), visibility = ["//visibility:public"])"""

# GoogleTest/GoogleMock framework. Used by most unit-tests.
# Last updated 2020-06-30.
http_archive(
    name = "com_google_googletest",
    patch_args = [
        "-p1",
    ],
    patches = [
        # fix for https://github.com/google/googletest/issues/2817
        "@//third_party:com_google_googletest_9d580ea80592189e6d44fa35bcf9cdea8bf620d6.diff",
    ],
    sha256 = "04a1751f94244307cebe695a69cc945f9387a80b0ef1af21394a490697c5c895",
    strip_prefix = "googletest-aee0f9d9b5b87796ee8a0ab26b7587ec30e8858e",
    urls = ["https://github.com/google/googletest/archive/aee0f9d9b5b87796ee8a0ab26b7587ec30e8858e.zip"],
)

# Google Benchmark library.
http_archive(
    name = "com_google_benchmark",
    build_file = "@//third_party:benchmark.BUILD",
    strip_prefix = "benchmark-master",
    urls = ["https://github.com/google/benchmark/archive/master.zip"],
)

# gflags needed by glog
http_archive(
    name = "com_github_gflags_gflags",
    sha256 = "19713a36c9f32b33df59d1c79b4958434cb005b5b47dc5400a7a4b078111d9b5",
    strip_prefix = "gflags-2.2.2",
    url = "https://github.com/gflags/gflags/archive/v2.2.2.zip",
)

# 2020-08-21
http_archive(
    name = "com_github_glog_glog",
    sha256 = "58c9b3b6aaa4dd8b836c0fd8f65d0f941441fb95e27212c5eeb9979cfd3592ab",
    strip_prefix = "glog-0a2e5931bd5ff22fd3bf8999eb8ce776f159cda6",
    urls = [
        "https://github.com/google/glog/archive/0a2e5931bd5ff22fd3bf8999eb8ce776f159cda6.zip",
    ],
)

http_archive(
    name = "com_github_glog_glog_no_gflags",
    build_file = "@//third_party:glog_no_gflags.BUILD",
    patch_args = [
        "-p1",
    ],
    patches = [
        "@//third_party:com_github_glog_glog_9779e5ea6ef59562b030248947f787d1256132ae.diff",
    ],
    sha256 = "58c9b3b6aaa4dd8b836c0fd8f65d0f941441fb95e27212c5eeb9979cfd3592ab",
    strip_prefix = "glog-0a2e5931bd5ff22fd3bf8999eb8ce776f159cda6",
    urls = [
        "https://github.com/google/glog/archive/0a2e5931bd5ff22fd3bf8999eb8ce776f159cda6.zip",
    ],
)

# easyexif
http_archive(
    name = "easyexif",
    build_file = "@//third_party:easyexif.BUILD",
    strip_prefix = "easyexif-master",
    url = "https://github.com/mayanklahiri/easyexif/archive/master.zip",
)

# libyuv
http_archive(
    name = "libyuv",
    build_file = "@//third_party:libyuv.BUILD",
    # Error: operand type mismatch for `vbroadcastss' caused by commit 8a13626e42f7fdcf3a6acbb0316760ee54cda7d8.
    urls = ["https://chromium.googlesource.com/libyuv/libyuv/+archive/2525698acba9bf9b701ba6b4d9584291a1f62257.tar.gz"],
)

# Note: protobuf-javalite is no longer released as a separate download, it's included in the main Java download.
# ...but the Java download is currently broken, so we use the "source" download.
http_archive(
    name = "com_google_protobuf_javalite",
    sha256 = "a79d19dcdf9139fa4b81206e318e33d245c4c9da1ffed21c87288ed4380426f9",
    strip_prefix = "protobuf-3.11.4",
    urls = ["https://github.com/protocolbuffers/protobuf/archive/v3.11.4.tar.gz"],
)

http_archive(
    name = "com_google_protobuf",
    patch_args = [
        "-p1",
    ],
    patches = [
        "@//third_party:com_google_protobuf_fixes.diff",
    ],
    sha256 = "a79d19dcdf9139fa4b81206e318e33d245c4c9da1ffed21c87288ed4380426f9",
    strip_prefix = "protobuf-3.11.4",
    urls = ["https://github.com/protocolbuffers/protobuf/archive/v3.11.4.tar.gz"],
)

http_archive(
    name = "com_google_audio_tools",
    strip_prefix = "multichannel-audio-tools-master",
    urls = ["https://github.com/google/multichannel-audio-tools/archive/master.zip"],
)

# 2020-07-09
http_archive(
    name = "pybind11_bazel",
    sha256 = "75922da3a1bdb417d820398eb03d4e9bd067c4905a4246d35a44c01d62154d91",
    strip_prefix = "pybind11_bazel-203508e14aab7309892a1c5f7dd05debda22d9a5",
    urls = ["https://github.com/pybind/pybind11_bazel/archive/203508e14aab7309892a1c5f7dd05debda22d9a5.zip"],
)

http_archive(
    name = "pybind11",
    build_file = "@pybind11_bazel//:pybind11.BUILD",
    sha256 = "1eed57bc6863190e35637290f97a20c81cfe4d9090ac0a24f3bbf08f265eb71d",
    strip_prefix = "pybind11-2.4.3",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/pybind/pybind11/archive/v2.4.3.tar.gz",
        "https://github.com/pybind/pybind11/archive/v2.4.3.tar.gz",
    ],
)

http_archive(
    name = "ceres_solver",
    patch_args = [
        "-p1",
    ],
    patches = [
        "@//third_party:ceres_solver_compatibility_fixes.diff",
    ],
    sha256 = "db12d37b4cebb26353ae5b7746c7985e00877baa8e7b12dc4d3a1512252fff3b",
    strip_prefix = "ceres-solver-2.0.0",
    url = "https://github.com/ceres-solver/ceres-solver/archive/2.0.0.zip",
)

http_archive(
    name = "opencv",
    build_file_content = all_content,
    strip_prefix = "opencv-3.4.10",
    urls = ["https://github.com/opencv/opencv/archive/3.4.10.tar.gz"],
)

new_local_repository(
    name = "linux_opencv",
    build_file = "@//third_party:opencv_linux.BUILD",
    path = "/usr",
)

new_local_repository(
    name = "linux_ffmpeg",
    build_file = "@//third_party:ffmpeg_linux.BUILD",
    path = "/usr",
)

new_local_repository(
    name = "macos_opencv",
    build_file = "@//third_party:opencv_macos.BUILD",
    path = "/usr/local/opt/opencv@3",
)

new_local_repository(
    name = "macos_ffmpeg",
    build_file = "@//third_party:ffmpeg_macos.BUILD",
    path = "/usr/local/opt/ffmpeg",
)

new_local_repository(
    name = "windows_opencv",
    build_file = "@//third_party:opencv_windows.BUILD",
    path = "C:\\opencv\\build",
)

http_archive(
    name = "android_opencv",
    build_file = "@//third_party:opencv_android.BUILD",
    strip_prefix = "OpenCV-android-sdk",
    type = "zip",
    #    url = "https://github.com/opencv/opencv/releases/download/3.4.3/opencv-3.4.3-android-sdk.zip",
    #    url = "https://github.com/opencv/opencv/releases/download/4.5.2/opencv-4.5.2-android-sdk.zip",
    url = "https://github.com/opencv/opencv/releases/download/4.1.0/opencv-4.1.0-android-sdk.zip",
)

# After OpenCV 3.2.0, the pre-compiled opencv2.framework has google protobuf symbols, which will
# trigger duplicate symbol errors in the linking stage of building a mediapipe ios app.
# To get a higher version of OpenCV for iOS, opencv2.framework needs to be built from source with
# '-DBUILD_PROTOBUF=OFF -DBUILD_opencv_dnn=OFF'.
http_archive(
    name = "ios_opencv",
    build_file = "@//third_party:opencv_ios.BUILD",
    sha256 = "7dd536d06f59e6e1156b546bd581523d8df92ce83440002885ec5abc06558de2",
    type = "zip",
    url = "https://github.com/opencv/opencv/releases/download/3.2.0/opencv-3.2.0-ios-framework.zip",
)

# You may run setup_android.sh to install Android SDK and NDK.
android_ndk_repository(
    name = "androidndk",
    #    path = "/home/hang/Android/Sdk/ndk/18.1.5063045",
    #    path = "/home/hang/Android/Sdk/ndk/20.0.5594570",
    path = "/home/hang/Android/Sdk/ndk/21.4.7075529",
)

android_sdk_repository(
    name = "androidsdk",
    path = "/home/hang/Android/Sdk",
)

# iOS basic build deps.

#http_archive(
#    name = "build_bazel_rules_apple",
#    patch_args = [
#        "-p1",
#    ],
#    patches = [
#        # Bypass checking ios unit test runner when building MP ios applications.
#        "@//third_party:build_bazel_rules_apple_bypass_test_runner_check.diff",
#    ],
#    sha256 = "7a7afdd4869bb201c9352eed2daf37294d42b093579b70423490c1b4d4f6ce42",
#    url = "https://github.com/bazelbuild/rules_apple/releases/download/0.19.0/rules_apple.0.19.0.tar.gz",
#)

http_archive(
    name = "build_bazel_rules_apple",
    sha256 = "0052d452af7742c8f3a4e0929763388a66403de363775db7e90adecb2ba4944b",
    urls = [
        "https://github.com/bazelbuild/rules_apple/releases/download/0.31.3/rules_apple.0.31.3.tar.gz",
    ],
)

load(
    "@build_bazel_rules_apple//apple:repositories.bzl",
    "apple_rules_dependencies",
)

apple_rules_dependencies()

load(
    "@build_bazel_rules_swift//swift:repositories.bzl",
    "swift_rules_dependencies",
)

swift_rules_dependencies()

http_archive(
    name = "build_bazel_apple_support",
    sha256 = "122ebf7fe7d1c8e938af6aeaee0efe788a3a2449ece5a8d6a428cb18d6f88033",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/apple_support/releases/download/0.7.1/apple_support.0.7.1.tar.gz",
        "https://github.com/bazelbuild/apple_support/releases/download/0.7.1/apple_support.0.7.1.tar.gz",
    ],
)

load(
    "@build_bazel_apple_support//lib:repositories.bzl",
    "apple_support_dependencies",
)

apple_support_dependencies()

# More iOS deps.

http_archive(
    name = "google_toolbox_for_mac",
    build_file = "@//third_party:google_toolbox_for_mac.BUILD",
    sha256 = "e3ac053813c989a88703556df4dc4466e424e30d32108433ed6beaec76ba4fdc",
    strip_prefix = "google-toolbox-for-mac-2.2.1",
    url = "https://github.com/google/google-toolbox-for-mac/archive/v2.2.1.zip",
)

# Maven dependencies.

RULES_JVM_EXTERNAL_TAG = "4.0"

RULES_JVM_EXTERNAL_SHA = "31701ad93dbfe544d597dbe62c9a1fdd76d81d8a9150c2bf1ecf928ecdf97169"

http_archive(
    name = "rules_jvm_external",
    sha256 = RULES_JVM_EXTERNAL_SHA,
    strip_prefix = "rules_jvm_external-%s" % RULES_JVM_EXTERNAL_TAG,
    url = "https://github.com/bazelbuild/rules_jvm_external/archive/%s.zip" % RULES_JVM_EXTERNAL_TAG,
)

load("@rules_jvm_external//:defs.bzl", "maven_install")

# Important: there can only be one maven_install rule. Add new maven deps here.
maven_install(
    artifacts = [
        "androidx.concurrent:concurrent-futures:1.0.0-alpha03",
        "androidx.lifecycle:lifecycle-common:2.2.0",
        "androidx.annotation:annotation:aar:1.1.0",
        "androidx.appcompat:appcompat:aar:1.1.0-rc01",
        "androidx.camera:camera-core:1.0.0-beta10",
        "androidx.camera:camera-camera2:1.0.0-beta10",
        "androidx.camera:camera-lifecycle:1.0.0-beta10",
        "androidx.constraintlayout:constraintlayout:aar:1.1.3",
        "androidx.core:core:aar:1.1.0-rc03",
        "androidx.legacy:legacy-support-v4:aar:1.0.0",
        "androidx.recyclerview:recyclerview:aar:1.1.0-beta02",
        "androidx.test.espresso:espresso-core:3.1.1",
        "com.github.bumptech.glide:glide:4.11.0",
        "com.google.android.material:material:aar:1.0.0-rc01",
        "com.google.code.findbugs:jsr305:3.0.2",
        "com.google.flogger:flogger-system-backend:0.3.1",
        "com.google.flogger:flogger:0.3.1",
        "com.google.guava:guava:27.0.1-android",
        "com.google.guava:listenablefuture:1.0",
        "junit:junit:4.12",
        "org.hamcrest:hamcrest-library:1.3",
        #        below not from mediapipe
        "androidx.core:core-ktx:1.+",
        "com.otaliastudios:cameraview:2.+",
        "org.tensorflow:tensorflow-lite:0.0.0-nightly-SNAPSHOT",
        "org.tensorflow:tensorflow-lite-support:0.1.0",
        "org.tensorflow:tensorflow-lite-gpu:0.0.0-nightly-SNAPSHOT",
        # firebase
        #        "com.google.firebase:firebase-auth:21.+",
        "com.google.firebase:firebase-storage:20.+",
        "com.google.android.gms:play-services-tasks:17.+",
    ],
    fetch_sources = True,
    repositories = [
        "https://maven.google.com",
        "https://dl.google.com/dl/android/maven2",
        "https://repo1.maven.org/maven2",
        "https://jcenter.bintray.com",
    ],
    version_conflict_policy = "pinned",
)

# Needed by TensorFlow
http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "e0a111000aeed2051f29fcc7a3f83be3ad8c6c93c186e64beb1ad313f0c7f9f9",
    strip_prefix = "rules_closure-cf1e44edb908e9616030cc83d085989b8e6cd6df",
    urls = [
        "http://mirror.tensorflow.org/github.com/bazelbuild/rules_closure/archive/cf1e44edb908e9616030cc83d085989b8e6cd6df.tar.gz",
        "https://github.com/bazelbuild/rules_closure/archive/cf1e44edb908e9616030cc83d085989b8e6cd6df.tar.gz",  # 2019-04-04
    ],
)

# Tensorflow repo should always go after the other external dependencies.
# 2021-03-25
_TENSORFLOW_GIT_COMMIT = "c67f68021824410ebe9f18513b8856ac1c6d4887"

_TENSORFLOW_SHA256 = "fd07d0b39422dc435e268c5e53b2646a8b4b1e3151b87837b43f86068faae87f"

http_archive(
    name = "org_tensorflow",
    patch_args = [
        "-p1",
    ],
    patches = [
        "@//third_party:org_tensorflow_compatibility_fixes.diff",
        "@//third_party:org_tensorflow_objc_cxx17.diff",
    ],
    sha256 = _TENSORFLOW_SHA256,
    strip_prefix = "tensorflow-%s" % _TENSORFLOW_GIT_COMMIT,
    urls = [
        "https://github.com/tensorflow/tensorflow/archive/%s.tar.gz" % _TENSORFLOW_GIT_COMMIT,
    ],
)

load("@org_tensorflow//tensorflow:workspace3.bzl", "tf_workspace3")

tf_workspace3()

load("@org_tensorflow//tensorflow:workspace2.bzl", "tf_workspace2")

tf_workspace2()
