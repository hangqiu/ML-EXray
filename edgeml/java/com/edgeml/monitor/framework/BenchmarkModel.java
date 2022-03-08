/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package com.edgeml.monitor.framework;

import android.os.Trace;
import android.util.Log;

/** Helper class for running a native TensorFlow Lite benchmark. */
public class BenchmarkModel {
  static {
    System.loadLibrary("tensorflowlite_benchmark");
  }

  private static final String TAG = "tflite_BenchmarkModelActivity";

  // Executes a standard TensorFlow Lite benchmark according to the provided args.
  //
  // Note that {@code args} will be split by the native execution code.
  public static void run_benchmark() {
    String args = "--graph=/data/local/tmp/model.tflite --num_threads=4 --enable_op_profiling=true " +
            "--profiling_output_csv_file=/profile.csv";
    Log.i(TAG, "Running TensorFlow Lite benchmark with args: " + args);

    Trace.beginSection("TFLite Benchmark Model");
    nativeRunBenchmark(args);
    Trace.endSection();
  }


  private static native void nativeRunBenchmark(String args);
}
