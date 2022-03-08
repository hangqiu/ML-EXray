// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mediapipe/framework/deps/status_builder.h"

#include "mediapipe/framework/port/gtest.h"

namespace mediapipe {

TEST(StatusBuilder, AnnotateMode) {
  absl::Status status = StatusBuilder(absl::Status(absl::StatusCode::kNotFound,
                                                   "original message"),
                                      MEDIAPIPE_LOC)
                        << "annotated message1 "
                        << "annotated message2";
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kNotFound);
  EXPECT_EQ(status.message(),
            "original message; annotated message1 annotated message2");
}

TEST(StatusBuilder, PrependMode) {
  absl::Status status =
      StatusBuilder(
          absl::Status(absl::StatusCode::kInvalidArgument, "original message"),
          MEDIAPIPE_LOC)
          .SetPrepend()
      << "prepended message1 "
      << "prepended message2 ";
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(status.message(),
            "prepended message1 prepended message2 original message");
}

TEST(StatusBuilder, AppendMode) {
  absl::Status status = StatusBuilder(absl::Status(absl::StatusCode::kInternal,
                                                   "original message"),
                                      MEDIAPIPE_LOC)
                            .SetAppend()
                        << " extra message1"
                        << " extra message2";
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
  EXPECT_EQ(status.message(), "original message extra message1 extra message2");
}

TEST(StatusBuilder, NoLoggingMode) {
  absl::Status status =
      StatusBuilder(
          absl::Status(absl::StatusCode::kUnavailable, "original message"),
          MEDIAPIPE_LOC)
          .SetNoLogging()
      << " extra message";
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kUnavailable);
  EXPECT_EQ(status.message(), "original message");
}

}  // namespace mediapipe
