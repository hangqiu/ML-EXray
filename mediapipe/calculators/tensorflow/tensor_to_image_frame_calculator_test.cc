// Copyright 2018 The MediaPipe Authors.
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

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/port/gtest.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"

namespace mediapipe {

namespace tf = ::tensorflow;
namespace {

constexpr char kTensor[] = "TENSOR";
constexpr char kImage[] = "IMAGE";

}  // namespace

class TensorToImageFrameCalculatorTest : public ::testing::Test {
 protected:
  void SetUpRunner() {
    CalculatorGraphConfig::Node config;
    config.set_calculator("TensorToImageFrameCalculator");
    config.add_input_stream("TENSOR:input_tensor");
    config.add_output_stream("IMAGE:output_image");
    runner_ = absl::make_unique<CalculatorRunner>(config);
  }

  std::unique_ptr<CalculatorRunner> runner_;
};

TEST_F(TensorToImageFrameCalculatorTest, Converts3DTensorToImageFrame) {
  SetUpRunner();
  constexpr int kWidth = 16;
  constexpr int kHeight = 8;
  const tf::TensorShape tensor_shape(
      std::vector<tf::int64>{kHeight, kWidth, 3});
  auto tensor = absl::make_unique<tf::Tensor>(tf::DT_FLOAT, tensor_shape);
  auto tensor_vec = tensor->flat<float>().data();

  // Writing sequence of integers as floats which we want back (as they were
  // written).
  for (int i = 0; i < kWidth * kHeight * 3; ++i) {
    tensor_vec[i] = i % 255;
  }

  const int64 time = 1234;
  runner_->MutableInputs()->Tag(kTensor).packets.push_back(
      Adopt(tensor.release()).At(Timestamp(time)));

  EXPECT_TRUE(runner_->Run().ok());
  const std::vector<Packet>& output_packets =
      runner_->Outputs().Tag(kImage).packets;
  EXPECT_EQ(1, output_packets.size());
  EXPECT_EQ(time, output_packets[0].Timestamp().Value());
  const ImageFrame& output_image = output_packets[0].Get<ImageFrame>();
  EXPECT_EQ(kWidth, output_image.Width());
  EXPECT_EQ(kHeight, output_image.Height());

  for (int i = 0; i < kWidth * kHeight * 3; ++i) {
    const uint8 pixel_value = output_image.PixelData()[i];
    EXPECT_EQ(i % 255, pixel_value);
  }
}

TEST_F(TensorToImageFrameCalculatorTest, Converts3DTensorToImageFrameGray) {
  SetUpRunner();
  constexpr int kWidth = 16;
  constexpr int kHeight = 8;
  const tf::TensorShape tensor_shape(
      std::vector<tf::int64>{kHeight, kWidth, 1});
  auto tensor = absl::make_unique<tf::Tensor>(tf::DT_FLOAT, tensor_shape);
  auto tensor_vec = tensor->flat<float>().data();

  // Writing sequence of integers as floats which we want back (as they were
  // written).
  for (int i = 0; i < kWidth * kHeight; ++i) {
    tensor_vec[i] = i % 255;
  }

  const int64 time = 1234;
  runner_->MutableInputs()->Tag(kTensor).packets.push_back(
      Adopt(tensor.release()).At(Timestamp(time)));

  EXPECT_TRUE(runner_->Run().ok());
  const std::vector<Packet>& output_packets =
      runner_->Outputs().Tag(kImage).packets;
  EXPECT_EQ(1, output_packets.size());
  EXPECT_EQ(time, output_packets[0].Timestamp().Value());
  const ImageFrame& output_image = output_packets[0].Get<ImageFrame>();
  EXPECT_EQ(kWidth, output_image.Width());
  EXPECT_EQ(kHeight, output_image.Height());

  for (int i = 0; i < kWidth * kHeight; ++i) {
    const uint8 pixel_value = output_image.PixelData()[i];
    EXPECT_EQ(i % 255, pixel_value);
  }
}

TEST_F(TensorToImageFrameCalculatorTest, Converts3DTensorToImageFrame2DGray) {
  SetUpRunner();
  constexpr int kWidth = 16;
  constexpr int kHeight = 8;
  const tf::TensorShape tensor_shape(std::vector<tf::int64>{kHeight, kWidth});
  auto tensor = absl::make_unique<tf::Tensor>(tf::DT_FLOAT, tensor_shape);
  auto tensor_vec = tensor->flat<float>().data();

  // Writing sequence of integers as floats which we want back (as they were
  // written).
  for (int i = 0; i < kWidth * kHeight; ++i) {
    tensor_vec[i] = i % 255;
  }

  const int64 time = 1234;
  runner_->MutableInputs()->Tag(kTensor).packets.push_back(
      Adopt(tensor.release()).At(Timestamp(time)));

  EXPECT_TRUE(runner_->Run().ok());
  const std::vector<Packet>& output_packets =
      runner_->Outputs().Tag(kImage).packets;
  EXPECT_EQ(1, output_packets.size());
  EXPECT_EQ(time, output_packets[0].Timestamp().Value());
  const ImageFrame& output_image = output_packets[0].Get<ImageFrame>();
  EXPECT_EQ(kWidth, output_image.Width());
  EXPECT_EQ(kHeight, output_image.Height());

  for (int i = 0; i < kWidth * kHeight; ++i) {
    const uint8 pixel_value = output_image.PixelData()[i];
    EXPECT_EQ(i % 255, pixel_value);
  }
}

}  // namespace mediapipe
