// Copyright 2020 The MediaPipe Authors.
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

#ifndef MEDIAPIPE_FRAMEWORK_FORMATS_TENSOR_H_
#define MEDIAPIPE_FRAMEWORK_FORMATS_TENSOR_H_

#include <algorithm>
#include <initializer_list>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/synchronization/mutex.h"
#include "mediapipe/framework/port.h"

#if MEDIAPIPE_METAL_ENABLED
#import <Metal/Metal.h>
#endif  // MEDIAPIPE_METAL_ENABLED

#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_30
#include "mediapipe/gpu/gl_base.h"
#include "mediapipe/gpu/gl_context.h"
#endif  // MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_30

namespace mediapipe {

// Tensor is a container of multi-dimensional data that supports sharing the
// content across different backends and APIs, currently: CPU / Metal / OpenGL.
// Texture2DView is limited to 4 dimensions.
// The content is accessible through requesting device specific views.
// Acquiring a view guarantees that the content is not changed by another thread
// until the view is released.
//
// Tensor::MtlBufferView view = tensor.GetMtlBufferWriteView(mtl_device);
// mtl_device is used to create MTLBuffer
// id<MTLBuffer> buffer = view.buffer();
// For OpenGL the code below must be called by a thread with valid OpenGL ES
// context bound:
// GLuint buffer = view.buffer();
// Then the buffer can be bound to the GPU command buffer.
// ...binding the buffer to the command buffer...
// ...commiting command buffer and releasing the view...
//
// The following request for the CPU view will be blocked until the GPU view is
// released and the GPU task is finished.
//
// auto view = tensor.GetCpuReadView();
// float* pointer = view.buffer<float>();
// ...reading the cpu memory...

class Tensor {
  class View {
   public:
    // Non-copyable.
    View(const View&) = delete;
    View& operator=(const View&) = delete;
    View(View&& src) = default;

   protected:
    View(std::unique_ptr<absl::MutexLock>&& lock) : lock_(std::move(lock)) {}
    std::unique_ptr<absl::MutexLock> lock_;
  };

 public:
  // No resources are allocated here.
  enum class ElementType { kNone, kFloat16, kFloat32 };
  struct Shape {
    Shape() = default;
    Shape(std::initializer_list<int> dimensions) : dims(dimensions) {}
    Shape(const std::vector<int>& dimensions) : dims(dimensions) {}
    int num_elements() const {
      int res = dims.empty() ? 0 : 1;
      std::for_each(dims.begin(), dims.end(), [&res](int i) { res *= i; });
      return res;
    }
    std::vector<int> dims;
  };

  Tensor(ElementType element_type, const Shape& shape);

  // Non-copyable.
  Tensor(const Tensor&) = delete;
  Tensor& operator=(const Tensor&) = delete;
  // Move-only.
  Tensor(Tensor&& src) { Move(&src); }
  Tensor& operator=(Tensor&&);
  ~Tensor() { Invalidate(); }

  template <typename T>
  class CpuView : public View {
   public:
    template <typename P>
    auto buffer() const {
      // const and non-const return  type selection.
      return static_cast<typename std::tuple_element<
          std::is_const<T>::value, std::tuple<P*, const P*> >::type>(buffer_);
    }
    CpuView(CpuView&& src) : View(std::move(src)), buffer_(src.buffer_) {
      src.buffer_ = nullptr;
    }

   protected:
    friend class Tensor;
    CpuView(T* buffer, std::unique_ptr<absl::MutexLock>&& lock)
        : View(std::move(lock)), buffer_(buffer) {}
    T* buffer_;
  };
  using CpuReadView = CpuView<const void>;
  CpuReadView GetCpuReadView() const;
  using CpuWriteView = CpuView<void>;
  CpuWriteView GetCpuWriteView() const;

#if MEDIAPIPE_METAL_ENABLED
  // TODO: id<MTLBuffer> vs. MtlBufferView.
  class MtlBufferView : public View {
   public:
    id<MTLBuffer> buffer() const { return buffer_; }
    MtlBufferView(MtlBufferView&& src)
        : View(std::move(src)), buffer_(src.buffer_) {
      src.buffer_ = nil;
    }

   protected:
    friend class Tensor;
    MtlBufferView(id<MTLBuffer> buffer, std::unique_ptr<absl::MutexLock>&& lock)
        : View(std::move(lock)), buffer_(buffer) {}
    id<MTLBuffer> buffer_;
  };
  // The command buffer status is checked for completeness if GPU-to-CPU
  // synchronization is required.
  // TODO: Design const and non-const view acquiring.
  MtlBufferView GetMtlBufferReadView(id<MTLCommandBuffer> command_buffer) const;
  MtlBufferView GetMtlBufferWriteView(
      id<MTLCommandBuffer> command_buffer) const;
  // Allocate new buffer.
  // TODO: GPU-to-CPU design considerations.
  MtlBufferView GetMtlBufferWriteView(id<MTLDevice> device) const;
#endif  // MEDIAPIPE_METAL_ENABLED

#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_30
  // TODO: Use GlTextureView instead.
  // Only float32 textures are supported with 1/2/3/4 depths.
  // OpenGlTexture2dView currently only supports BHWC memory layout.
  class OpenGlTexture2dView : public View {
   public:
    GLuint name() const { return name_; }
    OpenGlTexture2dView(OpenGlTexture2dView&& src)
        : View(std::move(src)), name_(src.name_) {
      src.name_ = GL_INVALID_INDEX;
    }
    // To fit a tensor into a texture two layouts are used:
    // 1. Aligned. Width of the texture = tensor_width * num_slices, where slice
    //    is a group of 4 depth values. Tensor depth is padded to 4.
    // 2. Linearized. If texture width or height with the layout 1. is greater
    //    than the GPU supports then all tensor values are packed into a texture
    //    with fixed width calculated by this method.
    // Must be called with the valid GL context bound to the current thread.
    enum class Layout { kAligned, kLinearized };
    static Layout GetLayoutDimensions(const Tensor::Shape& shape, int* width,
                                      int* height);

   protected:
    friend class Tensor;
    OpenGlTexture2dView(GLuint name, std::unique_ptr<absl::MutexLock>&& lock)
        : View(std::move(lock)), name_(name) {}
    GLuint name_;
  };
  // A valid OpenGL context must be bound to the calling thread due to possible
  // GPU resource allocation.
  OpenGlTexture2dView GetOpenGlTexture2dReadView() const;
  OpenGlTexture2dView GetOpenGlTexture2dWriteView() const;
#endif  // MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_30

#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
  class OpenGlBufferView : public View {
   public:
    GLuint name() const { return name_; }
    OpenGlBufferView(OpenGlBufferView&& src)
        : View(std::move(src)), name_(src.name_) {
      src.name_ = GL_INVALID_INDEX;
    }

   protected:
    friend class Tensor;
    OpenGlBufferView(GLuint name, std::unique_ptr<absl::MutexLock>&& lock)
        : View(std::move(lock)), name_(name) {}
    GLuint name_;
  };
  // A valid OpenGL context must be bound to the calling thread due to possible
  // GPU resource allocation.
  OpenGlBufferView GetOpenGlBufferReadView() const;
  OpenGlBufferView GetOpenGlBufferWriteView() const;
#endif  // MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31

  const Shape& shape() const { return shape_; }
  ElementType element_type() const { return element_type_; }
  int element_size() const {
    switch (element_type_) {
      case ElementType::kNone:
        return 0;
      case ElementType::kFloat16:
        return 2;
      case ElementType::kFloat32:
        return sizeof(float);
    }
  }
  int bytes() const { return shape_.num_elements() * element_size(); }

  bool ready_on_cpu() const { return valid_ & kValidCpu; }
  bool ready_on_gpu() const {
    return valid_ &
           (kValidMetalBuffer | kValidOpenGlBuffer | kValidOpenGlTexture2d);
  }
  bool ready_as_metal_buffer() const { return valid_ & kValidMetalBuffer; }
  bool ready_as_opengl_buffer() const { return valid_ & kValidOpenGlBuffer; }
  bool ready_as_opengl_texture_2d() const {
    return valid_ & kValidOpenGlTexture2d;
  }

 private:
  void Move(Tensor*);
  void Invalidate();

  ElementType element_type_;
  Shape shape_;

  // The flags describe the current source of truth resource type.
  enum {
    kValidNone = 0,
    kValidCpu = 1 << 0,
    kValidMetalBuffer = 1 << 1,
    kValidOpenGlBuffer = 1 << 2,
    kValidOpenGlTexture2d = 1 << 3,
  };
  // A list of resource which are currently allocated and synchronized between
  // each-other: valid_ = kValidCpu | kValidMetalBuffer;
  mutable int valid_ = 0;
  // The mutex is locked by Get*View and is kept by all Views.
  mutable absl::Mutex view_mutex_;

  mutable void* cpu_buffer_ = nullptr;
  void AllocateCpuBuffer() const;
#if MEDIAPIPE_METAL_ENABLED
  mutable id<MTLCommandBuffer> command_buffer_;
  mutable id<MTLDevice> device_;
  mutable id<MTLBuffer> metal_buffer_;
  void AllocateMtlBuffer(id<MTLDevice> device) const;
#endif  // MEDIAPIPE_METAL_ENABLED

#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_30
  mutable std::shared_ptr<mediapipe::GlContext> gl_context_;
  mutable GLuint opengl_texture2d_ = GL_INVALID_INDEX;
  mutable GLuint frame_buffer_ = GL_INVALID_INDEX;
  mutable int texture_width_;
  mutable int texture_height_;
  void AllocateOpenGlTexture2d() const;
#if MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
  mutable GLuint opengl_buffer_ = GL_INVALID_INDEX;
  void AllocateOpenGlBuffer() const;
#endif  // MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_31
#endif  // MEDIAPIPE_OPENGL_ES_VERSION >= MEDIAPIPE_OPENGL_ES_30
};

}  // namespace mediapipe

#endif  // MEDIAPIPE_FRAMEWORK_FORMATS_TENSOR_H_
