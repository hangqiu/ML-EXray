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

#include "mediapipe/objc/util.h"

#include "absl/base/macros.h"
#include "absl/memory/memory.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/source_location.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_builder.h"

namespace {

// NOTE: you must release the colorspace returned by this function, unless
// it's null.
// Returns an invalid format (all fields 0) if the requested format is
// unsupported.
vImage_CGImageFormat vImageFormatForCVPixelFormat(OSType pixel_format) {
  switch (pixel_format) {
    case kCVPixelFormatType_OneComponent8:
      return {
          .bitsPerComponent = 8,
          .bitsPerPixel = 8,
          .colorSpace = CGColorSpaceCreateDeviceGray(),
          .bitmapInfo = kCGImageAlphaNone | kCGBitmapByteOrderDefault,
      };

    case kCVPixelFormatType_32BGRA:
      return {
          .bitsPerComponent = 8,
          .bitsPerPixel = 32,
          .colorSpace = NULL,
          .bitmapInfo = kCGImageAlphaFirst | kCGBitmapByteOrder32Little,
      };

    case kCVPixelFormatType_32RGBA:
      return {
          .bitsPerComponent = 8,
          .bitsPerPixel = 32,
          .colorSpace = NULL,
          .bitmapInfo = kCGImageAlphaLast | kCGBitmapByteOrderDefault,
      };

    default:
      return {};
  }
}

CGColorSpaceRef CreateConversionCGColorSpaceForPixelFormat(
    OSType pixel_format) {
  // According to vImage documentation, YUV formats require the RGB colorspace
  // in which the RGB conversion should be interpreted. sRGB is suggested.
  // We cannot just pass sRGB all the time, though, since it breaks with
  // monochrome.
  switch (pixel_format) {
    case kCVPixelFormatType_422YpCbCr8:
    case kCVPixelFormatType_4444YpCbCrA8:
    case kCVPixelFormatType_4444YpCbCrA8R:
    case kCVPixelFormatType_4444AYpCbCr8:
    case kCVPixelFormatType_4444AYpCbCr16:
    case kCVPixelFormatType_444YpCbCr8:
    case kCVPixelFormatType_422YpCbCr16:
    case kCVPixelFormatType_422YpCbCr10:
    case kCVPixelFormatType_444YpCbCr10:
    case kCVPixelFormatType_420YpCbCr8Planar:
    case kCVPixelFormatType_420YpCbCr8PlanarFullRange:
    case kCVPixelFormatType_422YpCbCr_4A_8BiPlanar:
    case kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange:
    case kCVPixelFormatType_420YpCbCr8BiPlanarFullRange:
    case kCVPixelFormatType_422YpCbCr8_yuvs:
    case kCVPixelFormatType_422YpCbCr8FullRange:
      return CGColorSpaceCreateWithName(kCGColorSpaceSRGB);

    default:
      return NULL;
  }
}

vImageConverterRef vImageConverterForCVPixelFormats(OSType src_pixel_format,
                                                    OSType dst_pixel_format,
                                                    vImage_Error* error) {
  static CGFloat default_background[3] = {1.0, 1.0, 1.0};
  vImageConverterRef converter = NULL;

  vImage_CGImageFormat src_cg_format =
      vImageFormatForCVPixelFormat(src_pixel_format);
  vImage_CGImageFormat dst_cg_format =
      vImageFormatForCVPixelFormat(dst_pixel_format);

  // Use CV format functions if available (introduced in iOS 8).
  // Weak-linked symbols are NULL when not available.
  if (&vImageConverter_CreateForCGToCVImageFormat != NULL) {
    // Strangely, there is no function to convert between two
    // vImageCVImageFormat, so one side has to use a vImage_CGImageFormat
    // that we have to find ourselves.
    if (src_cg_format.bitsPerComponent > 0) {
      // We can handle source using a CGImageFormat.
      // TODO: check the final alpha hint parameter
      CGColorSpaceRef cv_color_space =
          CreateConversionCGColorSpaceForPixelFormat(dst_pixel_format);
      vImageCVImageFormatRef dst_cv_format = vImageCVImageFormat_Create(
          dst_pixel_format, kvImage_ARGBToYpCbCrMatrix_ITU_R_709_2,
          kCVImageBufferChromaLocation_Center, cv_color_space, 1);
      CGColorSpaceRelease(cv_color_space);

      converter = vImageConverter_CreateForCGToCVImageFormat(
          &src_cg_format, dst_cv_format, default_background,
          kvImagePrintDiagnosticsToConsole, error);
      vImageCVImageFormat_Release(dst_cv_format);
    } else if (dst_cg_format.bitsPerComponent > 0) {
      // We can use a CGImageFormat for the destination.
      CGColorSpaceRef cv_color_space =
          CreateConversionCGColorSpaceForPixelFormat(src_pixel_format);
      vImageCVImageFormatRef src_cv_format = vImageCVImageFormat_Create(
          src_pixel_format, kvImage_ARGBToYpCbCrMatrix_ITU_R_709_2,
          kCVImageBufferChromaLocation_Center, cv_color_space, 1);
      CGColorSpaceRelease(cv_color_space);

      converter = vImageConverter_CreateForCVToCGImageFormat(
          src_cv_format, &dst_cg_format, default_background,
          kvImagePrintDiagnosticsToConsole, error);
      vImageCVImageFormat_Release(src_cv_format);
    }
  }

  if (!converter) {
    // Try a CG to CG conversion.
    if (src_cg_format.bitsPerComponent > 0 &&
        dst_cg_format.bitsPerComponent > 0) {
      converter = vImageConverter_CreateWithCGImageFormat(
          &src_cg_format, &dst_cg_format, default_background, kvImageNoFlags,
          error);
    }
  }

  CGColorSpaceRelease(src_cg_format.colorSpace);
  CGColorSpaceRelease(dst_cg_format.colorSpace);
  return converter;
}

}  // unnamed namespace

vImage_Error vImageGrayToBGRA(const vImage_Buffer* src, vImage_Buffer* dst) {
  static vImageConverterRef converter = NULL;
  if (!converter) {
    converter = vImageConverterForCVPixelFormats(
        kCVPixelFormatType_OneComponent8, kCVPixelFormatType_32BGRA, NULL);
  }
  return vImageConvert_AnyToAny(converter, src, dst, NULL, kvImageNoFlags);
}

vImage_Error vImageBGRAToGray(const vImage_Buffer* src, vImage_Buffer* dst) {
  static vImageConverterRef converter = NULL;
  if (!converter) {
    converter = vImageConverterForCVPixelFormats(
        kCVPixelFormatType_32BGRA, kCVPixelFormatType_OneComponent8, NULL);
  }
  return vImageConvert_AnyToAny(converter, src, dst, NULL, kvImageNoFlags);
}

vImage_Error vImageRGBAToGray(const vImage_Buffer* src, vImage_Buffer* dst) {
  static vImageConverterRef converter = NULL;
  if (!converter) {
    converter = vImageConverterForCVPixelFormats(
        kCVPixelFormatType_32RGBA, kCVPixelFormatType_OneComponent8, NULL);
  }
  return vImageConvert_AnyToAny(converter, src, dst, NULL, kvImageNoFlags);
}

vImage_Error vImageConvertCVPixelBuffers(CVPixelBufferRef src,
                                         CVPixelBufferRef dst) {
  //  CGColorSpaceRef srgb_color_space =
  //  CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
  vImage_Error error;
  vImageConverterRef converter = vImageConverterForCVPixelFormats(
      CVPixelBufferGetPixelFormatType(src),
      CVPixelBufferGetPixelFormatType(dst), &error);
  if (!converter) {
    return error;
  }

  int src_buffer_count = vImageConverter_GetNumberOfSourceBuffers(converter);
  int dst_buffer_count =
      vImageConverter_GetNumberOfDestinationBuffers(converter);
  vImage_Buffer buffers[8];
  if (src_buffer_count + dst_buffer_count > ABSL_ARRAYSIZE(buffers)) {
    vImageConverter_Release(converter);
    return kvImageMemoryAllocationError;
  }
  vImage_Buffer* src_bufs = buffers;
  vImage_Buffer* dst_bufs = buffers + src_buffer_count;

  // vImageBuffer_InitForCopyToCVPixelBuffer can be used only if the converter
  // was created by vImageConverter_CreateForCGToCVImageFormat.
  // vImageBuffer_InitForCopyFromCVPixelBuffer can be used only if the converter
  // was created by vImageConverter_CreateForCVToCGImageFormat.
  // There does not seem to be a way to ask the converter for its type; however,
  // it is documented that all multi-planar formats are CV formats, so we use
  // these calls when there are multiple buffers.

  if (src_buffer_count > 1) {
    error = vImageBuffer_InitForCopyFromCVPixelBuffer(
        src_bufs, converter, src,
        kvImageNoAllocate | kvImagePrintDiagnosticsToConsole);
    if (error != kvImageNoError) {
      vImageConverter_Release(converter);
      return error;
    }
  } else {
    *src_bufs = vImageForCVPixelBuffer(src);
  }

  if (dst_buffer_count > 1) {
    error = vImageBuffer_InitForCopyToCVPixelBuffer(
        dst_bufs, converter, dst,
        kvImageNoAllocate | kvImagePrintDiagnosticsToConsole);
    if (error != kvImageNoError) {
      vImageConverter_Release(converter);
      return error;
    }
  } else {
    *dst_bufs = vImageForCVPixelBuffer(dst);
  }

  error = vImageConvert_AnyToAny(converter, src_bufs, dst_bufs, NULL,
                                 kvImageNoFlags);
  vImageConverter_Release(converter);
  return error;
}

void ReleaseMediaPipePacket(void* refcon, const void* base_address) {
  auto packet = (mediapipe::Packet*)refcon;
  delete packet;
}

CVPixelBufferRef CreateCVPixelBufferForImageFramePacket(
    const mediapipe::Packet& image_frame_packet) {
  CFHolder<CVPixelBufferRef> buffer;
  absl::Status status =
      CreateCVPixelBufferForImageFramePacket(image_frame_packet, &buffer);
  MEDIAPIPE_CHECK_OK(status) << "Failed to create CVPixelBufferRef";
  return (CVPixelBufferRef)CFRetain(*buffer);
}

absl::Status CreateCVPixelBufferForImageFramePacket(
    const mediapipe::Packet& image_frame_packet,
    CFHolder<CVPixelBufferRef>* out_buffer) {
  return CreateCVPixelBufferForImageFramePacket(image_frame_packet, false,
                                                out_buffer);
}

absl::Status CreateCVPixelBufferForImageFramePacket(
    const mediapipe::Packet& image_frame_packet, bool can_overwrite,
    CFHolder<CVPixelBufferRef>* out_buffer) {
  if (!out_buffer) {
    return ::mediapipe::InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
           << "out_buffer cannot be NULL";
  }
  CFHolder<CVPixelBufferRef> pixel_buffer;

  auto packet_copy = absl::make_unique<mediapipe::Packet>(image_frame_packet);
  const auto& frame = packet_copy->Get<mediapipe::ImageFrame>();
  void* frame_data =
      const_cast<void*>(reinterpret_cast<const void*>(frame.PixelData()));

  mediapipe::ImageFormat::Format image_format = frame.Format();
  OSType pixel_format = 0;
  CVReturn status;
  switch (image_format) {
    case mediapipe::ImageFormat::SRGBA: {
      pixel_format = kCVPixelFormatType_32BGRA;
      // Swap R and B channels.
      vImage_Buffer v_image = vImageForImageFrame(frame);
      vImage_Buffer v_dest;
      if (can_overwrite) {
        v_dest = v_image;
      } else {
        CVPixelBufferRef pixel_buffer_temp;
        status = CVPixelBufferCreate(
            kCFAllocatorDefault, frame.Width(), frame.Height(), pixel_format,
            GetCVPixelBufferAttributesForGlCompatibility(), &pixel_buffer_temp);
        RET_CHECK(status == kCVReturnSuccess)
            << "CVPixelBufferCreate failed: " << status;
        pixel_buffer.adopt(pixel_buffer_temp);
        status = CVPixelBufferLockBaseAddress(*pixel_buffer,
                                              kCVPixelBufferLock_ReadOnly);
        RET_CHECK(status == kCVReturnSuccess)
            << "CVPixelBufferLockBaseAddress failed: " << status;
        v_dest = vImageForCVPixelBuffer(*pixel_buffer);
      }
      const uint8_t permute_map[4] = {2, 1, 0, 3};
      vImage_Error vError = vImagePermuteChannels_ARGB8888(
          &v_image, &v_dest, permute_map, kvImageNoFlags);
      RET_CHECK(vError == kvImageNoError)
          << "vImagePermuteChannels failed: " << vError;
    } break;

    case mediapipe::ImageFormat::GRAY8:
      pixel_format = kCVPixelFormatType_OneComponent8;
      break;

    case mediapipe::ImageFormat::VEC32F1:
      pixel_format = kCVPixelFormatType_OneComponent32Float;
      break;

    case mediapipe::ImageFormat::VEC32F2:
      pixel_format = kCVPixelFormatType_TwoComponent32Float;
      break;

    default:
      return ::mediapipe::UnknownErrorBuilder(MEDIAPIPE_LOC)
             << "unsupported ImageFrame format: " << image_format;
  }

  if (*pixel_buffer) {
    status = CVPixelBufferUnlockBaseAddress(*pixel_buffer,
                                            kCVPixelBufferLock_ReadOnly);
    RET_CHECK(status == kCVReturnSuccess)
        << "CVPixelBufferUnlockBaseAddress failed: " << status;
  } else {
    CVPixelBufferRef pixel_buffer_temp;
    status = CVPixelBufferCreateWithBytes(
        NULL, frame.Width(), frame.Height(), pixel_format, frame_data,
        frame.WidthStep(), ReleaseMediaPipePacket, packet_copy.release(),
        GetCVPixelBufferAttributesForGlCompatibility(), &pixel_buffer_temp);
    RET_CHECK(status == kCVReturnSuccess)
        << "failed to create pixel buffer: " << status;
    pixel_buffer.adopt(pixel_buffer_temp);
  }

  *out_buffer = pixel_buffer;
  return absl::OkStatus();
}

absl::Status CreateCGImageFromCVPixelBuffer(CVPixelBufferRef image_buffer,
                                            CFHolder<CGImageRef>* image) {
  CVReturn status =
      CVPixelBufferLockBaseAddress(image_buffer, kCVPixelBufferLock_ReadOnly);
  RET_CHECK(status == kCVReturnSuccess)
      << "CVPixelBufferLockBaseAddress failed: " << status;

  void* base_address = CVPixelBufferGetBaseAddress(image_buffer);
  size_t bytes_per_row = CVPixelBufferGetBytesPerRow(image_buffer);
  size_t width = CVPixelBufferGetWidth(image_buffer);
  size_t height = CVPixelBufferGetHeight(image_buffer);
  OSType pixel_format = CVPixelBufferGetPixelFormatType(image_buffer);

  CGColorSpaceRef color_space = nullptr;
  uint32_t bitmap_info = 0;
  switch (pixel_format) {
    case kCVPixelFormatType_32BGRA:
      color_space = CGColorSpaceCreateDeviceRGB();
      bitmap_info =
          kCGBitmapByteOrder32Little | kCGImageAlphaPremultipliedFirst;
      break;

    case kCVPixelFormatType_OneComponent8:
      color_space = CGColorSpaceCreateDeviceGray();
      bitmap_info = kCGImageAlphaNone;
      break;

    default:
      LOG(FATAL) << "Unsupported pixelFormat " << pixel_format;
      break;
  }

  CGContextRef src_context = CGBitmapContextCreate(
      base_address, width, height, 8, bytes_per_row, color_space, bitmap_info);

  CGImageRef quartz_image = CGBitmapContextCreateImage(src_context);
  CGContextRelease(src_context);
  CGColorSpaceRelease(color_space);
  CFHolder<CGImageRef> cg_image_holder = MakeCFHolderAdopting(quartz_image);
  status =
      CVPixelBufferUnlockBaseAddress(image_buffer, kCVPixelBufferLock_ReadOnly);
  RET_CHECK(status == kCVReturnSuccess)
      << "CVPixelBufferUnlockBaseAddress failed: " << status;

  *image = cg_image_holder;
  return absl::OkStatus();
}

absl::Status CreateCVPixelBufferFromCGImage(
    CGImageRef image, CFHolder<CVPixelBufferRef>* out_buffer) {
  size_t width = CGImageGetWidth(image);
  size_t height = CGImageGetHeight(image);
  CFHolder<CVPixelBufferRef> pixel_buffer;

  CVPixelBufferRef pixel_buffer_temp;
  CVReturn status = CVPixelBufferCreate(
      kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA,
      GetCVPixelBufferAttributesForGlCompatibility(), &pixel_buffer_temp);
  RET_CHECK(status == kCVReturnSuccess)
      << "failed to create pixel buffer: " << status;
  pixel_buffer.adopt(pixel_buffer_temp);

  status = CVPixelBufferLockBaseAddress(*pixel_buffer, 0);
  RET_CHECK(status == kCVReturnSuccess)
      << "CVPixelBufferLockBaseAddress failed: " << status;

  void* base_address = CVPixelBufferGetBaseAddress(*pixel_buffer);
  CGColorSpaceRef color_space = CGColorSpaceCreateDeviceRGB();
  size_t bytes_per_row = CVPixelBufferGetBytesPerRow(*pixel_buffer);
  CGContextRef context = CGBitmapContextCreate(
      base_address, width, height, 8, bytes_per_row, color_space,
      kCGBitmapByteOrder32Little | kCGImageAlphaPremultipliedFirst);
  CGRect rect = CGRectMake(0, 0, width, height);
  CGContextClearRect(context, rect);
  CGContextDrawImage(context, rect, image);

  CGContextRelease(context);
  CGColorSpaceRelease(color_space);
  status = CVPixelBufferUnlockBaseAddress(*pixel_buffer, 0);
  RET_CHECK(status == kCVReturnSuccess)
      << "CVPixelBufferUnlockBaseAddress failed: " << status;

  *out_buffer = pixel_buffer;
  return absl::OkStatus();
}

std::unique_ptr<mediapipe::ImageFrame> CreateImageFrameForCVPixelBuffer(
    CVPixelBufferRef image_buffer) {
  return CreateImageFrameForCVPixelBuffer(image_buffer, false, false);
}

std::unique_ptr<mediapipe::ImageFrame> CreateImageFrameForCVPixelBuffer(
    CVPixelBufferRef image_buffer, bool can_overwrite, bool bgr_as_rgb) {
  CVReturn status =
      CVPixelBufferLockBaseAddress(image_buffer, kCVPixelBufferLock_ReadOnly);
  CHECK_EQ(status, kCVReturnSuccess)
      << "CVPixelBufferLockBaseAddress failed: " << status;

  void* base_address = CVPixelBufferGetBaseAddress(image_buffer);
  size_t bytes_per_row = CVPixelBufferGetBytesPerRow(image_buffer);
  size_t width = CVPixelBufferGetWidth(image_buffer);
  size_t height = CVPixelBufferGetHeight(image_buffer);
  std::unique_ptr<mediapipe::ImageFrame> frame;

  CVPixelBufferRetain(image_buffer);

  OSType pixel_format = CVPixelBufferGetPixelFormatType(image_buffer);
  mediapipe::ImageFormat::Format image_format = mediapipe::ImageFormat::UNKNOWN;
  switch (pixel_format) {
    case kCVPixelFormatType_32BGRA: {
      image_format = mediapipe::ImageFormat::SRGBA;
      if (!bgr_as_rgb) {
        // Swap R and B channels.
        vImage_Buffer v_image = vImageForCVPixelBuffer(image_buffer);
        vImage_Buffer v_dest;
        if (can_overwrite) {
          v_dest = v_image;
        } else {
          frame = absl::make_unique<mediapipe::ImageFrame>(image_format, width,
                                                           height);
          v_dest = vImageForImageFrame(*frame);
        }
        const uint8_t permute_map[4] = {2, 1, 0, 3};
        vImage_Error vError = vImagePermuteChannels_ARGB8888(
            &v_image, &v_dest, permute_map, kvImageNoFlags);
        CHECK(vError == kvImageNoError)
            << "vImagePermuteChannels failed: " << vError;
      }
    } break;

    case kCVPixelFormatType_32RGBA:
      image_format = mediapipe::ImageFormat::SRGBA;
      break;

    case kCVPixelFormatType_24RGB:
      image_format = mediapipe::ImageFormat::SRGB;
      break;

    case kCVPixelFormatType_OneComponent8:
      image_format = mediapipe::ImageFormat::GRAY8;
      break;

    default: {
      char format_str[5] = {static_cast<char>(pixel_format >> 24 & 0xFF),
                            static_cast<char>(pixel_format >> 16 & 0xFF),
                            static_cast<char>(pixel_format >> 8 & 0xFF),
                            static_cast<char>(pixel_format & 0xFF), 0};
      LOG(FATAL) << "unsupported pixel format: " << format_str;
    } break;
  }

  if (frame) {
    // We have already created a new frame that does not reference the buffer.
    status = CVPixelBufferUnlockBaseAddress(image_buffer,
                                            kCVPixelBufferLock_ReadOnly);
    CHECK_EQ(status, kCVReturnSuccess)
        << "CVPixelBufferUnlockBaseAddress failed: " << status;
    CVPixelBufferRelease(image_buffer);
  } else {
    frame = absl::make_unique<mediapipe::ImageFrame>(
        image_format, width, height, bytes_per_row,
        reinterpret_cast<uint8*>(base_address), [image_buffer](uint8* x) {
          CVPixelBufferUnlockBaseAddress(image_buffer,
                                         kCVPixelBufferLock_ReadOnly);
          CVPixelBufferRelease(image_buffer);
        });
  }
  return frame;
}

CFDictionaryRef GetCVPixelBufferAttributesForGlCompatibility() {
  static CFDictionaryRef attrs = NULL;
  if (!attrs) {
    CFDictionaryRef empty_dict = CFDictionaryCreate(
        kCFAllocatorDefault, NULL, NULL, 0, &kCFTypeDictionaryKeyCallBacks,
        &kCFTypeDictionaryValueCallBacks);

    // To ensure compatibility with CVOpenGLESTextureCache, these attributes
    // should be present. However, on simulator this IOSurface attribute
    // actually causes CVOpenGLESTextureCache to fail. b/144850076
    const void* keys[] = {
#if !TARGET_IPHONE_SIMULATOR
      kCVPixelBufferIOSurfacePropertiesKey,
#endif  // !TARGET_IPHONE_SIMULATOR

#if TARGET_OS_OSX
      kCVPixelFormatOpenGLCompatibility,
#else
      kCVPixelFormatOpenGLESCompatibility,
#endif  // TARGET_OS_OSX
    };

    const void* values[] = {
#if !TARGET_IPHONE_SIMULATOR
      empty_dict,
#endif  // !TARGET_IPHONE_SIMULATOR
      kCFBooleanTrue
    };

    attrs = CFDictionaryCreate(
        kCFAllocatorDefault, keys, values, ABSL_ARRAYSIZE(values),
        &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
    CFRelease(empty_dict);
  }
  return attrs;
}

void DumpCVPixelFormats() {
  CFArrayRef pf_descs =
      CVPixelFormatDescriptionArrayCreateWithAllPixelFormatTypes(
          kCFAllocatorDefault);
  CFIndex count = CFArrayGetCount(pf_descs);
  CFIndex i;

  printf("Core Video Supported Pixel Format Types:\n");

  for (i = 0; i < count; i++) {
    CFNumberRef pf_num = (CFNumberRef)CFArrayGetValueAtIndex(pf_descs, i);
    if (!pf_num) continue;

    int pf;
    CFNumberGetValue(pf_num, kCFNumberSInt32Type, &pf);

    if (pf <= 0x28) {
      printf("\nCore Video Pixel Format Type: %d\n", pf);
    } else {
      printf("\nCore Video Pixel Format Type (FourCC): %c%c%c%c\n",
             static_cast<char>(pf >> 24), static_cast<char>(pf >> 16),
             static_cast<char>(pf >> 8), static_cast<char>(pf));
    }

    CFDictionaryRef desc = CVPixelFormatDescriptionCreateWithPixelFormatType(
        kCFAllocatorDefault, pf);
    CFDictionaryApplyFunction(
        desc,
        [](const void* key, const void* value, void* context) {
          CFStringRef s = CFStringCreateWithFormat(
              kCFAllocatorDefault, nullptr, CFSTR("  %@: %@"), key, value);
          CFShow(s);
          CFRelease(s);
        },
        nullptr);
    CFRelease(desc);
  }
  CFRelease(pf_descs);
}
