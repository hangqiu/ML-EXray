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

#import "Foundation/Foundation.h"

#include "mediapipe/framework/port/status.h"

/// Error domain for absl::Status errors.
extern NSString *const kGUSGoogleUtilStatusErrorDomain;

/// Key for the absl::Status wrapper in an NSError's user info dictionary.
extern NSString *const kGUSGoogleUtilStatusErrorKey;

/// This just wraps absl::Status into an Objective-C object.
@interface GUSUtilStatusWrapper : NSObject

@property(nonatomic) absl::Status status;

+ (instancetype)wrapStatus:(const absl::Status &)status;

@end

/// This category adds methods for generating NSError objects from absl::Status
/// objects, and vice versa.
@interface NSError (GUSGoogleUtilStatus)

/// Generates an NSError representing a absl::Status. Note that NSError always
/// represents an error, so this should not be called with absl::Status::OK.
+ (NSError *)gus_errorWithStatus:(const absl::Status &)status;

/// Returns a absl::Status object representing an NSError. If the NSError was
/// generated from a absl::Status, the absl::Status returned is identical to
/// the original. Otherwise, this returns a status with code ::util::error::UNKNOWN
/// and a message extracted from the NSError.
@property(nonatomic, readonly) absl::Status gus_status;  // NOLINT(identifier-naming)

@end
