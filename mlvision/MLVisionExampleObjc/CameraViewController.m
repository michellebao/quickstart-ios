//
//  Copyright (c) 2018 Google Inc.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//

#import "CameraViewController.h"
#import "UIUtilities.h"
@import AVFoundation;
@import CoreVideo;

@import FirebaseMLVision;
@import FirebaseMLVisionObjectDetection;
@import FirebaseMLCommon;
@import FirebaseMLVisionAutoML;

NS_ASSUME_NONNULL_BEGIN

static NSString *const alertControllerTitle = @"Vision Detectors";
static NSString *const alertControllerMessage = @"Select a detector";
static NSString *const cancelActionTitleText = @"Cancel";
static NSString *const videoDataOutputQueueLabel = @"com.google.firebaseml.visiondetector.VideoDataOutputQueue";
static NSString *const sessionQueueLabel = @"com.google.firebaseml.visiondetector.SessionQueue";
static NSString *const noResultsMessage = @"No Results";

/** Name of the local AutoML model. */
static NSString *const FIRLocalAutoMLModelName = @"local_automl_model";

/** Name of the remote AutoML model. */
static NSString *const FIRRemoteAutoMLModelName = @"remote_automl_model";

/** Filename of AutoML local model manifest in the main resource bundle. */
static NSString *const FIRAutoMLLocalModelManifestFilename = @"automl_labeler_manifest";

/** File type of AutoML local model manifest in the main resource bundle. */
static NSString *const FIRAutoMLManifestFileType = @"json";

static float const labelConfidenceThreshold = 0.75;
static const CGFloat FIRSmallDotRadius = 4.0;
static const CGFloat FIRconstantScale = 1.0;
static const CGFloat padding = 10.0;
static const CGFloat resultsLabelHeight = 200.0;
static const int resultsLabelLines = 5;


@interface CameraViewController () <AVCaptureVideoDataOutputSampleBufferDelegate>

typedef NS_ENUM(NSInteger, Detector) {
  DetectorOnDeviceFace,
};

@property (nonatomic) NSArray *detectors;
@property (nonatomic) Detector currentDetector;
@property (nonatomic) bool isUsingFrontCamera;
@property (nonatomic, nonnull) AVCaptureVideoPreviewLayer *previewLayer;
@property (nonatomic) AVCaptureSession *captureSession;
@property (nonatomic) dispatch_queue_t sessionQueue;
@property (nonatomic) FIRVision *vision;
@property (nonatomic) UIView *annotationOverlayView;
@property (nonatomic) UIImageView *previewOverlayView;
@property (weak, nonatomic) IBOutlet UIView *cameraView;
@property (nonatomic) CMSampleBufferRef lastFrame;
@property(nonatomic) FIRModelManager *modelManager;

/** Whether the AutoML model(s) are registered. */
@property(nonatomic) BOOL areAutoMLModelsRegistered;
@property (strong, nonatomic) IBOutlet UIProgressView *downloadProgressView;


@end

@implementation CameraViewController

- (NSString *)stringForDetector:(Detector)detector {
  switch (detector) {
    case DetectorOnDeviceFace:
      return @"On-Device Face Detection";
  }
}

- (void)viewDidLoad {
  [super viewDidLoad];
  _detectors = @[@(DetectorOnDeviceFace),
                 ];
  _currentDetector = DetectorOnDeviceFace;
  _isUsingFrontCamera = YES;
  _captureSession = [[AVCaptureSession alloc] init];
  _sessionQueue = dispatch_queue_create(sessionQueueLabel.UTF8String, nil);
  _vision = [FIRVision vision];
  _modelManager = [FIRModelManager modelManager];
  _previewOverlayView = [[UIImageView alloc] initWithFrame:CGRectZero];
  _previewOverlayView.contentMode = UIViewContentModeScaleAspectFill;
  _previewOverlayView.translatesAutoresizingMaskIntoConstraints = NO;
  _annotationOverlayView = [[UIView alloc] initWithFrame:CGRectZero];
  _annotationOverlayView.translatesAutoresizingMaskIntoConstraints = NO;

  self.previewLayer = [AVCaptureVideoPreviewLayer layerWithSession:_captureSession];
  [self setUpPreviewOverlayView];
  [self setUpAnnotationOverlayView];
  [self setUpCaptureSessionOutput];
  [self setUpCaptureSessionInput];
}

- (void)viewDidAppear:(BOOL)animated {
  [super viewDidAppear:animated];
  [self startSession];
}

- (void)viewDidDisappear:(BOOL)animated {
  [super viewDidDisappear:animated];
  [self stopSession];
}

- (void)viewDidLayoutSubviews {
  [super viewDidLayoutSubviews];
  _previewLayer.frame = _cameraView.frame;
}

- (IBAction)selectDetector:(id)sender {
  [self presentDetectorsAlertController];
}

- (IBAction)switchCamera:(id)sender {
  self.isUsingFrontCamera = !_isUsingFrontCamera;
  [self removeDetectionAnnotations];
  [self setUpCaptureSessionInput];
}

#pragma mark - Notifications

- (void)remoteModelDownloadDidSucceed:(NSNotification *)notification {
  dispatch_async(dispatch_get_main_queue(), ^{
    self.downloadProgressView.hidden = YES;
    FIRRemoteModel *remotemodel = notification.userInfo[FIRModelDownloadUserInfoKeyRemoteModel];
    if (remotemodel == nil) {
      NSLog(@"firebaseMLModelDownloadDidSucceed notification posted without a RemoteModel instance.");
      return;
    }
    NSLog(@"Successfully downloaded the remote model with name: %@. The model is ready for detection.", remotemodel.name);
  });
}

- (void)remoteModelDownloadDidFail:(NSNotification *)notification {
  dispatch_async(dispatch_get_main_queue(), ^{
    self.downloadProgressView.hidden = YES;
    FIRRemoteModel *remoteModel = notification.userInfo[FIRModelDownloadUserInfoKeyRemoteModel];
    NSError *error = notification.userInfo[FIRModelDownloadUserInfoKeyError];
    if (error == nil) {
      NSLog(@"firebaseMLModelDownloadDidFail notification posted without a RemoteModel instance or error.");
      return;
    }
    NSLog(@"Failed to download the remote model with name: %@, error: %@.", remoteModel, error.localizedDescription);
  });
}

#pragma mark - Other On-Device Detections

- (void)detectFacesOnDeviceInImage:(FIRVisionImage *)image width:(CGFloat) width height:(CGFloat)height {
  FIRVisionFaceDetectorOptions *options = [[FIRVisionFaceDetectorOptions alloc] init];
  // When performing latency tests to determine ideal detection settings,
  // run the app in 'release' mode to get accurate performance metrics
  options.performanceMode = FIRVisionFaceDetectorPerformanceModeFast;
  options.contourMode = FIRVisionFaceDetectorContourModeAll;
  options.landmarkMode = FIRVisionFaceDetectorLandmarkModeNone;
  options.classificationMode = FIRVisionFaceDetectorClassificationModeNone;

  FIRVisionFaceDetector *faceDetector = [_vision faceDetectorWithOptions:options];
  NSError *error;
  NSArray<FIRVisionFace *> *faces = [faceDetector resultsInImage:image error:&error];
  if (error != nil) {
    NSLog(@"Failed to detect faces with error: %@", error.localizedDescription);
    return;
  }
  if (faces.count == 0) {
    NSLog(@"%@", @"On-Device face detector returned no results.");
    dispatch_sync(dispatch_get_main_queue(), ^{
      [self updatePreviewOverlayView];
      [self removeDetectionAnnotations];
    });
    return;
  }

  dispatch_sync(dispatch_get_main_queue(), ^{
    [self updatePreviewOverlayView];
    [self removeDetectionAnnotations];
    for (FIRVisionFace *face in faces) {
      CGRect normalizedRect = CGRectMake(face.frame.origin.x / width, face.frame.origin.y / height, face.frame.size.width / width, face.frame.size.height / height);
      CGRect standardizedRect = CGRectStandardize([self->_previewLayer rectForMetadataOutputRectOfInterest:normalizedRect]);
      [UIUtilities addRectangle:standardizedRect toView:self->_annotationOverlayView color:UIColor.greenColor];
    }
  });
}

#pragma mark - Private

- (void)setUpCaptureSessionOutput {
  dispatch_async(_sessionQueue, ^{
    [self->_captureSession beginConfiguration];
    // When performing latency tests to determine ideal capture settings,
    // run the app in 'release' mode to get accurate performance metrics
    self->_captureSession.sessionPreset = AVCaptureSessionPresetMedium;

    AVCaptureVideoDataOutput *output = [[AVCaptureVideoDataOutput alloc] init];
    output.videoSettings = @{(id)kCVPixelBufferPixelFormatTypeKey: [NSNumber numberWithUnsignedInt:kCVPixelFormatType_32BGRA]};
    dispatch_queue_t outputQueue = dispatch_queue_create(videoDataOutputQueueLabel.UTF8String, nil);
    [output setSampleBufferDelegate:self queue:outputQueue];
    if ([self.captureSession canAddOutput:output]) {
      [self.captureSession addOutput:output];
      [self.captureSession commitConfiguration];
    } else {
      NSLog(@"%@", @"Failed to add capture session output.");
    }
  });
}

- (void)setUpCaptureSessionInput {
  dispatch_async(_sessionQueue, ^{
    AVCaptureDevicePosition cameraPosition = self.isUsingFrontCamera ? AVCaptureDevicePositionFront : AVCaptureDevicePositionBack;
    AVCaptureDevice *device = [self captureDeviceForPosition:cameraPosition];
    if (device) {
      [self->_captureSession beginConfiguration];
      NSArray<AVCaptureInput *> *currentInputs = self.captureSession.inputs;
      for (AVCaptureInput *input in currentInputs) {
        [self.captureSession removeInput:input];
      }
      NSError *error;
      AVCaptureDeviceInput *input = [AVCaptureDeviceInput deviceInputWithDevice:device error:&error];
      if (error) {
        NSLog(@"Failed to create capture device input: %@", error.localizedDescription);
        return;
      } else {
        if ([self.captureSession canAddInput:input]) {
          [self.captureSession addInput:input];
        } else {
          NSLog(@"%@", @"Failed to add capture session input.");
        }
      }
      [self.captureSession commitConfiguration];
    } else {
      NSLog(@"Failed to get capture device for camera position: %ld", cameraPosition);
    }
  });
}

- (void)startSession {
  dispatch_async(_sessionQueue, ^{
    [self->_captureSession startRunning];
  });
}

- (void)stopSession {
  dispatch_async(_sessionQueue, ^{
    [self->_captureSession stopRunning];
  });
}

- (void)setUpPreviewOverlayView {
  [_cameraView addSubview:_previewOverlayView];
  [NSLayoutConstraint activateConstraints:@[
                                            [_previewOverlayView.centerYAnchor constraintEqualToAnchor:_cameraView.centerYAnchor],
                                            [_previewOverlayView.centerXAnchor constraintEqualToAnchor:_cameraView.centerXAnchor],
                                            [_previewOverlayView.leadingAnchor constraintEqualToAnchor:_cameraView.leadingAnchor],
                                            [_previewOverlayView.trailingAnchor constraintEqualToAnchor:_cameraView.trailingAnchor]
                                            ]];
}
- (void)setUpAnnotationOverlayView {
  [_cameraView addSubview:_annotationOverlayView];
  [NSLayoutConstraint activateConstraints:@[
                                            [_annotationOverlayView.topAnchor constraintEqualToAnchor:_cameraView.topAnchor],
                                            [_annotationOverlayView.leadingAnchor constraintEqualToAnchor:_cameraView.leadingAnchor],
                                            [_annotationOverlayView.trailingAnchor constraintEqualToAnchor:_cameraView.trailingAnchor],
                                            [_annotationOverlayView.bottomAnchor constraintEqualToAnchor:_cameraView.bottomAnchor]
                                            ]];
}

- (AVCaptureDevice *)captureDeviceForPosition:(AVCaptureDevicePosition)position  {
  if (@available(iOS 10, *)) {
    AVCaptureDeviceDiscoverySession *discoverySession =
      [AVCaptureDeviceDiscoverySession discoverySessionWithDeviceTypes:@[AVCaptureDeviceTypeBuiltInWideAngleCamera]
                                                             mediaType:AVMediaTypeVideo
                                                             position:AVCaptureDevicePositionUnspecified];
    for (AVCaptureDevice *device in discoverySession.devices) {
      if (device.position == position) {
        return device;
      }
    }
  }
  return nil;
}

- (void)presentDetectorsAlertController {
  UIAlertController *alertController = [UIAlertController alertControllerWithTitle:alertControllerTitle message:alertControllerMessage preferredStyle:UIAlertControllerStyleAlert];
  for (NSNumber *detectorType in _detectors) {
    NSInteger detector = detectorType.integerValue;
    UIAlertAction *action = [UIAlertAction actionWithTitle:[self stringForDetector:detector]
                                                     style:UIAlertActionStyleDefault handler:^(UIAlertAction * _Nonnull action) {
                                                       self.currentDetector = detector;
                                                       [self removeDetectionAnnotations];
                                                     }];
    if (detector == _currentDetector) {
      [action setEnabled:NO];
    }
    [alertController addAction:action];
  }
  [alertController addAction:[UIAlertAction actionWithTitle:cancelActionTitleText
                                                      style:UIAlertActionStyleCancel handler:nil]];
  [self presentViewController:alertController animated:YES completion:nil];
}

- (void)removeDetectionAnnotations {
  for (UIView *annotationView in _annotationOverlayView.subviews) {
    [annotationView removeFromSuperview];
  }
}

- (void)updatePreviewOverlayView {
  CVImageBufferRef imageBuffer = CMSampleBufferGetImageBuffer(_lastFrame);
  if (imageBuffer == nil) {
    return;
  }
  CIImage *ciImage = [CIImage imageWithCVPixelBuffer:imageBuffer];
  CIContext *context = [[CIContext alloc] initWithOptions:nil];
  CGImageRef cgImage = [context createCGImage:ciImage fromRect:ciImage.extent];
  if (cgImage == nil) {
    return;
  }
  UIImage *rotatedImage = [UIImage imageWithCGImage:cgImage scale:FIRconstantScale orientation:UIImageOrientationRight];
  if (_isUsingFrontCamera) {
    CGImageRef rotatedCGImage = rotatedImage.CGImage;
    if (rotatedCGImage == nil) {
      return;
    }
    UIImage *mirroredImage = [UIImage imageWithCGImage:rotatedCGImage scale:FIRconstantScale orientation:UIImageOrientationLeftMirrored];
    _previewOverlayView.image = mirroredImage;
  } else {
    _previewOverlayView.image = rotatedImage;
  }
  CGImageRelease(cgImage);
}

#pragma mark - AVCaptureVideoDataOutputSampleBufferDelegate

- (void)captureOutput:(AVCaptureOutput *)output didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer fromConnection:(AVCaptureConnection *)connection {
  CVImageBufferRef imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
  if (imageBuffer) {
    _lastFrame = sampleBuffer;
    FIRVisionImage *visionImage = [[FIRVisionImage alloc] initWithBuffer:sampleBuffer];
    FIRVisionImageMetadata *metadata = [[FIRVisionImageMetadata alloc] init];
    UIImageOrientation orientation = [UIUtilities imageOrientationFromDevicePosition:_isUsingFrontCamera ? AVCaptureDevicePositionFront : AVCaptureDevicePositionBack];

    FIRVisionDetectorImageOrientation visionOrientation = [UIUtilities visionImageOrientationFromImageOrientation:orientation];
    metadata.orientation = visionOrientation;
    visionImage.metadata = metadata;
    CGFloat imageWidth = CVPixelBufferGetWidth(imageBuffer);
    CGFloat imageHeight = CVPixelBufferGetHeight(imageBuffer);
    BOOL shouldEnableClassification = NO;
    BOOL shouldEnableMultipleObjects = NO;
    switch (_currentDetector) {
      case DetectorOnDeviceObjectProminentWithClassifier:
      case DetectorOnDeviceObjectMultipleWithClassifier:
      shouldEnableClassification = YES;
    default:
      break;
    }
    switch (_currentDetector) {
      case DetectorOnDeviceObjectMultipleNoClassifier:
      case DetectorOnDeviceObjectMultipleWithClassifier:
      shouldEnableMultipleObjects = YES;
    default:
      break;
    }

    switch (_currentDetector) {
      case DetectorOnDeviceFace:
        [self detectFacesOnDeviceInImage:visionImage width:imageWidth height:imageHeight];
        break;
    }
  } else {
    NSLog(@"%@", @"Failed to get image buffer from sample buffer.");
  }
}

@end

NS_ASSUME_NONNULL_END
