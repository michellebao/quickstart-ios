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

#import "ViewController.h"
#import "UIImage+VisionDetection.h"
#import "UIUtilities.h"
@import FirebaseMLVision;
@import FirebaseMLVisionAutoML;
@import FirebaseMLCommon;
@import FirebaseMLVisionObjectDetection;

NS_ASSUME_NONNULL_BEGIN

static NSArray *images;
static NSString *const ModelExtension = @"tflite";
static NSString *const localModelName = @"mobilenet";
static NSString *const quantizedModelFilename = @"mobilenet_quant_v1_224";

static NSString *const detectionNoResultsMessage = @"No results returned.";
static NSString *const failedToDetectObjectsMessage = @"Failed to detect objects in image.";
static NSString *const sparseTextModelName = @"Sparse";
static NSString *const denseTextModelName = @"Dense";

/** Name of the local AutoML model. */
static NSString *const FIRLocalAutoMLModelName = @"local_automl_model";

/** Name of the remote AutoML model. */
static NSString *const FIRRemoteAutoMLModelName = @"remote_automl_model";

/** Filename of AutoML local model manifest in the main resource bundle. */
static NSString *const FIRAutoMLLocalModelManifestFilename = @"automl_labeler_manifest";

/** File type of AutoML local model manifest in the main resource bundle. */
static NSString *const FIRAutoMLManifestFileType = @"json";

static float const labelConfidenceThreshold = 0.75;
static CGFloat const smallDotRadius = 5.0;
static CGFloat const largeDotRadius = 10.0;
static CGColorRef lineColor;
static CGColorRef fillColor;

static int const rowsCount = 14;
static int const componentsCount = 1;

/**
 * @enum DetectorPickerRow
 * Defines the Firebase ML SDK vision detector types.
 */
typedef NS_ENUM(NSInteger, DetectorPickerRow) {
  /** On-Device vision face vision detector. */
  DetectorPickerRowDetectFaceOnDevice,
};

@interface ViewController () <UINavigationControllerDelegate, UIPickerViewDelegate, UIPickerViewDataSource, UIImagePickerControllerDelegate>

@property(nonatomic) FIRVision *vision;

@property(nonatomic) FIRModelManager *modelManager;

/** Whether the AutoML model(s) are registered. */
@property(nonatomic) BOOL areAutoMLModelsRegistered;


/** A string holding current results from detection. */
@property(nonatomic) NSMutableString *resultsText;

/** An overlay view that displays detection annotations. */
@property(nonatomic) UIView *annotationOverlayView;

/** An image picker for accessing the photo library or camera. */
@property(nonatomic) UIImagePickerController *imagePicker;
@property (weak, nonatomic) IBOutlet UIBarButtonItem *detectButton;
@property (strong, nonatomic) IBOutlet UIProgressView *downloadProgressView;

// Image counter.
@property(nonatomic) NSUInteger currentImage;

@property (weak, nonatomic) IBOutlet UIPickerView *detectorPicker;
@property (weak, nonatomic) IBOutlet UIImageView *imageView;
@property (weak, nonatomic) IBOutlet UIBarButtonItem *photoCameraButton;
@property (weak, nonatomic) IBOutlet UIBarButtonItem *videoCameraButton;

@end

@implementation ViewController

- (NSString *)stringForDetectorPickerRow:(DetectorPickerRow)detectorPickerRow {
  switch (detectorPickerRow) {
    case DetectorPickerRowDetectFaceOnDevice:
      return @"Face On-Device";
  }
}

- (void)viewDidLoad {
  [super viewDidLoad];

  images = @["@uber.jpg", @"grace_hopper.jpg", @"barcode_128.png", @"qr_code.jpg", @"beach.jpg", @"image_has_text.jpg", @"liberty.jpg"];
  lineColor = UIColor.yellowColor.CGColor;
  fillColor = UIColor.clearColor.CGColor;

  // [START init_vision]
  self.vision = [FIRVision vision];
  // [END init_vision]

  _modelManager = [FIRModelManager modelManager];

  self.imagePicker = [UIImagePickerController new];
  self.resultsText = [NSMutableString new];
  _currentImage = 0;
  _imageView.image = [UIImage imageNamed: images[_currentImage]];
  _annotationOverlayView = [[UIView alloc] initWithFrame:CGRectZero];
  _annotationOverlayView.translatesAutoresizingMaskIntoConstraints = NO;
  [_imageView addSubview:_annotationOverlayView];
  [NSLayoutConstraint activateConstraints:@[
                                            [_annotationOverlayView.topAnchor constraintEqualToAnchor:_imageView.topAnchor],
                                            [_annotationOverlayView.leadingAnchor constraintEqualToAnchor:_imageView.leadingAnchor],
                                            [_annotationOverlayView.trailingAnchor constraintEqualToAnchor:_imageView.trailingAnchor],
                                            [_annotationOverlayView.bottomAnchor constraintEqualToAnchor:_imageView.bottomAnchor]
                                            ]];
  _imagePicker.delegate = self;
  _imagePicker.sourceType = UIImagePickerControllerSourceTypePhotoLibrary;

  _detectorPicker.delegate = self;
  _detectorPicker.dataSource = self;

  BOOL isCameraAvailable = [UIImagePickerController isCameraDeviceAvailable:UIImagePickerControllerCameraDeviceFront] ||
                           [UIImagePickerController isCameraDeviceAvailable:UIImagePickerControllerCameraDeviceRear];
  if (isCameraAvailable) {
    // `CameraViewController` uses `AVCaptureDeviceDiscoverySession` which is only supported for
    // iOS 10 or newer.
    if (@available(iOS 10, *)) {
      [_videoCameraButton setEnabled:YES];
    }
  } else {
    [_photoCameraButton setEnabled:NO];
  }

  int defaultRow = (rowsCount / 2) - 1;
  [_detectorPicker selectRow:defaultRow inComponent:0 animated:NO];
}

- (void)viewWillAppear:(BOOL)animated {
  [super viewWillAppear:animated];
  [self.navigationController.navigationBar setHidden:YES];
}

- (void)viewWillDisappear:(BOOL)animated {
  [super viewWillDisappear:animated];
  [self.navigationController.navigationBar setHidden:NO];
}

- (IBAction)detect:(id)sender {
  [self clearResults];
  NSInteger rowIndex = [_detectorPicker selectedRowInComponent:0];
  switch (rowIndex) {
    case DetectorPickerRowDetectFaceOnDevice:
      [self detectFacesInImage:_imageView.image];
      break;
  }
}

- (IBAction)openPhotoLibrary:(id)sender {
  _imagePicker.sourceType = UIImagePickerControllerSourceTypePhotoLibrary;
  [self presentViewController:_imagePicker animated:YES completion:nil];
}

- (IBAction)openCamera:(id)sender {
  if (![UIImagePickerController isCameraDeviceAvailable:UIImagePickerControllerCameraDeviceFront] && ![UIImagePickerController isCameraDeviceAvailable:UIImagePickerControllerCameraDeviceRear]) {
    return;
  }
  _imagePicker.sourceType = UIImagePickerControllerSourceTypeCamera;
  [self presentViewController:_imagePicker animated:YES completion:nil];
}
- (IBAction)changeImage:(id)sender {
  [self clearResults];
  self.currentImage = (_currentImage + 1) % images.count;
  _imageView.image = [UIImage imageNamed:images[_currentImage]];
}

/// Removes the detection annotations from the annotation overlay view.
- (void)removeDetectionAnnotations {
  for (UIView *annotationView in _annotationOverlayView.subviews) {
    [annotationView removeFromSuperview];
  }
}

/// Clears the results text view and removes any frames that are visible.
- (void)clearResults {
  [self removeDetectionAnnotations];
  self.resultsText = [NSMutableString new];
}

- (void)showResults {
  UIAlertController *resultsAlertController = [UIAlertController alertControllerWithTitle:@"Detection Results" message:nil preferredStyle:UIAlertControllerStyleActionSheet];
  [resultsAlertController addAction:[UIAlertAction actionWithTitle:@"OK" style:UIAlertActionStyleDestructive handler:^(UIAlertAction * _Nonnull action) {
    [resultsAlertController dismissViewControllerAnimated:YES completion:nil];
  }]];
  resultsAlertController.message = _resultsText;
  resultsAlertController.popoverPresentationController.barButtonItem = _detectButton;
  resultsAlertController.popoverPresentationController.sourceView = self.view;
  [self presentViewController:resultsAlertController animated:YES completion:nil];
  NSLog(@"%@", _resultsText);
}

/// Updates the image view with a scaled version of the given image.
- (void)updateImageViewWithImage:(UIImage *)image {
  CGFloat scaledImageWidth = 0.0;
  CGFloat scaledImageHeight = 0.0;
  switch (UIApplication.sharedApplication.statusBarOrientation) {
    case UIInterfaceOrientationPortrait:
    case UIInterfaceOrientationPortraitUpsideDown:
    case UIInterfaceOrientationUnknown:
      scaledImageWidth = _imageView.bounds.size.width;
      scaledImageHeight = image.size.height * scaledImageWidth / image.size.width;
      break;
    case UIInterfaceOrientationLandscapeLeft:
    case UIInterfaceOrientationLandscapeRight:
      scaledImageWidth = image.size.width * scaledImageHeight / image.size.height;
      scaledImageHeight = _imageView.bounds.size.height;
      break;
  }

  dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
    // Scale image while maintaining aspect ratio so it displays better in the UIImageView.
    UIImage *scaledImage = [image scaledImageWithSize:CGSizeMake(scaledImageWidth, scaledImageHeight)];
    if (!scaledImage) {
      scaledImage = image;
    }
    if (!scaledImage) {
      return;
    }
    dispatch_async(dispatch_get_main_queue(), ^{
      self->_imageView.image = scaledImage;
    });
  });
}

- (CGAffineTransform)transformMatrix {
  UIImage *image = _imageView.image;
  if (!image) {
    return CGAffineTransformMake(0, 0, 0, 0, 0, 0);
  }
  CGFloat imageViewWidth = _imageView.frame.size.width;
  CGFloat imageViewHeight = _imageView.frame.size.height;
  CGFloat imageWidth = image.size.width;
  CGFloat imageHeight = image.size.height;

  CGFloat imageViewAspectRatio = imageViewWidth / imageViewHeight;
  CGFloat imageAspectRatio = imageWidth / imageHeight;
  CGFloat scale = (imageViewAspectRatio > imageAspectRatio) ?
      imageViewHeight / imageHeight :
      imageViewWidth / imageWidth;

  // Image view's `contentMode` is `scaleAspectFit`, which scales the image to fit the size of the
  // image view by maintaining the aspect ratio. Multiple by `scale` to get image's original size.
  CGFloat scaledImageWidth = imageWidth * scale;
  CGFloat scaledImageHeight = imageHeight * scale;
  CGFloat xValue = (imageViewWidth - scaledImageWidth) / 2.0;
  CGFloat yValue = (imageViewHeight - scaledImageHeight) / 2.0;

  CGAffineTransform transform = CGAffineTransformTranslate(CGAffineTransformIdentity, xValue, yValue);
  return CGAffineTransformScale(transform, scale, scale);
}

- (CGPoint)pointFromVisionPoint:(FIRVisionPoint *)visionPoint {
  return CGPointMake(visionPoint.x.floatValue, visionPoint.y.floatValue);
}

- (void)process:(FIRVisionImage *)visionImage withTextRecognizer:(FIRVisionTextRecognizer *)textRecognizer {
  // [START recognize_text]
  [textRecognizer processImage:visionImage completion:^(FIRVisionText * _Nullable text, NSError * _Nullable error) {
    if (text == nil) {
      // [START_EXCLUDE]
      self.resultsText = [NSMutableString stringWithFormat:@"Text recognizer failed with error: %@", error ? error.localizedDescription : detectionNoResultsMessage];
      [self showResults];
      // [END_EXCLUDE]
      return;
    }

    // [START_EXCLUDE]
    // Blocks.
    for (FIRVisionTextBlock *block in text.blocks) {
      CGRect transformedRect = CGRectApplyAffineTransform(block.frame, [self transformMatrix]);
      [UIUtilities addRectangle:transformedRect toView:self.annotationOverlayView color:UIColor.purpleColor];

      // Lines.
      for (FIRVisionTextLine *line in block.lines) {
        CGRect transformedRect = CGRectApplyAffineTransform(line.frame, [self transformMatrix]);
        [UIUtilities addRectangle:transformedRect toView:self.annotationOverlayView color:UIColor.orangeColor];

        // Elements.
        for (FIRVisionTextElement *element in line.elements) {
          CGRect transformedRect = CGRectApplyAffineTransform(element.frame, [self transformMatrix]);
          [UIUtilities addRectangle:transformedRect toView:self.annotationOverlayView color:UIColor.greenColor];
          UILabel *label = [[UILabel alloc] initWithFrame:transformedRect];
          label.text = element.text;
          label.adjustsFontSizeToFitWidth = YES;
          [self.annotationOverlayView addSubview:label];
        }
      }
    }
    [self.resultsText appendFormat:@"%@\n", text.text];
    [self showResults];
    // [END_EXCLUDE]
  }];
  // [END recognize_text]
}

- (void)process:(FIRVisionImage *)visionImage withDocumentTextRecognizer:(FIRVisionDocumentTextRecognizer *)documentTextRecognizer {
  // [START recognize_document_text]
  [documentTextRecognizer processImage:visionImage completion:^(FIRVisionDocumentText * _Nullable text, NSError * _Nullable error) {
    if (text == nil) {
      // [START_EXCLUDE]
      self.resultsText = [NSMutableString stringWithFormat:@"Document text recognizer failed with error: %@", error ? error.localizedDescription : detectionNoResultsMessage];
      [self showResults];
      // [END_EXCLUDE]
      return;
    }
    // [START_EXCLUDE]
    // Blocks.
    for (FIRVisionDocumentTextBlock *block in text.blocks) {
      CGRect transformedRect = CGRectApplyAffineTransform(block.frame, [self transformMatrix]);
      [UIUtilities addRectangle:transformedRect toView:self.annotationOverlayView color:UIColor.purpleColor];

      // Paragraphs.
      for (FIRVisionDocumentTextParagraph *paragraph in block.paragraphs) {
        CGRect transformedRect = CGRectApplyAffineTransform(paragraph.frame, [self transformMatrix]);
        [UIUtilities addRectangle:transformedRect toView:self.annotationOverlayView color:UIColor.orangeColor];

        // Words.
        for (FIRVisionDocumentTextWord *word in paragraph.words) {
          CGRect transformedRect = CGRectApplyAffineTransform(word.frame, [self transformMatrix]);
          [UIUtilities addRectangle:transformedRect toView:self.annotationOverlayView color:UIColor.greenColor];

          // Symbols.
          for (FIRVisionDocumentTextSymbol *symbol in word.symbols) {
            CGRect transformedRect = CGRectApplyAffineTransform(symbol.frame, [self transformMatrix]);
            [UIUtilities addRectangle:transformedRect toView:self.annotationOverlayView color:UIColor.cyanColor];
            UILabel *label = [[UILabel alloc] initWithFrame:transformedRect];
            label.text = symbol.text;
            label.adjustsFontSizeToFitWidth = YES;
            [self.annotationOverlayView addSubview:label];
          }
        }
      }
    }
    [self.resultsText appendFormat:@"%@\n", text.text];
    [self showResults];
    // [END_EXCLUDE]
  }];
  // [END recognize_document_text]
}

#pragma mark - UIPickerViewDataSource

- (NSInteger)numberOfComponentsInPickerView:(nonnull UIPickerView *)pickerView {
  return componentsCount;
}

- (NSInteger)pickerView:(nonnull UIPickerView *)pickerView numberOfRowsInComponent:(NSInteger)component {
  return rowsCount;
}

#pragma mark - UIPickerViewDelegate

- (nullable NSString *)pickerView:(UIPickerView *)pickerView titleForRow:(NSInteger)row forComponent:(NSInteger)component {
  return [self stringForDetectorPickerRow:row];
}

- (void)pickerView:(UIPickerView *)pickerView didSelectRow:(NSInteger)row inComponent:(NSInteger)component {
  [self clearResults];
}


#pragma mark - UIImagePickerControllerDelegate

- (void)imagePickerController:(UIImagePickerController *)picker didFinishPickingMediaWithInfo:(NSDictionary<NSString *,id> *)info {
  [self clearResults];
  UIImage *pickedImage = info[UIImagePickerControllerOriginalImage];
  if (pickedImage) {
    [self updateImageViewWithImage:pickedImage];
  }
  [self dismissViewControllerAnimated:YES completion:nil];
}

#pragma mark - Vision On-Device Detection

/// Detects faces on the specified image and draws a frame around the detected faces using
/// On-Device face API.
///
/// - Parameter image: The image.
- (void)detectFacesInImage:(UIImage *)image {
  if (!image) {
    return;
  }

  // Create a face detector with options.
  // [START config_face]
  FIRVisionFaceDetectorOptions *options = [FIRVisionFaceDetectorOptions new];
  options.landmarkMode = FIRVisionFaceDetectorLandmarkModeAll;
  options.contourMode = FIRVisionFaceDetectorContourModeAll;
  options.classificationMode = FIRVisionFaceDetectorClassificationModeAll;
  options.performanceMode = FIRVisionFaceDetectorPerformanceModeAccurate;
  // [END config_face]

  // [START init_face]
  FIRVisionFaceDetector *faceDetector = [_vision faceDetectorWithOptions:options];
  // [END init_face]

  // Define the metadata for the image.
  FIRVisionImageMetadata *imageMetadata = [FIRVisionImageMetadata new];
  imageMetadata.orientation = [UIUtilities visionImageOrientationFromImageOrientation:image.imageOrientation];

  // Initialize a VisionImage object with the given UIImage.
  FIRVisionImage *visionImage = [[FIRVisionImage alloc] initWithImage:image];
  visionImage.metadata = imageMetadata;

  // [START detect_faces]
  [faceDetector processImage:visionImage completion:^(NSArray<FIRVisionFace *> * _Nullable faces, NSError * _Nullable error) {
    if (!faces || faces.count == 0) {
      // [START_EXCLUDE]
      NSString *errorString = error ? error.localizedDescription : detectionNoResultsMessage;
      self.resultsText = [NSMutableString stringWithFormat:@"On-Device face detection failed with error: %@", errorString];
      [self showResults];
      // [END_EXCLUDE]
      return;
    }

    // Faces detected
    // [START_EXCLUDE]
    [self.resultsText setString:@""];
    for (FIRVisionFace *face in faces) {
      CGAffineTransform transform = [self transformMatrix];
      CGRect transformedRect = CGRectApplyAffineTransform(face.frame, transform);
      [UIUtilities addRectangle:transformedRect toView:self.annotationOverlayView color:UIColor.greenColor];
      [self.resultsText appendFormat:@"Frame: %@\n", NSStringFromCGRect(face.frame)];
      NSString *headEulerAngleY = face.hasHeadEulerAngleY ? [NSString stringWithFormat: @"%.2f", face.headEulerAngleY] : @"NA";
      NSString *headEulerAngleZ = face.hasHeadEulerAngleZ ? [NSString stringWithFormat: @"%.2f", face.headEulerAngleZ] : @"NA";
      NSString *leftEyeOpenProbability = face.hasLeftEyeOpenProbability ? [NSString stringWithFormat: @"%.2f", face.leftEyeOpenProbability] : @"NA";
      NSString *rightEyeOpenProbability = face.hasRightEyeOpenProbability ? [NSString stringWithFormat: @"%.2f", face.rightEyeOpenProbability] : @"NA";
      NSString *smilingProbability = face.hasSmilingProbability ? [NSString stringWithFormat: @"%.2f", face.smilingProbability] : @"NA";
      [self.resultsText appendFormat:@"Head Euler Angle Y: %@\n", headEulerAngleY];
      [self.resultsText appendFormat:@"Head Euler Angle Z: %@\n", headEulerAngleZ];
      [self.resultsText appendFormat:@"Left Eye Open Probability: %@\n", leftEyeOpenProbability];
      [self.resultsText appendFormat:@"Right Eye Open Probability: %@\n", rightEyeOpenProbability];
      [self.resultsText appendFormat:@"Smiling Probability: %@\n", smilingProbability];
    }
    [self showResults];
    // [END_EXCLUDE]
  }];
  // [END detect_faces]
}

#pragma mark - Notifications

- (void)remoteModelDownloadDidSucceed:(NSNotification *)notification {
  dispatch_async(dispatch_get_main_queue(), ^{
    self.downloadProgressView.hidden = YES;
    FIRRemoteModel *remotemodel = notification.userInfo[FIRModelDownloadUserInfoKeyRemoteModel];
    if (remotemodel == nil) {
      [self.resultsText appendString:@"firebaseMLModelDownloadDidSucceed notification posted without a RemoteModel instance."];
      return;
    }
    [self.resultsText appendFormat:@"Successfully downloaded the remote model with name: %@. The model is ready for detection.", remotemodel.name];
  });
}

- (void)remoteModelDownloadDidFail:(NSNotification *)notification {
  dispatch_async(dispatch_get_main_queue(), ^{
    self.downloadProgressView.hidden = YES;
    FIRRemoteModel *remoteModel = notification.userInfo[FIRModelDownloadUserInfoKeyRemoteModel];
    NSError *error = notification.userInfo[FIRModelDownloadUserInfoKeyError];
    if (error == nil) {
      [self.resultsText appendString:@"firebaseMLModelDownloadDidFail notification posted without a RemoteModel instance or error."];
      return;
    }
    [self.resultsText appendFormat:@"Failed to download the remote model with name: %@, error: %@.", remoteModel, error.localizedDescription];
  });
}

@end

NS_ASSUME_NONNULL_END
