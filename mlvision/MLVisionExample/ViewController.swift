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

import UIKit
// [START import_vision]
import FirebaseMLVision
// [END import_vision]

import FirebaseMLVisionObjectDetection
import FirebaseMLVisionAutoML
import FirebaseMLCommon

/// Main view controller class.
@objc(ViewController)
class ViewController:  UIViewController, UINavigationControllerDelegate {
    /// Firebase vision instance.
    // [START init_vision]
    lazy var vision = Vision.vision()
    // [END init_vision]
    
    /// Manager for local and remote models.
    lazy var modelManager = ModelManager.modelManager()
    
    /// Whether the AutoML models are registered.
    var areAutoMLModelsRegistered = false
    
    /// A string holding current results from detection.
    var resultsText = ""
    
    /// An overlay view that displays detection annotations.
    private lazy var annotationOverlayView: UIView = {
        precondition(isViewLoaded)
        let annotationOverlayView = UIView(frame: .zero)
        annotationOverlayView.translatesAutoresizingMaskIntoConstraints = false
        return annotationOverlayView
    }()
    
    /// An image picker for accessing the photo library or camera.
    var imagePicker = UIImagePickerController()
    
    // Image counter.
    var currentImage = 0
    
    // MARK: - IBOutlets
    
    @IBOutlet fileprivate weak var detectorPicker: UIPickerView!
    @IBOutlet fileprivate weak var imageView: UIImageView!
    @IBOutlet fileprivate weak var photoCameraButton: UIBarButtonItem!
    @IBOutlet fileprivate weak var videoCameraButton: UIBarButtonItem!
    @IBOutlet weak var detectButton: UIBarButtonItem!
    @IBOutlet var downloadProgressView: UIProgressView!
    
    // MARK: - UIViewController
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        imageView.image = UIImage(named: Constants.images[currentImage])
        imageView.addSubview(annotationOverlayView)
        NSLayoutConstraint.activate([
            annotationOverlayView.topAnchor.constraint(equalTo: imageView.topAnchor),
            annotationOverlayView.leadingAnchor.constraint(equalTo: imageView.leadingAnchor),
            annotationOverlayView.trailingAnchor.constraint(equalTo: imageView.trailingAnchor),
            annotationOverlayView.bottomAnchor.constraint(equalTo: imageView.bottomAnchor),
            ])
        
        imagePicker.delegate = self
        imagePicker.sourceType = .photoLibrary
        
        detectorPicker.delegate = self
        detectorPicker.dataSource = self
        
        let isCameraAvailable = UIImagePickerController.isCameraDeviceAvailable(.front) ||
            UIImagePickerController.isCameraDeviceAvailable(.rear)
        if isCameraAvailable {
            // `CameraViewController` uses `AVCaptureDevice.DiscoverySession` which is only supported for
            // iOS 10 or newer.
            if #available(iOS 10.0, *) {
                videoCameraButton.isEnabled = true
            }
        } else {
            photoCameraButton.isEnabled = false
        }
        
        let defaultRow = (DetectorPickerRow.rowsCount / 2) - 1
        detectorPicker.selectRow(defaultRow, inComponent: 0, animated: false)
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        
        navigationController?.navigationBar.isHidden = true
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        
        navigationController?.navigationBar.isHidden = false
    }
    
    // MARK: - IBActions
    
    @IBAction func detect(_ sender: Any) {
        clearResults()
        let row = detectorPicker.selectedRow(inComponent: 0)
        if let rowIndex = DetectorPickerRow(rawValue: row) {
            switch rowIndex {
            case .detectFaceOnDevice:
                detectFaces(image: imageView.image)
            }
        } else {
            print("No such item at row \(row) in detector picker.")
        }
    }
    
    @IBAction func openPhotoLibrary(_ sender: Any) {
        imagePicker.sourceType = .photoLibrary
        present(imagePicker, animated: true)
    }
    
    @IBAction func openCamera(_ sender: Any) {
        guard UIImagePickerController.isCameraDeviceAvailable(.front) ||
            UIImagePickerController.isCameraDeviceAvailable(.rear)
            else {
                return
        }
        imagePicker.sourceType = .camera
        present(imagePicker, animated: true)
    }
    
    @IBAction func changeImage(_ sender: Any) {
        clearResults()
        currentImage = (currentImage + 1) % Constants.images.count
        imageView.image = UIImage(named: Constants.images[currentImage])
    }
    
    // MARK: - Private
    
    /// Removes the detection annotations from the annotation overlay view.
    private func removeDetectionAnnotations() {
        for annotationView in annotationOverlayView.subviews {
            annotationView.removeFromSuperview()
        }
    }
    
    /// Clears the results text view and removes any frames that are visible.
    private func clearResults() {
        removeDetectionAnnotations()
        self.resultsText = ""
    }
    
    private func showResults() {
        let resultsAlertController = UIAlertController(
            title: "Detection Results",
            message: nil,
            preferredStyle: .actionSheet
        )
        resultsAlertController.addAction(
            UIAlertAction(title: "OK", style: .destructive) { _ in
                resultsAlertController.dismiss(animated: true, completion: nil)
            }
        )
        resultsAlertController.message = resultsText
        resultsAlertController.popoverPresentationController?.barButtonItem = detectButton
        resultsAlertController.popoverPresentationController?.sourceView = self.view
        present(resultsAlertController, animated: true, completion: nil)
        print(resultsText)
    }
    
    /// Updates the image view with a scaled version of the given image.
    private func updateImageView(with image: UIImage) {
        let orientation = UIApplication.shared.statusBarOrientation
        var scaledImageWidth: CGFloat = 0.0
        var scaledImageHeight: CGFloat = 0.0
        switch orientation {
        case .portrait, .portraitUpsideDown, .unknown:
            scaledImageWidth = imageView.bounds.size.width
            scaledImageHeight = image.size.height * scaledImageWidth / image.size.width
        case .landscapeLeft, .landscapeRight:
            scaledImageWidth = image.size.width * scaledImageHeight / image.size.height
            scaledImageHeight = imageView.bounds.size.height
        }
        DispatchQueue.global(qos: .userInitiated).async {
            // Scale image while maintaining aspect ratio so it displays better in the UIImageView.
            var scaledImage = image.scaledImage(
                with: CGSize(width: scaledImageWidth, height: scaledImageHeight)
            )
            scaledImage = scaledImage ?? image
            guard let finalImage = scaledImage else { return }
            DispatchQueue.main.async {
                self.imageView.image = finalImage
            }
        }
    }
    
    private func transformMatrix() -> CGAffineTransform {
        guard let image = imageView.image else { return CGAffineTransform() }
        let imageViewWidth = imageView.frame.size.width
        let imageViewHeight = imageView.frame.size.height
        let imageWidth = image.size.width
        let imageHeight = image.size.height
        
        let imageViewAspectRatio = imageViewWidth / imageViewHeight
        let imageAspectRatio = imageWidth / imageHeight
        let scale = (imageViewAspectRatio > imageAspectRatio) ?
            imageViewHeight / imageHeight :
            imageViewWidth / imageWidth
        
        // Image view's `contentMode` is `scaleAspectFit`, which scales the image to fit the size of the
        // image view by maintaining the aspect ratio. Multiple by `scale` to get image's original size.
        let scaledImageWidth = imageWidth * scale
        let scaledImageHeight = imageHeight * scale
        let xValue = (imageViewWidth - scaledImageWidth) / CGFloat(2.0)
        let yValue = (imageViewHeight - scaledImageHeight) / CGFloat(2.0)
        
        var transform = CGAffineTransform.identity.translatedBy(x: xValue, y: yValue)
        transform = transform.scaledBy(x: scale, y: scale)
        return transform
    }
    
    private func pointFrom(_ visionPoint: VisionPoint) -> CGPoint {
        return CGPoint(x: CGFloat(visionPoint.x.floatValue), y: CGFloat(visionPoint.y.floatValue))
    }
    
    private func process(_ visionImage: VisionImage, with textRecognizer: VisionTextRecognizer?) {
        //    textRecognizer?.process(visionImage) { text, error in
        //      guard error == nil, let text = text else {
        //        let errorString = error?.localizedDescription ?? Constants.detectionNoResultsMessage
        //        self.resultsText = "Text recognizer failed with error: \(errorString)"
        //        self.showResults()
        //        return
        //      }
        //      // Blocks.
        //      for block in text.blocks {
        //        let transformedRect = block.frame.applying(self.transformMatrix())
        //        UIUtilities.addRectangle(
        //          transformedRect,
        //          to: self.annotationOverlayView,
        //          color: UIColor.purple
        //        )
        //
        //        // Lines.
        //        for line in block.lines {
        //          let transformedRect = line.frame.applying(self.transformMatrix())
        //          UIUtilities.addRectangle(
        //            transformedRect,
        //            to: self.annotationOverlayView,
        //            color: UIColor.orange
        //          )
        //
        //          // Elements.
        //          for element in line.elements {
        //            let transformedRect = element.frame.applying(self.transformMatrix())
        //            UIUtilities.addRectangle(
        //              transformedRect,
        //              to: self.annotationOverlayView,
        //              color: UIColor.green
        //            )
        //            let label = UILabel(frame: transformedRect)
        //            label.text = element.text
        //            label.adjustsFontSizeToFitWidth = true
        //            self.annotationOverlayView.addSubview(label)
        //          }
        //        }
        //      }
        //      self.resultsText += "\(text.text)\n"
        //      self.showResults()
        //    }
    }
    
    private func process(
        _ visionImage: VisionImage,
        with documentTextRecognizer: VisionDocumentTextRecognizer?
        ) {
        //    documentTextRecognizer?.process(visionImage) { text, error in
        //      guard error == nil, let text = text else {
        //        let errorString = error?.localizedDescription ?? Constants.detectionNoResultsMessage
        //        self.resultsText = "Document text recognizer failed with error: \(errorString)"
        //        self.showResults()
        //        return
        //      }
        //      // Blocks.
        //      for block in text.blocks {
        //        let transformedRect = block.frame.applying(self.transformMatrix())
        //        UIUtilities.addRectangle(
        //          transformedRect,
        //          to: self.annotationOverlayView,
        //          color: UIColor.purple
        //        )
        //
        //        // Paragraphs.
        //        for paragraph in block.paragraphs {
        //          let transformedRect = paragraph.frame.applying(self.transformMatrix())
        //          UIUtilities.addRectangle(
        //            transformedRect,
        //            to: self.annotationOverlayView,
        //            color: UIColor.orange
        //          )
        //
        //          // Words.
        //          for word in paragraph.words {
        //            let transformedRect = word.frame.applying(self.transformMatrix())
        //            UIUtilities.addRectangle(
        //              transformedRect,
        //              to: self.annotationOverlayView,
        //              color: UIColor.green
        //            )
        //
        //            // Symbols.
        //            for symbol in word.symbols {
        //              let transformedRect = symbol.frame.applying(self.transformMatrix())
        //              UIUtilities.addRectangle(
        //                transformedRect,
        //                to: self.annotationOverlayView,
        //                color: UIColor.cyan
        //              )
        //              let label = UILabel(frame: transformedRect)
        //              label.text = symbol.text
        //              label.adjustsFontSizeToFitWidth = true
        //              self.annotationOverlayView.addSubview(label)
        //            }
        //          }
        //        }
        //      }
        //      self.resultsText += "\(text.text)\n"
        //      self.showResults()
        //    }
    }
    
    // MARK: - Notifications
    
    @objc
    private func remoteModelDownloadDidSucceed(_ notification: Notification) {
        //    let notificationHandler = {
        //      self.downloadProgressView.isHidden = true
        //      guard let userInfo = notification.userInfo,
        //        let remoteModel =
        //        userInfo[ModelDownloadUserInfoKey.remoteModel.rawValue] as? RemoteModel
        //        else {
        //          self.resultsText += "firebaseMLModelDownloadDidSucceed notification posted without a RemoteModel instance."
        //          return
        //      }
        //      self.resultsText += "Successfully downloaded the remote model with name: \(remoteModel.name). The model is ready for detection."
        //    }
        //    if Thread.isMainThread { notificationHandler(); return }
        //    DispatchQueue.main.async { notificationHandler() }
    }
    
    @objc
    private func remoteModelDownloadDidFail(_ notification: Notification) {
        //    let notificationHandler = {
        //      self.downloadProgressView.isHidden = true
        //      guard let userInfo = notification.userInfo,
        //        let remoteModel =
        //        userInfo[ModelDownloadUserInfoKey.remoteModel.rawValue] as? RemoteModel,
        //        let error = userInfo[ModelDownloadUserInfoKey.error.rawValue] as? NSError
        //        else {
        //          self.resultsText += "firebaseMLModelDownloadDidFail notification posted without a RemoteModel instance or error."
        //          return
        //      }
        //      self.resultsText += "Failed to download the remote model with name: \(remoteModel.name), error: \(error)."
        //    }
        //    if Thread.isMainThread { notificationHandler(); return }
        //    DispatchQueue.main.async { notificationHandler() }
        //  }
    }
}

extension ViewController: UIPickerViewDataSource, UIPickerViewDelegate {
    
    // MARK: - UIPickerViewDataSource
    
    func numberOfComponents(in pickerView: UIPickerView) -> Int {
        return DetectorPickerRow.componentsCount
    }
    
    func pickerView(_ pickerView: UIPickerView, numberOfRowsInComponent component: Int) -> Int {
        return DetectorPickerRow.rowsCount
    }
    
    // MARK: - UIPickerViewDelegate
    
    func pickerView(
        _ pickerView: UIPickerView,
        titleForRow row: Int,
        forComponent component: Int
        ) -> String? {
        return DetectorPickerRow(rawValue: row)?.description
    }
    
    func pickerView(_ pickerView: UIPickerView, didSelectRow row: Int, inComponent component: Int) {
        clearResults()
    }
}

// MARK: - UIImagePickerControllerDelegate

extension ViewController: UIImagePickerControllerDelegate {
    
    func imagePickerController(
        _ picker: UIImagePickerController,
        didFinishPickingMediaWithInfo info: [String: Any]
        ) {
        clearResults()
        if let pickedImage = info[UIImagePickerControllerOriginalImage] as? UIImage {
            updateImageView(with: pickedImage)
        }
        dismiss(animated: true)
    }
}

/// Extension of ViewController for On-Device and Cloud detection.
extension ViewController {
    
    // MARK: - Vision On-Device Detection
    
    /// Detects faces on the specified image and draws a frame around the detected faces using
    /// On-Device face API.
    ///
    /// - Parameter image: The image.
    func detectFaces(image: UIImage?) {
        guard let image = image else { return }
        
        // Create a face detector with options.
        // [START config_face]
        let options = VisionFaceDetectorOptions()
        options.landmarkMode = .all
        options.classificationMode = .all
        options.performanceMode = .accurate
        options.contourMode = .all
        // [END config_face]
        
        // [START init_face]
        let faceDetector = vision.faceDetector(options: options)
        // [END init_face]
        
        // Define the metadata for the image.
        let imageMetadata = VisionImageMetadata()
        imageMetadata.orientation = UIUtilities.visionImageOrientation(from: image.imageOrientation)
        
        // Initialize a VisionImage object with the given UIImage.
        let visionImage = VisionImage(image: image)
        visionImage.metadata = imageMetadata
        
        // [START detect_faces]
        faceDetector.process(visionImage) { faces, error in
            guard error == nil, let faces = faces, !faces.isEmpty else {
                // [START_EXCLUDE]
                let errorString = error?.localizedDescription ?? Constants.detectionNoResultsMessage
                self.resultsText = "On-Device face detection failed with error: \(errorString)"
                self.showResults()
                // [END_EXCLUDE]
                return
            }
            
            // Faces detected
            // [START_EXCLUDE]
            faces.forEach { face in
//                let transform = self.transformMatrix()
//                let transformedRect = face.frame.applying(transform)
//                UIUtilities.addRectangle(
//                    transformedRect,
//                    to: self.annotationOverlayView,
//                    color: UIColor.green
//                )
            }
            self.resultsText = faces.map { face in
                let headEulerAngleY = face.hasHeadEulerAngleY ? face.headEulerAngleY.description : "NA"
                let headEulerAngleZ = face.hasHeadEulerAngleZ ? face.headEulerAngleZ.description : "NA"
                let leftEyeOpenProbability = face.hasLeftEyeOpenProbability ? face.leftEyeOpenProbability.description : "NA"
                let rightEyeOpenProbability = face.hasRightEyeOpenProbability ? face.rightEyeOpenProbability.description : "NA"
                let smilingProbability = face.hasSmilingProbability ? face.smilingProbability.description : "NA"
                let output = """
                Frame: \(face.frame)
                Head Euler Angle Y: \(headEulerAngleY)
                Head Euler Angle Z: \(headEulerAngleZ)
                Left Eye Open Probability: \(leftEyeOpenProbability)
                Right Eye Open Probability: \(rightEyeOpenProbability)
                Smiling Probability: \(smilingProbability)
                """
                return "\(output)"
                }.joined(separator: "\n")
            self.showResults()
            // [END_EXCLUDE]
        }
        // [END detect_faces]
    }
}

// MARK: - Enums

private enum DetectorPickerRow: Int {
    case detectFaceOnDevice = 0
    
    static let rowsCount = 1
    static let componentsCount = 1
    
    public var description: String {
        return "Face On-Device"
    }
}

private enum Constants {
    static let images = ["uber.jpg", "grace_hopper.jpg", "barcode_128.png", "qr_code.jpg", "beach.jpg",
                         "image_has_text.jpg", "liberty.jpg"]
    static let modelExtension = "tflite"
    static let localModelName = "mobilenet"
    static let quantizedModelFilename = "mobilenet_quant_v1_224"
    
    static let detectionNoResultsMessage = "No results returned."
    static let failedToDetectObjectsMessage = "Failed to detect objects in image."
    static let sparseTextModelName = "Sparse"
    static let denseTextModelName = "Dense"
    
//    static let localAutoMLModelName = "local_automl_model"
//    static let remoteAutoMLModelName = "remote_automl_model"
//    static let localModelManifestFileName = "automl_labeler_manifest"
//    static let autoMLManifestFileType = "json"
    
    static let labelConfidenceThreshold: Float = 0.75
    static let smallDotRadius: CGFloat = 5.0
    static let largeDotRadius: CGFloat = 10.0
    static let lineColor = UIColor.yellow.cgColor
    static let fillColor = UIColor.clear.cgColor
}
