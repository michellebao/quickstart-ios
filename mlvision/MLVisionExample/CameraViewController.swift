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

import AVFoundation
import CoreVideo

import FirebaseMLVision
import FirebaseMLVisionObjectDetection
import FirebaseMLCommon
import FirebaseMLVisionAutoML

@objc(CameraViewController)
class CameraViewController: UIViewController {
    //  private let detectors: [Detector] = [.onDeviceAutoMLImageLabeler,
    //                                       .onDeviceFace,
    //                                       .onDeviceText,
    //                                       .onDeviceObjectProminentNoClassifier,
    //                                       .onDeviceObjectProminentWithClassifier,
    //                                       .onDeviceObjectMultipleNoClassifier,
    //                                       .onDeviceObjectMultipleWithClassifier]
    private let detectors: [Detector] = [.onDeviceFace]
    private var currentDetector: Detector = .onDeviceFace
    private var isUsingFrontCamera = true
    private var previewLayer: AVCaptureVideoPreviewLayer!
    private lazy var captureSession = AVCaptureSession()
    private lazy var sessionQueue = DispatchQueue(label: Constant.sessionQueueLabel)
    private lazy var vision = Vision.vision()
    private var lastFrame: CMSampleBuffer?
    private var areAutoMLModelsRegistered = false
    private lazy var modelManager = ModelManager.modelManager()
    @IBOutlet var downloadProgressView: UIProgressView!
    
    private lazy var previewOverlayView: UIImageView = {
        
        precondition(isViewLoaded)
        let previewOverlayView = UIImageView(frame: .zero)
        previewOverlayView.contentMode = UIViewContentMode.scaleAspectFill
        previewOverlayView.translatesAutoresizingMaskIntoConstraints = false
        return previewOverlayView
    }()
    
    private lazy var annotationOverlayView: UIView = {
        precondition(isViewLoaded)
        let annotationOverlayView = UIView(frame: .zero)
        annotationOverlayView.translatesAutoresizingMaskIntoConstraints = false
        return annotationOverlayView
    }()
    
    // MARK: - IBOutlets
    
    @IBOutlet private weak var cameraView: UIView!
    
    // MARK: - UIViewController
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        setUpPreviewOverlayView()
        setUpAnnotationOverlayView()
        setUpCaptureSessionOutput()
        setUpCaptureSessionInput()
        //view.addSubview(UILabel)
    }
    
    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        
        startSession()
    }
    
    override func viewDidDisappear(_ animated: Bool) {
        super.viewDidDisappear(animated)
        
        stopSession()
    }
    
    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        
        previewLayer.frame = cameraView.frame
    }
    
    // MARK: - IBActions
    
    @IBAction func selectDetector(_ sender: Any) {
        presentDetectorsAlertController()
    }
    
    @IBAction func switchCamera(_ sender: Any) {
        isUsingFrontCamera = !isUsingFrontCamera
        removeDetectionAnnotations()
        setUpCaptureSessionInput()
    }
    
    // MARK: - Notifications
    
    @objc
    private func remoteModelDownloadDidSucceed(_ notification: Notification) {
        let notificationHandler = {
            self.downloadProgressView.isHidden = true
            guard let userInfo = notification.userInfo,
                let remoteModel =
                userInfo[ModelDownloadUserInfoKey.remoteModel.rawValue] as? RemoteModel
                else {
                    print("firebaseMLModelDownloadDidSucceed notification posted without a RemoteModel instance.")
                    return
            }
            print("Successfully downloaded the remote model with name: \(remoteModel.name). The model is ready for detection.")
        }
        if Thread.isMainThread { notificationHandler(); return }
        DispatchQueue.main.async { notificationHandler() }
    }
    
    @objc
    private func remoteModelDownloadDidFail(_ notification: Notification) {
        let notificationHandler = {
            self.downloadProgressView.isHidden = true
            guard let userInfo = notification.userInfo,
                let remoteModel =
                userInfo[ModelDownloadUserInfoKey.remoteModel.rawValue] as? RemoteModel,
                let error = userInfo[ModelDownloadUserInfoKey.error.rawValue] as? NSError
                else {
                    print("firebaseMLModelDownloadDidFail notification posted without a RemoteModel instance or error.")
                    return
            }
            print("Failed to download the remote model with name: \(remoteModel.name), error: \(error).")
        }
        if Thread.isMainThread { notificationHandler(); return }
        DispatchQueue.main.async { notificationHandler() }
    }
    
    // MARK: Other On-Device Detections
    
//    private func featureClassification(storeValue: Double, feature: String) {
//        switch feature {
//        case "smile":
//            let threshold = 0.6
//            if storeValue > threshold {
//                status = 1
//            }
//            else {
//                status = 0
//            }
//        }
//        case "eyes" {
//            let threshold = 0.6
//            if storeValue > threshold {
//                status = 1
//            }
//            else {
//                status = 0
//            }
//        }
//        case "turn" {
//            let threshold = 30
//
//
//        }
//        case "tilt" {
//            let threshold = 30
//        }
//        default: break
//
//        let probThreshold = 0.6
//        let eulerThreshold = 30.0
//        if feature == "smile" {
//            if storeValue > probThreshold {
//                featureTracker(1, feature)
//            }
//            else {
//                featureTracker(0, feature)
//            }
//        }
//        else if feature == "eyes" {
//
//        }
//    }
//
//    private func featureTracker(storeValue: Bool, feature: String)
    
    private func detectFacesOnDevice(in image: VisionImage, width: CGFloat, height: CGFloat) {
        let options = VisionFaceDetectorOptions()
        
        // When performing latency tests to determine ideal detection settings,
        // run the app in 'release' mode to get accurate performance metrics
        options.landmarkMode = .all
        options.contourMode = .all
        options.classificationMode = .all
        options.isTrackingEnabled = true
        
        options.performanceMode = .fast
        let faceDetector = vision.faceDetector(options: options)
        
        var detectedFaces: [VisionFace]? = nil
        do {
            detectedFaces = try faceDetector.results(in: image)
        } catch let error {
            print("Failed to detect faces with error: \(error.localizedDescription).")
        }
        guard let faces = detectedFaces, !faces.isEmpty else {
            print("Please show your face on the screen.")
            DispatchQueue.main.sync {
                self.updatePreviewOverlayView()
                self.removeDetectionAnnotations()
            }
            return
        }
        
        DispatchQueue.main.sync {
            self.updatePreviewOverlayView()
            self.removeDetectionAnnotations()
            for face in faces {
                if face.hasSmilingProbability {
                    let smileProb = face.smilingProbability
                    if smileProb > 0.6 {
                        print(":D")
                    }
                    else {
                        print("D:")
                    }
                }
                
                if face.hasLeftEyeOpenProbability && face.hasRightEyeOpenProbability {
                    let leftProb = face.leftEyeOpenProbability
                    let rightProb = face.rightEyeOpenProbability
                    if leftProb > 0.6 && rightProb > 0.6 {
                        print("EYES OPEN")
                    }
                    else {
                        print ("EYES CLOSED")
                    }
                }
                
                if face.hasHeadEulerAngleY {
                    let rotY = face.headEulerAngleY  // Head is rotated to the right rotY degrees
                    if rotY > 30 {
                        print("LOOKING TO THE RIGHT")
                    }
                    else if rotY < -30 {
                        print("LOOKING TO THE LEFT")
                    }
                }
                if face.hasHeadEulerAngleZ {
                    let rotZ = face.headEulerAngleZ  // Head is rotated upward rotZ degrees
                    //print(rotZ)
                    if rotZ > 30 {
                        print("TILTING HEAD TO THE LEFT")
                    }
                    else if rotZ < -30 {
                        print("TILTING HEAD TO THE RIGHT")
                    }
                }
            }
        }
    }
    
    // MARK: - Private
    
    private func setUpCaptureSessionOutput() {
        sessionQueue.async {
            self.captureSession.beginConfiguration()
            // When performing latency tests to determine ideal capture settings,
            // run the app in 'release' mode to get accurate performance metrics
            self.captureSession.sessionPreset = AVCaptureSession.Preset.medium
            
            let output = AVCaptureVideoDataOutput()
            output.videoSettings =
                [(kCVPixelBufferPixelFormatTypeKey as String): kCVPixelFormatType_32BGRA]
            let outputQueue = DispatchQueue(label: Constant.videoDataOutputQueueLabel)
            output.setSampleBufferDelegate(self, queue: outputQueue)
            guard self.captureSession.canAddOutput(output) else {
                print("Failed to add capture session output.")
                return
            }
            self.captureSession.addOutput(output)
            self.captureSession.commitConfiguration()
        }
    }
    
    private func setUpCaptureSessionInput() {
        sessionQueue.async {
            let cameraPosition: AVCaptureDevice.Position = self.isUsingFrontCamera ? .front : .back
            guard let device = self.captureDevice(forPosition: cameraPosition) else {
                print("Failed to get capture device for camera position: \(cameraPosition)")
                return
            }
            do {
                self.captureSession.beginConfiguration()
                let currentInputs = self.captureSession.inputs
                for input in currentInputs {
                    self.captureSession.removeInput(input)
                }
                
                let input = try AVCaptureDeviceInput(device: device)
                guard self.captureSession.canAddInput(input) else {
                    print("Failed to add capture session input.")
                    return
                }
                self.captureSession.addInput(input)
                self.captureSession.commitConfiguration()
            } catch {
                print("Failed to create capture device input: \(error.localizedDescription)")
            }
        }
    }
    
    private func startSession() {
        sessionQueue.async {
            self.captureSession.startRunning()
        }
    }
    
    private func stopSession() {
        sessionQueue.async {
            self.captureSession.stopRunning()
        }
    }
    
    private func setUpPreviewOverlayView() {
        cameraView.addSubview(previewOverlayView)
        NSLayoutConstraint.activate([
            previewOverlayView.centerXAnchor.constraint(equalTo: cameraView.centerXAnchor),
            previewOverlayView.centerYAnchor.constraint(equalTo: cameraView.centerYAnchor),
            previewOverlayView.leadingAnchor.constraint(equalTo: cameraView.leadingAnchor),
            previewOverlayView.trailingAnchor.constraint(equalTo: cameraView.trailingAnchor),
            
            ])
    }
    
    private func setUpAnnotationOverlayView() {
        cameraView.addSubview(annotationOverlayView)
        NSLayoutConstraint.activate([
            annotationOverlayView.topAnchor.constraint(equalTo: cameraView.topAnchor),
            annotationOverlayView.leadingAnchor.constraint(equalTo: cameraView.leadingAnchor),
            annotationOverlayView.trailingAnchor.constraint(equalTo: cameraView.trailingAnchor),
            annotationOverlayView.bottomAnchor.constraint(equalTo: cameraView.bottomAnchor),
            ])
    }
    
    private func captureDevice(forPosition position: AVCaptureDevice.Position) -> AVCaptureDevice? {
        if #available(iOS 10.0, *) {
            let discoverySession = AVCaptureDevice.DiscoverySession(
                deviceTypes: [.builtInWideAngleCamera],
                mediaType: .video,
                position: .unspecified
            )
            return discoverySession.devices.first { $0.position == position }
        }
        return nil
    }
    
    private func presentDetectorsAlertController() {
        let alertController = UIAlertController(
            title: Constant.alertControllerTitle,
            message: Constant.alertControllerMessage,
            preferredStyle: .alert
        )
        detectors.forEach { detectorType in
            let action = UIAlertAction(title: detectorType.rawValue, style: .default) {
                [unowned self] (action) in
                guard let value = action.title else { return }
                guard let detector = Detector(rawValue: value) else { return }
                self.currentDetector = detector
                self.removeDetectionAnnotations()
            }
            if detectorType.rawValue == currentDetector.rawValue { action.isEnabled = false }
            alertController.addAction(action)
        }
        alertController.addAction(UIAlertAction(title: Constant.cancelActionTitleText, style: .cancel))
        present(alertController, animated: true)
    }
    
    private func removeDetectionAnnotations() {
        for annotationView in annotationOverlayView.subviews {
            annotationView.removeFromSuperview()
        }
    }
    
    private func updatePreviewOverlayView() {
        guard let lastFrame = lastFrame,
            let imageBuffer = CMSampleBufferGetImageBuffer(lastFrame)
            else {
                return
        }
        let ciImage = CIImage(cvPixelBuffer: imageBuffer)
        let context = CIContext(options: nil)
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else {
            return
        }
        let rotatedImage =
            UIImage(cgImage: cgImage, scale: Constant.originalScale, orientation: .right)
        if isUsingFrontCamera {
            guard let rotatedCGImage = rotatedImage.cgImage else {
                return
            }
            let mirroredImage = UIImage(
                cgImage: rotatedCGImage, scale: Constant.originalScale, orientation: .leftMirrored)
            previewOverlayView.image = mirroredImage
        } else {
            previewOverlayView.image = rotatedImage
        }
    }
}

// MARK: AVCaptureVideoDataOutputSampleBufferDelegate

extension CameraViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
    
    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
        ) {
        guard let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            print("Failed to get image buffer from sample buffer.")
            return
        }
        lastFrame = sampleBuffer
        let visionImage = VisionImage(buffer: sampleBuffer)
        let metadata = VisionImageMetadata()
        let orientation = UIUtilities.imageOrientation(
            fromDevicePosition: isUsingFrontCamera ? .front : .back
        )
        
        let visionOrientation = UIUtilities.visionImageOrientation(from: orientation)
        metadata.orientation = visionOrientation
        visionImage.metadata = metadata
        let imageWidth = CGFloat(CVPixelBufferGetWidth(imageBuffer))
        let imageHeight = CGFloat(CVPixelBufferGetHeight(imageBuffer))
        
        switch currentDetector {
        case .onDeviceFace:
            detectFacesOnDevice(in: visionImage, width: imageWidth, height: imageHeight)
        }
    }
}

// MARK: - Constants

public enum Detector: String {
    case onDeviceFace = "On-Device Face Detection"
}

private enum Constant {
    static let alertControllerTitle = "Vision Detectors"
    static let alertControllerMessage = "Select a detector"
    static let cancelActionTitleText = "Cancel"
    static let videoDataOutputQueueLabel = "com.google.firebaseml.visiondetector.VideoDataOutputQueue"
    static let sessionQueueLabel = "com.google.firebaseml.visiondetector.SessionQueue"
    static let noResultsMessage = "No Results"
    static let labelConfidenceThreshold: Float = 0.75
    static let smallDotRadius: CGFloat = 4.0
    static let originalScale: CGFloat = 1.0
    static let padding: CGFloat = 10.0
    static let resultsLabelHeight: CGFloat = 200.0
    static let resultsLabelLines = 5
}
