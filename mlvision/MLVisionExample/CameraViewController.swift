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
    @IBOutlet var instructionsLabel: UILabel!
    @IBOutlet var statusLabel: UILabel!
    
    private var chosen = [String]()
    private var allGestures = ["smile", "blink", "turnRight", "turnLeft", "tiltRight", "tiltLeft"]

    var featuresStruct = trackFeatures(smile: 0, blink: 0, turnRight: 0, turnLeft: 0, tiltRight: 0, tiltLeft: 0)
    var smileArr = [Double](repeating: 0.0, count: 6), blinkArr = [Double](repeating: 0.0, count: 6), turnArr = [Double](repeating: 0.0, count: 6), tiltArr = [Double](repeating: 0.0, count: 6)

    let probThreshold = 0.5
    let turnThreshold = 30.0
    let tiltThreshold = 20.0
    
    var faceOffScreen = false

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
        chooseGestures(number: 3)
        setupStruct()
        
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
    
    private func chooseGestures(number: Int) {
        for _ in 1...number {
            var choose = allGestures.randomElement()!
            while chosen.last == choose {
                choose = allGestures.randomElement()!
            }
            chosen.append(choose)
        }
        instructionsLabel.text = chosen.joined(separator: ", ")
    }
    
    private func setStruct(gesture: String, val: Int) {
        switch gesture {
        case "smile":
            featuresStruct.smile = val
        case "blink":
            featuresStruct.blink = val
        case "turnRight":
            featuresStruct.turnRight = val
        case "turnLeft":
            featuresStruct.turnLeft = val
        case "tiltLeft":
            featuresStruct.tiltLeft = val
        case "tiltRight":
            featuresStruct.tiltRight = val
        default: break
        }
    }
    
    private func setupStruct() {
        for feature in chosen {
            setStruct(gesture: feature, val: 1)
        }
    }

    private func featureTracker(storeValue: Double, feature: String, states: inout Array<Double>) -> Bool {
        states.append(storeValue)
        var recent = states.suffix(4)
        var check = false
        
        if feature == "smile" {
            check = recent.allSatisfy { $0 > probThreshold }
        }
        if feature == "blink" {
            recent = states.suffix(3)
            check = recent.allSatisfy { $0 < probThreshold }
        }
        if feature == "turnRight" {
            check = recent.allSatisfy{ $0 > turnThreshold }
        }
        if feature == "turnLeft" {
            check = recent.allSatisfy{ $0 < -turnThreshold }
        }
        if feature == "tiltLeft" {
            check = recent.allSatisfy{ $0 > tiltThreshold }
        }
        if feature == "tiltRight" {
            check = recent.allSatisfy{ $0 < -tiltThreshold }
        }
        return check
    }
    
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
        
        print(faceOffScreen)
        
        var detectedFaces: [VisionFace]? = nil
        do {
            detectedFaces = try faceDetector.results(in: image)
        } catch let error {
            print("Failed to detect faces with error: \(error.localizedDescription).")
        }
        guard let faces = detectedFaces, !faces.isEmpty else {
            print("Please show your face on the screen.")
            faceOffScreen = true
            DispatchQueue.main.sync {
                self.updatePreviewOverlayView()
                self.removeDetectionAnnotations()
                statusLabel.text = "FACE IS OFF SCREEN"
            }
            return
        }
        faceOffScreen = false

        DispatchQueue.main.sync {
            self.updatePreviewOverlayView()
            self.removeDetectionAnnotations()
            for face in faces {
                statusLabel.text = "FACE IS NOT OFF SCREEN"
                var checkSmile = false, checkBlink = false, checkturnRight = false, checkturnLeft = false, checktiltRight = false, checktiltLeft = false
                if featuresStruct.smile == 1 && chosen[0] == "smile" {
                    let smileProb = face.smilingProbability
                    checkSmile = featureTracker(storeValue : Double(smileProb), feature: "smile", states: &smileArr)
                }
                if featuresStruct.blink == 1  && chosen[0] == "blink" {
                    let blinkProb = (face.leftEyeOpenProbability + face.rightEyeOpenProbability) / 2
                    checkBlink = featureTracker(storeValue : Double(blinkProb), feature: "blink", states: &blinkArr)
                }
                if featuresStruct.turnRight == 1  && chosen[0] == "turnRight" {
                    let turnProb = face.headEulerAngleY
                    checkturnRight = featureTracker(storeValue : Double(turnProb), feature: "turnRight", states: &turnArr)
                }
                if featuresStruct.turnLeft == 1  && chosen[0] == "turnLeft" {
                    let turnProb = face.headEulerAngleY
                    checkturnLeft = featureTracker(storeValue : Double(turnProb), feature: "turnLeft", states: &turnArr)
                }
                if featuresStruct.tiltRight == 1  && chosen[0] == "tiltRight" {
                    let tiltProb = face.headEulerAngleZ
                    checktiltRight = featureTracker(storeValue : Double(tiltProb), feature: "tiltEight", states: &tiltArr)
                }
                if featuresStruct.tiltLeft == 1  && chosen[0] == "tiltLeft" {
                    let tiltProb = face.headEulerAngleZ
                    checktiltLeft = featureTracker(storeValue : Double(tiltProb), feature: "tiltLeft", states: &tiltArr)
                }
                let arr = [checkSmile, checkBlink, checkturnRight, checkturnLeft, checktiltRight, checktiltLeft]
                //statusLabel.text = ""
                //print(arr)
                for elem in arr {
                    if elem == true {
                        print(chosen)
                        chosen.removeFirst(1)
                        print(chosen)
                        instructionsLabel.text = chosen.joined(separator: ", ")
                        if chosen.isEmpty {
                            instructionsLabel.text = "ALL GESTURES COMPLETED"
                        }
                        let index = arr.index(of: elem)!
                        let feature = allGestures[index]
                        setStruct(gesture: feature, val: 0)
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

struct trackFeatures {
    var smile = 0
    var blink = 0
    var turnRight = 0
    var turnLeft = 0
    var tiltRight = 0
    var tiltLeft = 0
}
