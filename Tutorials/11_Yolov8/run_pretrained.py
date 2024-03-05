import cv2
from ultralytics.engine.model import Model as BaseModel
from mltu.torch.yolo.detectors.torch_detector import Detector as TorchDetector
from mltu.torch.yolo.detectors.onnx_detector import Detector as OnnxDetector


classes = {v: v for v in range(80)}
input_width, input_height = 320, 320

confidence_threshold = 0.5
iou_threshold = 0.5

# base_model = BaseModel("yolov8m.pt")
# detector = TorchDetector(base_model.model, input_width, input_height, classes, confidence_threshold, iou_threshold)
detector = OnnxDetector("yolov8m.onnx", input_width, input_height, confidence_threshold, iou_threshold)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform Yolo object detection
    detections = detector(frame)

    # Apply the detections to the frame
    frame = detections.applyToFrame(frame)

    # Print the FPS
    print(detector.fps)

    # Display the output image
    cv2.imshow("Object Detection", frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()