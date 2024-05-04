import os
import cv2
from mltu.annotations.detections import Detections
from mltu.torch.yolo.detectors.onnx_detector import Detector as OnnxDetector

# https://www.kaggle.com/datasets/andrewmvd/car-plate-detection
images_path = "Datasets/car-plate-detection/images"

input_width, input_height = 416, 416
confidence_threshold = 0.5
iou_threshold = 0.5

detector = OnnxDetector("Models/11_Yolov8/1714135287/model.onnx", input_width, input_height, confidence_threshold, iou_threshold, force_cpu=False)

for image_path in os.listdir(images_path):

    frame = cv2.imread(os.path.join(images_path, image_path))

    # Perform Yolo object detection
    detections: Detections = detector(frame)

    # Apply the detections to the frame
    frame = detections.applyToFrame(frame)

    # Print the FPS
    print(detector.fps)

    # Display the output image
    cv2.imshow("Object Detection", frame)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()