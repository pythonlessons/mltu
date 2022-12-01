import cv2

class ImageReader:
    def __init__(self, method: int = cv2.IMREAD_COLOR, *args, **kwargs):
        self._method = method

    def __call__(self, image_path: str, label: str):
        return cv2.imread(image_path, self._method), label