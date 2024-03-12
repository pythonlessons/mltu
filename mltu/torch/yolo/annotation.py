import os
import typing
from pathlib import Path
import xml.etree.ElementTree as ET
from mltu.annotations.detections import Detections, Detection, BboxType

class VOCAnnotationReader:
    """Reads annotations from VOC format
    """
    def __init__(self, labels: dict, images_path: str=None):
        self.labels = labels
        self.images_path = images_path

    @staticmethod
    def readFromVOC(voc_annotation_path: str, labels: dict, images_path: str=None) -> Detections:
        annotation_path = Path(voc_annotation_path)
        tree = ET.parse(voc_annotation_path)
        root = tree.getroot()

        annotation_dict = {}

        # Iterate through child elements
        for child in root:
            if child.tag == 'object':
                obj_dict = {}
                for obj_child in child:
                    if obj_child.tag == 'bndbox':
                        bbox_dict = {}
                        for bbox_child in obj_child:
                            bbox_dict[bbox_child.tag] = int(bbox_child.text)
                        obj_dict[obj_child.tag] = bbox_dict
                    else:
                        obj_dict[obj_child.tag] = obj_child.text
                if 'objects' not in annotation_dict:
                    annotation_dict['objects'] = []
                annotation_dict['objects'].append(obj_dict)
            elif child.tag == 'size':
                size_dict = {}
                for size_child in child:
                    size_dict[size_child.tag] = int(size_child.text)
                annotation_dict['size'] = size_dict
            else:
                annotation_dict[child.tag] = child.text

        # Get the image path if not provided
        if images_path is None:
            images_path = annotation_path.parent.parent / annotation_dict["folder"]

        image_path = os.path.join(images_path, annotation_dict['filename'])
        dets = []
        for obj in annotation_dict['objects']:
            if obj['name'] not in labels.values():
                print(f"Label {obj['name']} not found in labels")
                continue

            dets.append(Detection(
                bbox=[obj['bndbox']['xmin'], obj['bndbox']['ymin'], obj['bndbox']['xmax'], obj['bndbox']['ymax']],
                label=obj['name'],
                bbox_type=BboxType.XYXY,
                confidence=1,
                image_path=image_path,
                width=annotation_dict['size']['width'],
                height=annotation_dict['size']['height'],
                relative=False
                ))
        
        detections = Detections(
            labels=labels,
            width=annotation_dict['size']['width'],
            height=annotation_dict['size']['height'],
            image_path=image_path,
            detections=dets
        )

        return detections
    
    def __call__(self, image: typing.Any, annotation: str) -> typing.Tuple[typing.Any, Detections]:
        detections = self.readFromVOC(annotation, self.labels, self.images_path)
        if image is None:
            image = detections.image_path
        return image, detections