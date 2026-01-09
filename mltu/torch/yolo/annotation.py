import os
import json
import typing
from tqdm import tqdm
from pathlib import Path
import xml.etree.ElementTree as ET
from mltu.annotations.detections import Detections, Detection, BboxType

import logging
logger = logging.getLogger(__name__)


class VOCAnnotationReader:
    """Reads annotations from VOC format
    """
    def __init__(self, labels: dict, images_path: str=None):
        self.labels = labels
        self.images_path = images_path
        self.dataset_found_labels = {}

    @staticmethod
    def readFromVOC(voc_annotation_path: str, labels: dict={}, images_path: str=None) -> Detections:
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
            if labels and obj['name'] not in labels.values():
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
    

class TextOCRAnnotationReader:
    """ Reads annotations from TextOCR format into Detections format
    https://www.kaggle.com/datasets/robikscube/textocr-text-extraction-from-images-dataset/code

    Methods:
        readAnnotations: Reads annotations from TextOCR format into dictionary format
        readTextOCR: Reads a single annotation from TextOCR format into Detections format
    """
    def __init__(self, labels: dict={0: 'text'}):
        """ Initializes the TextOCRAnnotationReader
        
        Args:
            labels (dict, optional): A dictionary containing the labels. Defaults to {0: 'text'}.
        """
        self.labels = labels

    @staticmethod
    def readAnnotations(json_path: str, images_path: str, limit: int=None) -> typing.List:
        """ Reads annotations from TextOCR format into dictionary format

        Args:
            json_path (str): Path to the json file containing the annotations
            images_path (str): Path to the images folder
            limit (int, optional): Limit the number of annotations to read. Defaults to None.

        Returns:
            list: A list of lists containing the image path and the detections
        """
        if not os.path.exists(json_path):
            logger.error(f"File {json_path} not found")
            raise FileNotFoundError(f"File {json_path} not found")
        
        # Read full json file
        with open(json_path, 'r') as f:
            annotations = json.load(f)

        # Iterate through the annotations
        dataset = []
        for key, values in tqdm(annotations["imgToAnns"].items(), desc="Reading annotations..."):
            image_data = annotations["imgs"].get(key, None)
            image_path = os.path.normpath(os.path.join(images_path, Path(image_data["file_name"]).name))
            image_data["image_path"] = image_path
            if not os.path.exists(image_path):
                logger.error(f"Image {image_path} not found")
                continue

            dataset.append([
                image_path, 
                {
                    "image_data": image_data, 
                    "annotations": [annotations["anns"].get(ann_id, None) for ann_id in values]
                }
            ])

            if limit and len(dataset) >= limit:
                break

        return dataset
    
    @staticmethod
    def readTextOCR(annotation: dict, labels: dict) -> Detections:
        """ Reads a single annotation from TextOCR format into Detections format

        Args:
            annotation (dict): A dictionary containing the annotation
            labels (dict): A dictionary containing the labels

        Returns:
            Detections: A Detections object containing the detections
        """
        dets = []
        image_data = annotation["image_data"]
        anns = annotation["annotations"]
        for ann in anns:
            dets.append(Detection(
                bbox=ann["bbox"],
                label=labels[0],
                bbox_type=BboxType.LTWH,
                confidence=1,
                image_path=image_data["image_path"],
                width=image_data["width"],
                height=image_data["height"],
                relative=False,
                metadata=ann
                ))
            
        return Detections(
            labels=labels,
            width=image_data["width"],
            height=image_data["height"],
            image_path=image_data["image_path"],
            detections=dets
        )

    def __call__(self, image: typing.Any, annotation: typing.List) -> typing.Tuple[typing.Any, Detections]:
        """ Reads a single annotation from TextOCR format into Detections format

        Args:
            image (typing.Any): The image path or image object
            annotation (typing.List): A list containing the image path and the annotation

        Returns:
            typing.Tuple[typing.Any, Detections]: A tuple containing the image path and the detections
        """
        detections = TextOCRAnnotationReader.readTextOCR(annotation, self.labels)
        if image is None:
            image = detections.image_path
        return image, detections