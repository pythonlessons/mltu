from mltu.preprocessors import ImageReader
from mltu.dataProvider import DataProvider
from mltu.annotations.images import CVImage
from mltu.transformers import ImageShowCV2
from mltu.torch.yolo.annotation import TextOCRAnnotationReader


dataset = TextOCRAnnotationReader.readAnnotations(
    json_path = "Datasets/TextOCR/TextOCR_0.1_train.json",
    images_path = "Datasets/TextOCR/train_val_images/train_images",
    limit=100
    )

data_provider = DataProvider(
    dataset=dataset,
    skip_validation=True,
    batch_size=8,
    data_preprocessors=[
        TextOCRAnnotationReader(),
        ImageReader(CVImage),
        ImageShowCV2(draw_label=False),
        ],
    numpy=False,
)

for batch in data_provider:
    pass