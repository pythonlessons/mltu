## [0.1.5] - 2022-01-03

### Changed
- changed CWERMetric in mltu.metrics, Character/word rate was calculatted in a wrong way
- created @setter for augmentors and transformers in DataProvider, to properlly add augmentors and transformers to the pipeline
- augmentors and transformers must inherit from `mltu.augmentors.base.Augmentor` and `mltu.transformers.base.Transformer` respectively
- added better explained documentation

### Added:
- added RandomSharpen to mltu.augmentors, used for simple image augmentation;
- added ImageShowCV2 to mltu.transformers, used to show image with cv2 for debugging purposes;

## [0.1.4] - 2022-12-21

### Added:
- added mltu.augmentors (RandomBrightness, RandomRotate, RandomErodeDilate) - used for simple image augmentation;

## [0.1.3] - 2022-12-20

Initial release of mltu (Machine Learning Training Utilities)

- Project to help with training machine learning models