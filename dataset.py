from fastai.vision import ImageMultiDataset
from utils import open_4_channel


class ImageMulti4Channel(ImageMultiDataset):
    def __init__(self, fns, labels, classes=None, **kwargs):
        super().__init__(fns, labels, classes, **kwargs)
        self.image_opener = open_4_channel