import cv2
import numpy as np
from fastai.vision import ImageMultiDataset
from fastai.vision.image import *


# adapted from https://www.kaggle.com/iafoss/pretrained-resnet34-with-rgby-0-460-public-lb
def open_4_channel(fname):
    fname = str(fname)
    # strip extension before adding color
    if fname.endswith('.png'):
        fname = fname[:-4]
    colors = ['red','green','blue','yellow']
    flags = cv2.IMREAD_GRAYSCALE
    img = [cv2.imread(fname+'_'+color+'.png', flags).astype(np.float32)/255
           for color in colors]
    
    x = np.stack(img, axis=-1)
    return Image(pil2tensor(x, np.float32).float())


class ImageMulti4Channel(ImageMultiDataset):
    def __init__(self, fns, labels, classes=None, **kwargs):
        super().__init__(fns, labels, classes, **kwargs)
        self.image_opener = open_4_channel