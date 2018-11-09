# Human Protein Atlas Image Classification Starter Pack

This is the code you need to train a resnet model and submit to the Human Protein Atlas competition. 
It uses the newest version (v1) of the [fastai library](https://github.com/fastai/fastai).

To download the data, I used the [Official Kaggle API package](https://github.com/Kaggle/kaggle-api), which you can install with pip. Once installed, you can run `kaggle competitions download -c human-protein-atlas-image-classification` to get the data (just make sure to update the `path` variable in the `resnet50_basic.ipynb` notebook to point to the data on your machine).
