# Autoencoder to extract key clips from video
This repo is based on:\
Junyu Chen et al 2021 J. Phys.: Conf. Ser. 2025 012018

## Architecture
The architecture is as follows:
![Architecture](https://github.com/ashrithjacob/Best-Frame-Retrieval/blob/main/images/architecture.png?raw=true)

## Dataset
The dataset used is Caltech-256. It can be downloaded from [here](http://www.vision.caltech.edu/Image_Datasets/Caltech256/).\
Having trained the autoencoder on the above dataset, we get the lwest training loss at epoch 50. The loss is as follows:
![Test-Loss-Cal256](https://github.com/ashrithjacob/Best-Frame-Retrieval/blob/main/images/test_loss_cal256.png?raw=true)

On further fine-tuning the model on the frames extracted from the videos, we get the following loss:
![Test-Loss-custom-data](https://github.com/ashrithjacob/Best-Frame-Retrieval/blob/main/images/test_loss_customdata.png?raw=true)

Thus epoch 50 is used to extract features from the encoder part of the autoencoder.

## Comparing consecutive frame similarity with and without autoencoder
In order to extract the key frames, conseccutive similar frames are to be identified and removed. This is done by calculating the similarity between consecutive frames. For this, the metric used is the MS-SSIM-L1 loss function([paper](https://arxiv.org/pdf/1511.08861.pdf))./
Two consective frames are considered similar if the loss between them is less than a threshold value. This value can be set by the user and will determine the number of frames generated to represent the video (higher the threshold, fewer the number of frames).\
One can see the difference in the loss between consecutive frames with and without the autoencoder below:
![Loss-Comparison](https://github.com/ashrithjacob/Best-Frame-Retrieval/blob/main/images/consecutive_frames.png?raw=true)
