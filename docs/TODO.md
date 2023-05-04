## TODO:
1. Load Caltech-256 -> `DONE`
2. Train-Test split -> `DONE`
3. Define loss function (MS-SSIM + L1) -> `DONE`
4. Ways to reduce loss on training:
    - Add batch norm. See paper again --> `Batch normalization is applied to the input of each convolutional layer except the last one: Got 4x better results (0-10 epochs in checkpoint)`
    - Try to remove relu from last part of encoder + leaky relu--> `DONE: got marginally better results(10-20 epochs in checkpoint)`
    - Use loss params as in paper (alpha =0.84 from 0.025 (`checkpoint 30`) , see G value as well)
    - SGD >>> Adam (`2x better results`)
    - epoch 40-50: `batch_size=16`
5. See 'num_workers' in dataloader
6. See below for model.eval vs model.train mode discrepancy in loss
https://github.com/pytorch/pytorch/issues/5406
https://discuss.pytorch.org/t/model-eval-gives-incorrect-loss-for-model-with-batchnorm-layers/7561/3
https://www.youtube.com/watch?v=nUUqwaxLnWs&ab_channel=DeepLearningAI

## Feature extraction
1. Extract features from the encoder part of the autoencoder
2. https://link.springer.com/chapter/10.1007/978-3-319-51281-5_40

## Getting least blurry frame
1. https://stackoverflow.com/questions/7765810/is-there-a-way-to-detect-if-an-image-is-blurry
2. https://pysource.com/2019/09/03/detect-when-an-image-is-blurry-opencv-with-python/
3. https://pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
4. See how katna did image blur: https://aloksaan.medium.com/video-key-frame-extraction-with-katna-11971ac45c76