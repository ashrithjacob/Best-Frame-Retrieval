# Autoencoder to extract key clips from video
This repo is based on:\
Junyu Chen et al 2021 J. Phys.: Conf. Ser. 2025 012018

## Architecture
The architecture is as follows:
![Architecture](https://github.com/ashrithjacob/Best-Frame-Retrieval/blob/main/images/architecture.png?raw=true)

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
Summary:
In training we calculate batch statistics (mean and variance) and use them to normalize the data. In testing we use the statistics calculated during training. This is why we need to call model.eval() before testing. If you don’t call model.eval() before testing, the model will use the batch statistics calculated during training, which will not be accurate. This is why you get different results when you call model.eval() before testing.
So how is batch normalisation calculated for the test set?
During training a running mean and std are calculated as follows:
```
running_mean = momentum * running_mean + (1 - momentum) * sample_mean
running_var = momentum * running_var + (1 - momentum) * sample_var
```
During testing, this running mean and std are used to normalise the data (not the batch mean and std). This is why you need to call model.eval() before testing.