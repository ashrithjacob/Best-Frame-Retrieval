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
https://www.youtube.com/watch?v=nUUqwaxLnWs&ab_channel=DeepLearningAI

Summary:
In batch normalization, the mean and variance are calculated for each batch(mew and sigma) and the data is normalised by these values (x' = (x-mew)/sigma). Here the mew and sigma are from the mean and standard deviation of the batch (all done in model.train() mode).
the data is then scaled by gamma and shifted by beta (y = gamma*x' + beta). This is done in both model.train() and model.eval() modes. Here gamma and beta are learnable parameters of the model.
In testing we use the statistics calculated during training. This is why we need to call model.eval() before testing. If you don’t call model.eval() before testing, the model will use the batch statistics calculated during training, which will not be accurate. This is why you get different results when you call model.eval() before testing.
So how is batch normalisation calculated for the test set?
During training a running mean and std are calculated as follows:
```
running_mean = momentum * running_mean + (1 - momentum) * sample_mean
running_var = momentum * running_var + (1 - momentum) * sample_var
```
During testing, this running mean and std are used to normalise the data (not the batch mean and std). This is why you need to call model.eval() before testing.

Intuition:
We can think of the batch norm layer as a linear transformation of the input data. The linear transformation is parameterized by two learnable parameters, gamma and beta. The linear transformation is applied to the input data in both training and testing. The difference is that in training, the mean and std of the input data are calculated for each batch and used to normalize the data. In testing, the mean and std of the input data are calculated for the entire training set and used to normalize the data. This is why we need to call model.eval() before testing.
Batch norm thus in a sense tries to decouple the input data to a layer from the layers before by simply applying the 'best possible' linear transformation to the input data. This is why batch norm is used in the middle of a network and not at the beginning or end.

## Feature extraction
1. Extract features from the encoder part of the autoencoder
2. https://link.springer.com/chapter/10.1007/978-3-319-51281-5_40