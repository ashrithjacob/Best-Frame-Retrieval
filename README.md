# Autoencoder to extract key clips from video
This repo is based on:\
Junyu Chen et al 2021 J. Phys.: Conf. Ser. 2025 012018

## Architecture
The architecture is as follows:
![Architecture](https://github.com/ashrithjacob/Best-Frame-Retrieval/blob/main/images/architecture.png?raw=true)

## TODO:
1. Load Caltech-256 - DONE
2. Train-Test split - DONE
3. Define loss function (MS-SSIM + L1) - DONE
4. Train autoencoder - DONE (20 epochs)
5. Try to remove relu from last part of encoder
6. Use loss params as in paper (alphs =0.84 , see G value as well)
7. Add batch norm. See paper again
8. Compare leaky relu vs relu
9. SGD >>> Adam
