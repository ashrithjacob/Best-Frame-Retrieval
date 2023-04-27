#!/bin/bash

if [ -z "$1" ]
  then
    echo "Please provide the directory name as the first argument"
    exit 1
fi

cd $1

new_filename="FT_LR_001"
rename "s/MS_SSIM_L1/$new_filename/" MS_SSIM_L1-epoch-*.pth


