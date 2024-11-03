#!/bin/bash

# Download directories vars
root_dl="/home/ayshrv/turbo_datasets/kinetics700-2020"
root_dl_targz="/home/ayshrv/turbo_datasets/kinetics700-2020_targz"

# Make root directories
[ ! -d $root_dl ] && mkdir $root_dl
[ ! -d $root_dl_targz ] && mkdir $root_dl_targz

# # Download validation tars, will resume
# curr_dl=${root_dl_targz}/val
# [ ! -d $curr_dl ] && mkdir -p $curr_dl
# wget -c -i https://s3.amazonaws.com/kinetics/700_2020/val/k700_2020_val_path.txt -P $curr_dl

# Download train tars, will resume
curr_dl=${root_dl_targz}/train
[ ! -d $curr_dl ] && mkdir -p $curr_dl
wget -c -i https://s3.amazonaws.com/kinetics/700_2020/train/k700_2020_train_path.txt -P $curr_dl

# # Download annotations csv files
# curr_dl=${root_dl}/annotations
# [ ! -d $curr_dl ] && mkdir -p $curr_dl
# wget -c https://s3.amazonaws.com/kinetics/700_2020/annotations/train.csv -P $curr_dl
# wget -c https://s3.s.com/kinetics/700_2020/annotations/val.csv -P $curr_dl
# wget -c https://s3.amazonaws.com/kinetics/700_2020/annotations/test.csv -P $curr_dl

# # Download readme
# wget -c http://s3.amazonaws.com/kinetics/700_2020/K700_2020_readme.txt -P $root_dl

# # Downloads complete
# echo -e "\nDownloads complete! Now run extractor, k700_2020_extractor.sh"