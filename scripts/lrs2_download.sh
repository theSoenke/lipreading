#!/bin/bash

urls=(
    "http://www.robots.ox.ac.uk/~vgg/data/lip_reading/data2/lrs2_v1_partaa"
    "http://www.robots.ox.ac.uk/~vgg/data/lip_reading/data2/lrs2_v1_partab"
    "http://www.robots.ox.ac.uk/~vgg/data/lip_reading/data2/lrs2_v1_partac"
    "http://www.robots.ox.ac.uk/~vgg/data/lip_reading/data2/lrs2_v1_partad"
    "http://www.robots.ox.ac.uk/~vgg/data/lip_reading/data2/lrs2_v1_partae"
    "http://www.robots.ox.ac.uk/~vgg/data/lip_reading/data2/pretrain.txt"
    "http://www.robots.ox.ac.uk/~vgg/data/lip_reading/data2/train.txt"
    "http://www.robots.ox.ac.uk/~vgg/data/lip_reading/data2/val.txt"
    "http://www.robots.ox.ac.uk/~vgg/data/lip_reading/data2/test.txt"
)

dir=${1:-"data/datasets/lrs2"}
mkdir -p $dir
for url in "${urls[@]}"; do
    wget --user $USER --password $PASSWORD -P $dir $url
done

cat $dir/lrs2_v1_parta* > $dir/lrs2_v1.tar
echo "Extracting data"
tar xf $dir/lrs2_v1.tar
mkdir -p data/datasets
ln -sf $dir data/datasets/lrs2
