#!/bin/bash

urls=(
    "https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data1/lrw-v1-partaa"
    "https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data1/lrw-v1-partab"
    "https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data1/lrw-v1-partac"
    "https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data1/lrw-v1-partad"
    "https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data1/lrw-v1-partae"
    "https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data1/lrw-v1-partaf"
    "https://thor.robots.ox.ac.uk/~vgg/data/lip_reading/data1/lrw-v1-partag"
)

dir=${1:-"data/datasets/lrw"}
mkdir -p $dir
for url in "${urls[@]}"; do
    wget --user $USER --password $PASSWORD -P $dir $url
done

cat $dir/lrw-v1-parta* > $dir/lrw-v1.tar
tar xf $dir/lrs2_v1.tar -O $dir
mkdir -p data/datasets
ln -sf $dir/lipread_mp4 data/datasets/lrw
