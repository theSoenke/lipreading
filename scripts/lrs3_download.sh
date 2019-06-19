#!/bin/bash

urls=(
    "http://www.robots.ox.ac.uk/~vgg/data/lip_reading/data3/lrs3_pretrain_partaa"
    "http://www.robots.ox.ac.uk/~vgg/data/lip_reading/data3/lrs3_pretrain_partab"
    "http://www.robots.ox.ac.uk/~vgg/data/lip_reading/data3/lrs3_pretrain_partac"
    "http://www.robots.ox.ac.uk/~vgg/data/lip_reading/data3/lrs3_pretrain_partad"
    "http://www.robots.ox.ac.uk/~vgg/data/lip_reading/data3/lrs3_pretrain_partae"
    "http://www.robots.ox.ac.uk/~vgg/data/lip_reading/data3/lrs3_pretrain_partaf"
    "http://www.robots.ox.ac.uk/~vgg/data/lip_reading/data3/lrs3_pretrain_partag"
    "http://www.robots.ox.ac.uk/~vgg/data/lip_reading/data3/lrs3_trainval.zip"
    "http://www.robots.ox.ac.uk/~vgg/data/lip_reading/data3/lrs3_test_v0.4.zip"
)

dir=data/datasets/lrs3
mkdir -p dir
for url in "${urls[@]}"; do
    wget --user $USER --password $PASSWORD -P $dir $url
done

cat $dir/lrs3_pretrain_part* > $dir/lrs3_pretrain.zip
