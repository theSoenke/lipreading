#!/bin/bash

urls=(
    "http://www.robots.ox.ac.uk/~vgg/data/lip_reading/data1/lrw-v1-partaa"
    "http://www.robots.ox.ac.uk/~vgg/data/lip_reading/data1/lrw-v1-partab"
    "http://www.robots.ox.ac.uk/~vgg/data/lip_reading/data1/lrw-v1-partac"
    "http://www.robots.ox.ac.uk/~vgg/data/lip_reading/data1/lrw-v1-partad"
    "http://www.robots.ox.ac.uk/~vgg/data/lip_reading/data1/lrw-v1-partae"
    "http://www.robots.ox.ac.uk/~vgg/data/lip_reading/data1/lrw-v1-partaf"
    "http://www.robots.ox.ac.uk/~vgg/data/lip_reading/data1/lrw-v1-partag"
)

dir=data/datasets/lrw
mkdir -p $dir
for url in "${urls[@]}"; do
    wget --user $USER --password $PASSWORD -P $dir $url
done

cat $dir/lrw-v1-parta* > $dir/lrw-v1.tar
