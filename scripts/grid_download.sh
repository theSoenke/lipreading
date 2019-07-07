#!/bin/bash

mkdir -p data/datasets/grid/{videos,aligns}
for i in {1..34}; do
    if [[ $i == 21 ]]; then
        continue
    fi

    wget http://spandh.dcs.shef.ac.uk/gridcorpus/s$i/video/s$i.mpg_vcd.zip
    unzip s$i.mpg_vcd.zip
    mv s$i data/datasets/grid/videos/s$i
    rm s$i.mpg_vcd.zip

    wget http://spandh.dcs.shef.ac.uk/gridcorpus/s$i/align/s$i.tar
    tar -xf s$i.tar
    mv align data/datasets/grid/aligns/s$i
    rm s$i.tar
done
