# Ensemble-Based Multi-View Lipreading


## Setup
- Install [ctcdecode](https://github.com/parlance/ctcdecode)
- `conda env create -f environment.yml`

## LRW

    USER=$USER PASSWORD=$PASS ./scripts/lrw_download.sh data/datasets/lrw
    python3 preprocess.py lrw --data data/datasets/lrw
    python3 train.py --data data/lrw --words 10

## Train LRS2

    USER=$USER PASSWORD=$PASS ./scripts/lrs2_download.sh data/datasets/lrs2
    python3 preprocess.py lrs2 --data data/datasets/lrs2
    python3 train_ctc.py --data data/datasets/lrs2

## Train in Docker

    ./scripts/docker/train.sh "<command>"

### Sample

    ./scripts/docker/train.sh "python train.py --words 10"

