# Ensemble-Based Multi-View Lipreading


## Setup
- Install [ctcdecode](https://github.com/parlance/ctcdecode)
- `conda env create -f environment.yml`

## Preprocess

    ./scripts/dlib_model.sh
    python3 preprocess.py lrw --data data/datasets/lrw

## Train

    python3 train.py --hdf5 data/preprocessed/lrw_10.h5 --words 10

### Docker

    ./scripts/docker/train.sh /data/datasets/lrw

