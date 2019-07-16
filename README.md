# Ensemble-Based Multi-View Lipreading

## Preprocess

    ./scripts/dlib_model.sh
    python3 preprocess.py lrw --data data/datasets/lrw --words 10

## Train

    python3 train.py --hdf5 data/preprocessed/lrw_10.h5 --words 10

### Docker

    ./scripts/docker/train.sh /data/datasets/lrw

