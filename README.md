# Ensemble-Based Multi-View Lipreading

## Preprocess

    ./scripts/dlib_model.sh
    python3 preprocess.py lrw --data data/datasets/lrw --words 10

## Train

    python3 train.py --data data/preprocessed/lrw.h5

### Docker

    ./scripts/docker/train.sh /data/datasets/lrw

