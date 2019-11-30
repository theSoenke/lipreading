# Lipreading


## Setup
- `conda env create -f environment.yml`
- Get access to datatsets https://www.robots.ox.ac.uk/~vgg/data/lip_reading

## LRW

    USER='' PASSWORD='' ./scripts/lrw_download.sh data/datasets/lrw
    python3 preprocess.py lrw --data data/datasets/lrw
    python3 train_words.py --data data/lrw --words 10

## LRS2

    USER='' PASSWORD='' ./scripts/lrs2_download.sh data/datasets/lrs2
    python3 preprocess.py lrs2 --data data/datasets/lrs2
    python3 train_sentences.py --data data/datasets/lrs2 --pretrain

## Train in Docker

    ./scripts/docker/build.sh
    docker run -it --rm --ipc=host -e WANDB_API_KEY=<API_KEY> --runtime nvidia -v /data/lrw:/project/data/datasets/lrw lipreading python train_words.py
