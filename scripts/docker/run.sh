#!/bin/bash

source activate lipreading
python3 train.py --data /data --workers $(nproc) --checkpoint_dir /output/models --tensorboard_logdir /output/tensorboard

