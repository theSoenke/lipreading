#!/bin/bash

source activate lipreading
python3 train.py --data /data --checkpoint_dir /output/models --tensorboard_logdir /output/tensorboard

