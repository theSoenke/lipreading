#!/bin/bash

source activate lipreading
python3 train.py --data /data --checkpoint_dir /output/checkpoints
