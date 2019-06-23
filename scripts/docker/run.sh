#!/bin/bash

source activate lipreading
python3 train.py --data /data --workers $(nproc) --checkpoint /checkpoints

