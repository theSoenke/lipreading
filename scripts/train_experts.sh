#!/bin/bash

data=$1

source activate lipreading
rm -rf data/checkpoints/lrw/lrw_*.ckpt
for i in {0..10}; do
    python3 train.py --data $data --words  10 --seed $i
    mv data/checkpoints/lrw/lrw_*.ckpt data/checkpoints/lrw/best_seed_${i}.pkl

    for query in "-90,-20" "-20,20" "20,90"; do
        python3 train.py --data $data --words  10 --seed $i --query $query --checkpoint data/checkpoints/lrw/best_seed_${i}.pkl
        rm -rf data/checkpoints/lrw/lrw_*.ckpt
    done
done
