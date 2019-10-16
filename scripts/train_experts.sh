#!/bin/bash

data=$1
batch_size=${2:-64}

source activate lipreading
rm -rf data/checkpoints/lrw/lrw_*.ckpt
for i in {0..10}; do
    python3 train.py --data $data --words 10 --seed $i --batch_size $batch_size
    mv data/checkpoints/lrw/lrw_*.ckpt data/checkpoints/lrw/best_seed_${i}.pkl

    for query in "-90,-20" "-20,20" "20,90"; do
        python3 train.py --data $data --words 10 --seed $i --query $query --checkpoint data/checkpoints/lrw/best_seed_${i}.pkl --batch_size $batch_size
        mv data/checkpoints/lrw/lrw_*.ckpt data/checkpoints/lrw/expert_seed_${i}_${query}.pkl
    done

    python3 train_attn.py --data $data --words 10 --seed $i --batch_size $batch_size \
        --checkpoint_left data/checkpoints/lrw/expert_seed_${i}_-90,-20.pkl \
        --checkpoint_center data/checkpoints/lrw/expert_seed_${i}_-20,20.pkl \
        --checkpoint_right data/checkpoints/lrw/expert_seed_${i}_20,90.pkl
done
