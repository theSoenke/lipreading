#!/bin/bash

set -eo pipefail

data=$1
batch_size=${2:-24}

if [[ $data == "" ]]; then
  echo "Data path missing"
  exit
fi

source activate lipreading
rm -rf data/checkpoints/lrw/lrw_*.ckpt

trap "exit" INT
for seed in {0..30}; do
    printf "\nTrain with seed: ${seed}"
    python3 train.py --data $data --words 10 --seed $seed --batch_size $batch_size
    mv data/checkpoints/lrw/lrw_*.ckpt data/checkpoints/lrw/best_seed_${seed}.pkl

    for query in "-90,-20" "-20,20" "20,90"; do
        printf "\nTrain expert for query: ${query}"
        python3 train.py --data $data --words 10 --seed $seed --query " ${query}" --checkpoint data/checkpoints/lrw/best_seed_${seed}.pkl --batch_size $batch_size
        mv data/checkpoints/lrw/lrw_*.ckpt data/checkpoints/lrw/expert_seed_${seed}_${query}.pkl
    done

    printf "\nTrain attention layer"
    python3 train_attn.py --data $data --words 10 --seed $seed --batch_size $batch_size \
        --checkpoint_left data/checkpoints/lrw/expert_seed_${seed}_-90,-20.pkl \
        --checkpoint_center data/checkpoints/lrw/expert_seed_${seed}_-20,20.pkl \
        --checkpoint_right data/checkpoints/lrw/expert_seed_${seed}_20,90.pkl

    printf "\nTrain early attention layer"
    python3 train_early_attn.py --data $data --words 10 --seed $seed --batch_size $batch_size \
        --checkpoint_left data/checkpoints/lrw/expert_seed_${seed}_-90,-20.pkl \
        --checkpoint_center data/checkpoints/lrw/expert_seed_${seed}_-20,20.pkl \
        --checkpoint_right data/checkpoints/lrw/expert_seed_${seed}_20,90.pkl

    rm -rf data/checkpoints/lrw/lrw_*.ckpt
done
