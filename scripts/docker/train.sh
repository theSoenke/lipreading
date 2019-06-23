#!/bin/bash

if [[ "$(docker images -q lipreading:latest 2> /dev/null)" == "" ]]; then
  sh ./scripts/docker/build.sh
fi

if [[ $1 == "" ]]; then
  echo "Data path missing"
  exit
fi

if [[ $2 == "" ]]; then
  echo "Checkpoint path missing"
  exit
fi

docker run -it --rm --ipc=host --runtime nvidia -v $1:/data -v $2:/checkpoints lipreading
