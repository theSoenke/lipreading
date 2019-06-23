#!/bin/bash

sh ./scripts/docker/build.sh

if [[ $1 == "" ]]; then
  echo "Data path missing"
  exit
fi

docker run -it --rm --ipc=host --runtime nvidia -v $1:/data -v $(pwd)/data:/output lipreading
