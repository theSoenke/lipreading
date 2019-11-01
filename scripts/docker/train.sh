#!/bin/bash

sh ./scripts/docker/build.sh

docker run -it --rm --ipc=host --runtime nvidia -v $(pwd)/data:/data lipreading conda run -n lipreading $1
