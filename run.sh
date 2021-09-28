#!/usr/bin/env bash

# Example
# ./run.sh image-caption

# Check args
if [ "$#" -ne 1 ]; then
  echo "usage: ./run.sh IMAGE_NAME"
  exit 1
fi

docker run -it \
  --rm \
  --privileged \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v /dev:/dev:rw \
  -v $(pwd)/:/root/:rw \
  -e DISPLAY=$DISPLAY \
  -e "QT_X11_NO_MITSHM=1" \
  --gpus all \
  --name $1 \
  $1 \
  /bin/bash
