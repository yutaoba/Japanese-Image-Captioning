#!/usr/bin/env bash

# Example
# ./build.sh Dockerfile image-caption

# Check args
if [ "$#" -ne 2 ]; then
  echo "usage: ./build.sh DOCKERFILE_NAME IMAGE_NAME"
  exit 1
fi

docker build --tag $2 --file $1 .
