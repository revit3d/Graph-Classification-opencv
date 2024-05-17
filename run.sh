#!/bin/bash

docker build -t graph-classification-app:latest .

xhost +local:

docker run --rm -it -e DISPLAY=unix$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix graph-classification-app:latest
