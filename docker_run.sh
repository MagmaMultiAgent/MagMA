#!/bin/bash

if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo "Usage: $0 <device> <version>"
    echo "Arguments:"
    echo "  device  : Device type (CPU/GPU)"
    echo "  version : Docker version tag"
    exit 0
fi

if [ "$#" -ne 2 ]; then
    echo "Error: Incorrect number of arguments. Use '$0 --help' for usage information."
    exit 1
fi

if [ "$1" = "CPU" ]; then
    docker run -it -p 8888:8888 -v $(pwd)/src:/usr/src/luxai ranuon98/luxai_cpu:"$2" /bin/bash
else
    docker run -it --gpus all -p 8888:8888 -v $(pwd)/src:/usr/src/luxai ranuon98/luxai_gpu:"$2" /bin/bash
fi
