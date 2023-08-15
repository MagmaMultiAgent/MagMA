#!/bin/bash

while getopts ":d:v:h" opt; do
    case $opt in
        device)
            device=$OPTARG
            ;;
        ver)
            version=$OPTARG
            ;;
        h)
            echo "Usage: $0 -device <cpu/gpu> -ver <version>"
            exit 0
            ;;
        \?)
            echo "Invalid option: -$OPTARG. Use '$0 -h' for usage information." >&2
            exit 1
            ;;
    esac
done

if [ -z "$device" ] || [ -z "$version" ]; then
    echo "Error: Missing arguments. Use '$0 -h' for usage information."
    exit 1
fi

if [ "$device" = "CPU" ]; then
    docker run -it -p 8888:8888 -v $(pwd)/src:/usr/src/luxai ranuon98/luxai_cpu:"$version" /bin/bash
else
    docker run -it --gpus all -p 8888:8888 -v $(pwd)/src:/usr/src/luxai ranuon98/luxai_gpu:"$version" /bin/bash
fi