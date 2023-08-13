#!/bin/bash

if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo "Usage: $0 <os> <device> <version>"
    echo "Arguments:"
    echo "  os      : Operating system (mac/windows)"
    echo "  device  : Device type (CPU/GPU)"
    echo "  version : Docker version tag"
    exit 0
fi

if [ "$#" -ne 3 ]; then
    echo "Error: Incorrect number of arguments. Use '$0 --help' for usage information."
    exit 1
fi

if [ "$1" = "mac" ]; then
    if [ "$2" = "CPU" ]; then
        docker build --platform linux/x86_64 -t ranuon98/luxai_cpu:"$3" -f Dockerfile.CPU .
    else
        docker build --platform linux/x86_64 -t ranuon98/luxai_gpu:"$3" -f Dockerfile.GPU .
    fi
else
    if [ "$2" = "CPU" ]; then
        docker build -t ranuon98/luxai_cpu:"$3" -f Dockerfile.CPU .
    else
        docker build -t ranuon98/luxai_gpu:"$3" -f Dockerfile.GPU .
    fi
fi


