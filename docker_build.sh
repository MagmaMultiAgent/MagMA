#!/bin/bash

while getopts ":os:device:ver:h" opt; do
    case $opt in
        os)
            os=$OPTARG
            ;;
        device)
            device=$OPTARG
            ;;
        ver)
            version=$OPTARG
            ;;
        h)
            echo "Usage: $0 -os <win/mac> -device <cpu/gpu> -ver <version>"
            exit 0
            ;;
        \?)
            echo "Invalid option: -$OPTARG. Use '$0 -h' for usage information." >&2
            exit 1
            ;;
    esac
done

if [ -z "$os" ] || [ -z "$device" ] || [ -z "$version" ]; then
    echo "Error: Missing arguments. Use '$0 -h' for usage information."
    exit 1
fi

if [ "$os" = "mac" ]; then
    platform="--platform linux/x86_64"
else
    platform=""
fi

if [ "$device" = "CPU" ]; then
    docker build $platform -t ranuon98/luxai_cpu:"$version" -f Dockerfile.CPU .
else
    docker build $platform -t ranuon98/luxai_gpu:"$version" -f Dockerfile.GPU .
fi
