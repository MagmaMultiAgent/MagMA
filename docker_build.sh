#!/bin/bash

usage() { echo "Usage: $0 [-o <win|mac>] [-d <cpu|gpu>] [-v <version|latest>]" 1>&2; exit 1; }

while getopts "o:d:v:" opt; do
    case $opt in
        o)
            os=${OPTARG}
            ;;
        d)
            device=${OPTARG}
            ;;
        v)
            version=${OPTARG}
            ;;
        :) 
            echo "Option -$OPTARG requires an argument." >&2;;
        *)
            usage
            ;;
    esac
done

if [ -z "$os" ] || [ -z "$device" ] || [ -z "$version" ]; then
    usage
fi

if [ "$os" = "mac" ]; then
    platform="--platform linux/x86_64"
else
    platform=""
fi

if [ "$device" = "cpu" ]; then
    docker build $platform -t ranuon98/luxai_cpu:"$version" -f Dockerfile.CPU .
else
    docker build $platform -t ranuon98/luxai_gpu:"$version" -f Dockerfile.GPU .
fi
