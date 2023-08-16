#!/bin/bash

usage() { echo "Usage: $0 [-d <cpu|gpu>] [-v <version>]" 1>&2; exit 1; }

while getopts ":d:v:h" opt; do
    case $opt in
        d)
            device=$OPTARG
            ;;
        v)
            version=$OPTARG
            ;;
        :) 
            echo "Option -$OPTARG requires an argument." >&2;;
        *)
            usage
            ;;
    esac
done

if [ -z "$device" ] || [ -z "$version" ]; then
    usage
fi

if [ "$device" = "CPU" ]; then
    docker run -it -p 8888:8888 -v $(pwd)/src:/usr/src/luxai ranuon98/luxai_cpu:"$version" /bin/bash
else
    docker run -it --gpus all -p 8888:8888 -v $(pwd)/src:/usr/src/luxai ranuon98/luxai_gpu:"$version" /bin/bash
fi