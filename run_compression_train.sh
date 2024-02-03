#!/bin/bash
directory=~/Projects/hierarchical-walks/hw/configs
for file in "$directory"/*; do
    echo "$file"
    fileNamePattern="$directory""/$1"
    if [ -f "$file" ] && [[ $file = $fileNamePattern ]]; then
        filename=$(basename -- "$file")
        extension="${filename##*.}"
        filename="${filename%.*}"
        python hw/tools/train.py --config-name=$filename \
    ++datamodule.additional_parameters.num_compressions=2 \
    ++datamodule.additional_parameters.compression_selection_ratio=.5
    fi
done