#!/bin/bash
directory=~/Projects/hierarchical-walks/hw/configs
for file in "$directory"/*; do
    echo "$file"
    fileNamePattern="$directory""/cora_adv_*"
    if [ -f "$file" ] && [[ $file = $fileNamePattern ]]; then
        filename=$(basename -- "$file")
        extension="${filename##*.}"
        filename="${filename%.*}"
        python hw/tools/train.py --config-name=$filename
    fi
done