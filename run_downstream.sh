#!/bin/bash
directory=~/Projects/hierarchical-walks/hw/configs
for file in "$directory"/*; do
    echo "$file"
    fileNamePattern="$directory""/$1"
    if [ -f "$file" ] && [[ $file = $fileNamePattern ]]; then
        filename=$(basename -- "$file")
        extension="${filename##*.}"
        filename="${filename%.*}"
        python hw/tools/graph_model_downstream_classification.py --config-name=$filename
    fi
done