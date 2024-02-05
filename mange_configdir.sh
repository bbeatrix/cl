#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

directory=$1

# Array to keep track of processed gin files
processed_gin_files=()

# Iterate over gin_config_files in the specified directory
for gin_config in "$directory"/*.gin; do
    if [ -f "$gin_config" ]; then
        # Check if the gin file has already been processed
        if [[ ! " ${processed_gin_files[@]} " =~ " ${gin_config} " ]]; then
            echo "Processing: ${gin_config} ..."
            
            # Run main.py with the specified gin file
            nohup python main.py --gin_file="${gin_config}" > "${gin_config%.*}_c.out" 2>&1
            
            # Add the processed gin file to the list
            processed_gin_files+=("${gin_config}")
        else
            echo "Already processed: ${gin_config}, skipping..."
        fi
    fi
done