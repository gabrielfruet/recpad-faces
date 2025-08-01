#!/bin/bash

RESIZE=30
PCA=false
PCA_VALUE=0.95
NORMALIZATION='z_score'
NR=50
RESULTS="./results"

if ! [ -f "./processed_data/recfaces_${RESIZE}.dat" ]; then
    uv run ./face_processing.py \
        --input_dir "./data"\
        --resize $RESIZE \
        --output_file \
        "./processed_data/recfaces_${RESIZE}.dat"
else
    echo "Processed data already exists for recfaces_${RESIZE}. Skipping..."
fi

for norm in "z_score" "scale_change" "none"; do
    experiment="recfaces_${RESIZE}_${norm}"

    if [ -f "./results/${experiment}.csv" ]; then
        echo "Results already exist for recfaces_${RESIZE}_${norm}. Skipping..."
        continue
    fi
    uv run ./compara_todos.py \
        --data "./processed_data/recfaces_${RESIZE}.dat" \
        --experiment "$experiment" \
        --normalization_method $norm \
        -n $NR

done

