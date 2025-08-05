#!/bin/bash

RESIZE=30
PCA=false
NR=50
RESULTS="./results"

set -e
#################

data="./processed_data/recfaces_${RESIZE}.dat"
if ! [ -f "$data" ]; then
    uv run ./face_processing.py \
        --input_dir "./data"\
        --resize $RESIZE \
        --output_file \
        "$data"
else
    echo "Processed data already exists for $data. Skipping..."
fi

experiment="recfaces_${RESIZE}"

if [ -f "./results/${experiment}.csv" ]; then
    echo "Results already exist for $experiment Skipping..."
else

    uv run ./compara_todos.py \
        --data "$data" \
        --experiment "$experiment" \
        --normalization_method "z_score" \
        -n $NR
fi
#################

data="./processed_data/recfaces_${RESIZE}_pca_full.dat"
if ! [ -f "$data" ]; then
    echo "Processing data with PCA..."
    uv run ./face_processing.py \
        --input_dir "./data"\
        --resize $RESIZE \
        --output_file "$data" \
        -pca -1
else
    echo "Processed data already exists for $data. Skipping..."
fi

experiment="recfaces_${RESIZE}_pca_full"

if [ -f "./results/${experiment}.csv" ]; then
    echo "Results already exist for $experiment Skipping..."
else
    uv run ./compara_todos.py \
        --data "$data" \
        --experiment "$experiment" \
        --normalization_method "none" \
        -n $NR
fi
#################

data="./processed_data/recfaces_${RESIZE}_pca_1.dat"
if ! [ -f "$data" ]; then
    echo "Processing data with PCA..."
    uv run ./face_processing.py \
        --input_dir "./data"\
        --resize $RESIZE \
        --output_file "$data" \
        -pca 1
else
    echo "Processed data already exists for $data. Skipping..."
fi

experiment="recfaces_${RESIZE}_pca_1"

if [ -f "./results/${experiment}.csv" ]; then
    echo "Results already exist for $experiment Skipping..."
else
    uv run ./compara_todos.py \
        --data "$data" \
        --experiment "$experiment" \
        --normalization_method "z_score" \
        -n $NR
fi

#################

data="./processed_data/recfaces_${RESIZE}_pca_98.dat"
if ! [ -f "$data" ]; then
    echo "Processing data with PCA..."
    uv run ./face_processing.py \
        --input_dir "./data"\
        --resize $RESIZE \
        --output_file "$data" \
        -pca 0.98
else
    echo "Processed data already exists for $data. Skipping..."
fi


experiment="recfaces_${RESIZE}_pca_98"

if [ -f "./results/${experiment}.csv" ]; then
    echo "Results already exist for $experiment Skipping..."
else
    uv run ./compara_todos.py \
        --data "$data" \
        --experiment "$experiment" \
        --normalization_method "z_score" \
        -n $NR
fi

############### 

data="./processed_data/recfaces_${RESIZE}_pca_98_box_cox.dat"
if ! [ -f "$data" ]; then
    echo "Processing data with PCA..."
    uv run ./face_processing.py \
        --input_dir "./data"\
        --resize $RESIZE \
        --output_file "$data" \
        -pca 0.98 --box_cox
else
    echo "Processed data already exists for $data. Skipping..."
fi

experiment="recfaces_${RESIZE}_pca_98_box_cox"

if [ -f "./results/${experiment}.csv" ]; then
    echo "Results already exist for $experiment Skipping..."
else
    uv run ./compara_todos.py \
        --data "$data" \
        --experiment "$experiment" \
        --normalization_method "z_score" \
        -n $NR
fi
