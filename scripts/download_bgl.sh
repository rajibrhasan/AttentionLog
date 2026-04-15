#!/bin/bash
# Download BGL dataset from Zenodo (LogHub)

DATA_DIR="./data/bgl"
mkdir -p "$DATA_DIR"

if [ ! -f "$DATA_DIR/BGL.log" ]; then
    echo "Downloading BGL dataset from Zenodo..."
    wget -O "$DATA_DIR/BGL.tar.gz" \
        "https://zenodo.org/record/3227177/files/BGL.tar.gz"
    echo "Extracting..."
    tar -xzf "$DATA_DIR/BGL.tar.gz" -C "$DATA_DIR"
    rm "$DATA_DIR/BGL.tar.gz"
    echo "BGL dataset ready at $DATA_DIR/BGL.log"
else
    echo "BGL dataset already exists at $DATA_DIR/BGL.log"
fi
