#!/bin/bash
# Download Thunderbird dataset from Zenodo (LogHub)
# WARNING: This dataset is ~2 GB compressed, ~29.6 GB uncompressed

DATA_DIR="./data/thunderbird"
mkdir -p "$DATA_DIR"

if [ ! -f "$DATA_DIR/Thunderbird.log" ]; then
    echo "Downloading Thunderbird dataset from Zenodo (~2 GB)..."
    wget -O "$DATA_DIR/Thunderbird.tar.gz" \
        "https://zenodo.org/records/8196385/files/Thunderbird.tar.gz?download=1"
    echo "Extracting..."
    tar -xzf "$DATA_DIR/Thunderbird.tar.gz" -C "$DATA_DIR"
    rm "$DATA_DIR/Thunderbird.tar.gz"
    echo "Thunderbird dataset ready at $DATA_DIR/Thunderbird.log"
else
    echo "Thunderbird dataset already exists at $DATA_DIR/Thunderbird.log"
fi
