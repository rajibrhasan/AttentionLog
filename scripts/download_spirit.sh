#!/bin/bash
# Download Spirit dataset
# Option 1: Truncated 5M-line version from Zenodo (~50 MB) — recommended
# Option 2: Full dataset from USENIX CFDR (~864 MB)

DATA_DIR="./data/spirit"
mkdir -p "$DATA_DIR"

if [ ! -f "$DATA_DIR/spirit2.log" ]; then
    echo "Downloading truncated Spirit dataset from Zenodo (~50 MB)..."
    wget -O "$DATA_DIR/spirit2_5m.tar.gz" \
        "https://zenodo.org/records/7851024/files/spirit2_5m.tar.gz?download=1"
    echo "Extracting..."
    tar -xzf "$DATA_DIR/spirit2_5m.tar.gz" -C "$DATA_DIR"
    rm "$DATA_DIR/spirit2_5m.tar.gz"
    echo "Spirit dataset ready at $DATA_DIR/"
else
    echo "Spirit dataset already exists at $DATA_DIR/spirit2.log"
fi

echo ""
echo "For the full Spirit dataset (~37 GB uncompressed), download from USENIX CFDR:"
echo "  wget -O $DATA_DIR/spirit2.gz http://0b4af6cdc2f0c5998459-c0245c5c937c5dedcca3f1764ecc9b2f.r43.cf2.rackcdn.com/hpc4/spirit2.gz"
echo "  gunzip $DATA_DIR/spirit2.gz"
