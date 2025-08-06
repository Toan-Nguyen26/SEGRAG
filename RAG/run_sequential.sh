#!/bin/bash

# Run the script with 256 chunking
python embedding.py --dataset narrativeqa --chunk_type 256

# Run the script with 512 chunking
python embedding.py --dataset narrativeqa --chunk_type 512

# Run the script with seg chunking
python embedding.py --dataset quality --chunk_type 256

# Run the script with segclus chunking
python embedding.py --dataset quality --chunk_type 512
