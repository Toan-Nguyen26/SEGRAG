#!/bin/bash

# chmod +x run.sh

# ./run.sh
# # Run the script with 512 chunking
# python embedding.py --dataset narrativeqa --chunk_type 512

# # Run the script with seg chunking
# python embedding.py --dataset narrativeqa --chunk_type 1024

# # Run the script with segclus chunking
# python embedding.py --dataset narrativeqa --chunk_type 2048


# cd RAG

python qa.py --dataset narrativeqa --top_k 5 --chunk_type seg

python qa.py --dataset narrativeqa --top_k 5 --chunk_type segclus

python qa.py --dataset quality --top_k 5 --chunk_type seg

python qa.py --dataset quality --top_k 5 --chunk_type segclus

python qa.py --dataset qasper --top_k 5 --chunk_type seg

python qa.py --dataset qasper --top_k 5 --chunk_type segclus

# python test_accuracy.py --cuda --model checkpoints_new/best_model.t7 --wiki --dataset quality --is_json True --bs 1 --max_chunk_size 1024 --seg_threshold 0.3 

# python test_accuracy.py --cuda --model checkpoints_new/best_model.t7 --wiki --dataset quality --is_json True --bs 1 --max_chunk_size 2048 --seg_threshold 0.3 


# python test_accuracy.py --cuda --model checkpoints_new/best_model.t7 --wiki --dataset narrativeqa --is_json True --bs 1 --max_chunk_size 1024 --seg_threshold 0.3 

# python test_accuracy.py --cuda --model checkpoints_new/best_model.t7 --wiki --dataset narrativeqa --is_json True --bs 1 --max_chunk_size 2048 --seg_threshold 0.3 

# python test_accuracy.py --cuda --model checkpoints_new/best_model.t7 --wiki --dataset qasper --is_json True --bs 1 --max_chunk_size 1024 --seg_threshold 0.3 

# python test_accuracy.py --cuda --model checkpoints_new/best_model.t7 --wiki --dataset qasper --is_json True --bs 1 --max_chunk_size 2048 --seg_threshold 0.3 

# cd RAG

# python qa.py --dataset quality --top_k 4 --chunk_type 256
