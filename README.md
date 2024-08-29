# Text Segmentation as a Supervised Learning Task

This repository contains code and supplementary materials which are required to train and evaluate a model as described in the paper [Text Segmentation as a Supervised Learning Task](https://arxiv.org/abs/1803.09337)

## Downalod required resources

wiki-727K, wiki-50 datasets:

> https://www.dropbox.com/sh/k3jh0fjbyr0gw0a/AADzAd9SDTrBnvs1qLCJY5cza?dl=0

word2vec:

> https://drive.google.com/a/audioburst.com/uc?export=download&confirm=zrin&id=0B7XkCwpI5KDYNlNUTTlSS21pQmM

Fill relevant paths in configgenerator.py, and execute the script (git repository includes Choi dataset)

## Creating an environment:

    conda create -n textseg python=3.10
    source activate textseg

Version python=2.9 was used in the original paper but 3.10 works just fine

    pip install -r requirements.txt
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

Installing torch here require a specific version of CUDA (The one above use CUDA 12.4). So check your CUDA version beforehand

## How to run training process?

    python run.py --help

Example:

    python run.py --cuda --model max_sentence_embedding --wiki --epochs 20 --early_stops 3

## How to evaluate trained model (on wiki-727/choi dataset)?

    python test_accuracy.py  --help

Example:

    python test_accuracy.py --cuda --model <path_to_model> --wiki

## How to create a new wikipedia dataset:

    python wiki_processor.py --input <input> --temp <temp_files_folder> --output <output_folder> --train <ratio> --test <ratio>

Input is the full path to the wikipedia dump, temp is the path to the temporary files folder, and output is the path to the newly generated wikipedia dataset.

Wikipedia dump can be downloaded from following url:

> https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
