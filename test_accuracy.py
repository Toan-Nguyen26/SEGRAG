from __future__ import division

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
import json
from choiloader import ChoiDataset, collate_fn
from tqdm import tqdm
from argparse import ArgumentParser
from utils import maybe_cuda
import gensim
import utils
import os
import sys
from pathlib2 import Path
from wiki_loader import WikipediaDataSet
import accuracy
from models import naive
from timeit import default_timer as timer
import random
from transformers import AutoTokenizer
# import nltk
# import matplotlib.pyplot as plt

# Ensure you have the necessary NLTK data files
# nltk.download('punkt')

logger  = None
document = None
json_file_path = None
# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3", cache_dir='/path/to/local/cache')
def softmax(x):
    max_each_row = np.max(x, axis=1, keepdims=True)
    exps = np.exp(x - max_each_row)
    sums = np.sum(exps, axis=1, keepdims=True)
    return exps / sums

def getSegmentsFolders(path):

    ret_folders = []
    folders = [o for o in os.listdir(path) if os.path.isdir(os.path.join(path, o))]
    for folder in folders:
        if folder.__contains__("-"):
            ret_folders.append(os.path.join(path,folder))
    return ret_folders

def open_file(file_path):
    # Initialize document as an empty list or dictionary depending on your expected structure
    document = []  # or {} if you expect the document to be a dictionary

    # Check if the JSON file exists
    if os.path.exists(file_path):
        # Load the JSON file
        with open(file_path, 'r', encoding='utf-8') as json_file:
            document = json.load(json_file)
    else:
        print(f"File {file_path} does not exist.")

    return document
def close_file(file_path, document):
    # Save the updated document back to the JSON file
    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(document, json_file, indent=4)
    print(f"Saved the updated document to {file_path}")
    return

def adjust_segment_for_length(output_prob, token_lengths, max_chunk_size, threshold, scaling_factor_base):

    segment_size = sum(token_lengths)
    output_prob = np.array(output_prob)
    if segment_size > max_chunk_size:
        excess_ratio = segment_size / max_chunk_size
        scaling_factor = scaling_factor_base * excess_ratio
        scaled_probs = output_prob * scaling_factor
        adjusted_segments = scaled_probs > threshold
    else:
        adjusted_segments = output_prob > threshold

    return adjusted_segments

def main(args):
    start = timer()

    sys.path.append(str(Path(__file__).parent))

    utils.read_config_file(args.config)
    utils.config.update(args.__dict__)
    logger = utils.setup_logger(__name__, f'{args.dataset}_{args.max_chunk_size}_test_accuracy.log')
    logger.debug('Running with config %s', utils.config)
    print ('Running with threshold: ' + str(args.seg_threshold))
    preds_stats = utils.predictions_analysis()

    if not args.test:
        word2vec = gensim.models.KeyedVectors.load_word2vec_format(utils.config['word2vecfile'], binary=True)
    else:
        word2vec = None

    word2vec_done = timer()
    print('Loading word2vec ellapsed: ' + str(word2vec_done - start) + ' seconds')
    dirname = 'train'
    if args.wiki:
        dataset_folders = [Path(utils.config['snippets']) / dirname]
        if (args.wiki_folder):
            dataset_folders = []
            dataset_folders.append(args.wiki_folder)
        print('running on wikipedia')
    else:
        print('running on RAG dataset')


    with open(args.model, 'rb') as f:
        model = torch.load(f)

    model = maybe_cuda(model)
    model.eval()

    if (args.naive):
        model = naive.create()

    for dataset_path in dataset_folders:

        if (args.bySegLength):
            print('Segment is ',os.path.basename(dataset_path), " :")
        
        if args.dataset:
            print('running on dataset: ', args.dataset)
            # We're handling 512 tokens on avr
            if args.max_chunk_size == 1024:
                json_file_path = f"RAG\data\{args.dataset}\individual_documents"
            # We're handline 1024 on avarage
            else:
                json_file_path = f"RAG\data\{args.dataset}\individual_documents_2048"
            dataset = WikipediaDataSet(dataset_path, word2vec, high_granularity=False, is_json=True, json_data_path=json_file_path)
        elif args.wiki:
            if (args.wiki_folder):
                dataset = WikipediaDataSet(dataset_path, word2vec, folder=True, high_granularity=False)
            else :
                dataset = WikipediaDataSet(dataset_path, word2vec, high_granularity=False)
        else:
            dataset = ChoiDataset(dataset_path , word2vec)

        dl = DataLoader(dataset, batch_size=args.bs, collate_fn=collate_fn, shuffle=False)


        with tqdm(desc='Testing', total=len(dl)) as pbar:
            total_accurate = 0
            total_count = 0
            total_loss = 0
            acc =  accuracy.Accuracy()

            max_chunk_size = args.max_chunk_size
            scaling_factor_base = args.scaling_factor_base
            # Add this line at the start of the main function to open a file for writing segment details
            segment_output_file = open(f'{args.dataset}_segment_output.txt', 'w')
            
            accumulated_probs = []  # Accumulate probabilities for sentences in a segment
            for i, (data, targets, paths) in enumerate(dl):
                if i == args.stop_after:
                    break
                
                pbar.update()
                output = model(data)
                targets_var = Variable(maybe_cuda(torch.cat(targets, 0), args.cuda), requires_grad=False)
                batch_loss = 0
                output_prob = softmax(output.data.cpu().numpy())
                # use for narrativeqa
                output_probability = output_prob[:, 1]
                if args.dataset == 'qasper':
                    seg_threshold =  args.seg_threshold
                else:
                    # Use for qasper
                    seg_threshold = np.percentile(output_probability, 96)
                # print(seg_threshold)
                # output_seg = output_probability > args.seg_threshold
                output_seg = output_probability > seg_threshold
                target_seg = targets_var.data.cpu().numpy()
                batch_accurate = (output_seg == target_seg).sum()
                total_accurate += batch_accurate
                total_count += len(target_seg)
                total_loss += batch_loss
                preds_stats.add(output_seg,target_seg)
                current_target_idx = 0
                for k, t in enumerate(targets):
                    document_sentence_count = len(t)          
                    sentences_length = [s.size()[0] for s in data[k]] if args.calc_word else None
                    to_idx = int(current_target_idx + document_sentence_count)
                    h = output_seg[current_target_idx: to_idx]
                    h = np.append(h, [1])
                    segment_output_file.write(f'original Segments: {h}\n')
                    # =================================================================
                    accumulated_token_lengths = []
                    accumulated_probs = []
                    start_idx = 0  # Track the start of the current segment
                    content = paths[k][1]
                    for i in range(len(h) - 1):
                        sentence = content[i]
                        sentence_length = len(tokenizer.encode(sentence, truncation=True))

                        accumulated_token_lengths.append(sentence_length)
                        accumulated_probs.append(output_probability[i])

                        # If we encounter a `1` (end of a segment)
                        if h[i] == 1:
                            # Check if the accumulated segment is too long
                            if sum(accumulated_token_lengths) > max_chunk_size:
                                # Adjust the segmentation for the current segment
                                adjusted_segments = adjust_segment_for_length(
                                    accumulated_probs, 
                                    accumulated_token_lengths, 
                                    max_chunk_size, 
                                    seg_threshold, 
                                    scaling_factor_base
                                )
                                h[start_idx:i + 1] = np.logical_or(h[start_idx:i + 1], adjusted_segments)
                                segment_output_file.write(f'New Segments: {h}\n')
                            # Reset accumulators for the next segment
                            accumulated_token_lengths = []
                            accumulated_probs = []
                            start_idx = i + 1 

                    # =================================================================
                    t = np.append(t.cpu().numpy(), [1])
                    acc.update(h,t, sentences_length=sentences_length)
                    # Update the segmented_sentences field in the document JSON file
                    if args.dataset:
                        for file_name in os.listdir(json_file_path):
                           if file_name.endswith('.json') and file_name[:-5] == str(paths[k][0]):
                                file_path = os.path.join(json_file_path, file_name)
                                with open(file_path, 'r', encoding='utf-8') as json_file:
                                    data = json.load(json_file)
                                data['segmented_sentences'] = h.tolist()
                                with open(file_path, 'w', encoding='utf-8') as json_file:
                                    json.dump(data, json_file, ensure_ascii=False, indent=4)
                                break

                    # Add this block to log or write out the segment details
                    # Add this block to log or write out the segment details along with the corresponding file path
                    top_5_values = np.sort(output_probability)[-5:]
                    lowest_value = np.min(output_probability)
                    segment_output_file.write(f'Top 5 value: {top_5_values}\n')
                    segment_output_file.write(f'Lowest value: {lowest_value}\n')
                    segment_output_file.write(f'Batch {i}, Document {k}, File Path: {paths[k][0]}:\n')  # Include the file path
                    # segment_output_file.write(f'Target Segments: {t}\n\n')

                    current_target_idx = to_idx

                logger.debug('Batch %s - error %7.4f, Accuracy: %7.4f', i, batch_loss, batch_accurate / len(target_seg))
                pbar.set_description('Testing, Accuracy={:.4}'.format(batch_accurate / len(target_seg)))

            # Make sure to close the file at the end of the process
            segment_output_file.close()

        average_loss = total_loss / len(dl)
        average_accuracy = total_accurate / total_count
        calculated_pk, _ = acc.calc_accuracy()

        logger.info('Finished testing.')
        logger.info('Average loss: %s', average_loss)
        logger.info('Average accuracy: %s', average_accuracy)
        logger.info('Pk: {:.4}.'.format(calculated_pk))
        logger.info('F1: {:.4}.'.format(preds_stats.get_f1()))


        end = timer()
        print ('Seconds to execute to whole flow: ' + str(end - start))



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cuda', help='Use cuda?', action='store_true')
    parser.add_argument('--test', help='Test mode? (e.g fake word2vec)', action='store_true')
    parser.add_argument('--bs', help='Batch size', type=int, default=8)
    parser.add_argument('--model', help='Model to run - will import and run', required=True)
    parser.add_argument('--stop_after', help='Number of batches to stop after', default=None, type=int)
    parser.add_argument('--config', help='Path to config.json', default='config.json')
    parser.add_argument('--wiki', help='Use wikipedia as dataset?', action='store_true')
    parser.add_argument('--bySegLength', help='calc pk on choi by segments length?', action='store_true')
    parser.add_argument('--wiki_folder', help='path to folder which contains wiki documents')
    parser.add_argument('--naive', help='use naive model', action='store_true')
    parser.add_argument('--seg_threshold', help='Threshold for binary classificetion', type=float, default=0.4)
    parser.add_argument('--calc_word', help='Whether to calc P_K by word', action='store_true')
    parser.add_argument('--dataset', help='whenever it is squad or narrative_qa', type=str, default=None)
    parser.add_argument('--is_json', help='Are we loading a json_file for RAG', type=bool, default=False)
    parser.add_argument('--min_range', help='Min percentile', type=bool, default=95)
    parser.add_argument('--max_range', help='Max percentile', type=bool, default=98)
    parser.add_argument('--max_chunk_size', help='Max chunk size', type=int, default=1024)
    parser.add_argument('--scaling_factor_base', help='Scaling factor base', type=float, default=1.2)
    main(parser.parse_args())
