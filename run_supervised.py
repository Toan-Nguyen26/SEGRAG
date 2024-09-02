import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import multiprocessing
from choiloader import ChoiDataset, collate_fn_2_head
from tqdm import tqdm
from argparse import ArgumentParser
from utils import maybe_cuda
import gensim
import utils
from tensorboard_logger import configure, log_value
import os
import sys
from pathlib2 import Path
from wiki_loader import WikipediaDataSet
import accuracy
import numpy as np
from termcolor import colored
import logging

torch.multiprocessing.set_sharing_strategy('file_system')

preds_stats = utils.predictions_analysis()

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("output.log"),
                        logging.StreamHandler()  # This will output to the console as well
                    ])

logger = logging.getLogger()


def softmax(x):
    max_each_row = np.max(x, axis=1, keepdims=True)
    exps = np.exp(x - max_each_row)
    sums = np.sum(exps, axis=1, keepdims=True)
    return exps / sums


def import_model(model_name):
    module = __import__('models.' + model_name, fromlist=['models'])
    return module.create()


class Accuracies(object):
    def __init__(self):
        self.thresholds = np.arange(0, 1, 0.05)
        self.accuracies = {k: accuracy.Accuracy() for k in self.thresholds}

    def update(self, output_np, targets_np):
        current_idx = 0
        for k, t in enumerate(targets_np):
            document_sentence_count = len(t)
            to_idx = int(current_idx + document_sentence_count)

            for threshold in self.thresholds:
                output = ((output_np[current_idx: to_idx, :])[:, 1] > threshold)
                h = np.append(output, [1])
                tt = np.append(t, [1])

                self.accuracies[threshold].update(h, tt)

            current_idx = to_idx

    def calc_accuracy(self):
        min_pk = np.inf
        min_threshold = None
        min_epoch_windiff = None
        for threshold in self.thresholds:
            epoch_pk, epoch_windiff = self.accuracies[threshold].calc_accuracy()
            if epoch_pk < min_pk:
                min_pk = epoch_pk
                min_threshold = threshold
                min_epoch_windiff = epoch_windiff

        return min_pk, min_epoch_windiff, min_threshold


def train(model, args, epoch, dataset, logger, optimizer):
    model.train()
    total_loss = float(0)
    with tqdm(desc='Training', total=len(dataset)) as pbar:
        for i, (data, segment_targets , document_targets, paths) in enumerate(dataset):
            if True:
                pbar.update()
                model.zero_grad()
                try:

                    sentence_preds, document_preds = model(data)
            
                    seg_tar = Variable(maybe_cuda(torch.cat(segment_targets, 0), args.cuda), requires_grad=False)
                    doc_tar = Variable(maybe_cuda(torch.cat(document_targets, 0), args.cuda), requires_grad=False)
                    # print(output.shape, target_var.shape)
                    loss_sentence = model.criterion_sentence(sentence_preds, seg_tar)
                    loss_document = model.criterion_document(document_preds, doc_tar)
                    loss = loss_sentence + loss_document
                     # Check for NaN loss
                    if torch.isnan(loss):
                        logger.error(f"Loss is NaN at batch {i}, skipping the batch. Paths: {paths}")
                        continue
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    pbar.set_description('Training, loss={:.4}'.format(loss.item()))
                except RuntimeError as e:
                    if 'Length of all samples has to be greater than 0' in str(e):
                        logger.warning(f"Skipping batch due to empty sequence. Error: {str(e)}")
                        continue
                    else:
                        raise e  

            # except Exception as e:
            #     logger.info('Exception "%s" in batch %s', e, i)
            #     logger.debug('Exception while handling batch with file paths: %s', paths, exc_info=True)
            #     pass

    total_loss = total_loss / len(dataset)
    logger.debug('Training Epoch: {}, Loss: {:.4}.'.format(epoch + 1, total_loss))
    log_value('Training Loss', total_loss, epoch + 1)


def validate(model, args, epoch, dataset, logger):
    model.eval()
    with tqdm(desc='Validating', total=len(dataset)) as pbar:
        acc_sentence = Accuracies()
        acc_document = Accuracies()

        for i, (data, segment_targets, document_targets, paths) in enumerate(dataset):
            pbar.update()

            try:
                # Forward pass: Get predictions for both tasks
                sentence_output, document_output = model(data)

                # Apply softmax to both outputs
                sentence_output_softmax = F.softmax(sentence_output, 1)
                document_output_softmax = F.softmax(document_output, 1)

                # Convert targets to torch Variables
                segment_targets_var = Variable(maybe_cuda(torch.cat(segment_targets, 0), args.cuda), requires_grad=False)
                document_targets_var = Variable(maybe_cuda(torch.cat(document_targets, 0), args.cuda), requires_grad=False)

                # Convert model outputs and targets to numpy arrays for metric computation
                output_seg = sentence_output.data.cpu().numpy().argmax(axis=1)
                output_doc = document_output.data.cpu().numpy().argmax(axis=1)

                target_seg = segment_targets_var.data.cpu().numpy()
                target_doc = document_targets_var.data.cpu().numpy()

                # Update accuracy and other metrics for sentence segmentation
                acc_sentence.update(sentence_output_softmax.data.cpu().numpy(), segment_targets)
                preds_stats.add(output_seg, target_seg)

                # Update accuracy and other metrics for document segmentation
                acc_document.update(document_output_softmax.data.cpu().numpy(), document_targets)
                preds_stats.add(output_doc, target_doc)

            except RuntimeError as e:
                if 'Length of all samples has to be greater than 0' in str(e):
                    logger.warning(f"Skipping batch due to empty sequence. Error: {str(e)}")
                    continue
                else:
                    raise e  

        # Compute accuracy, Pk, Windiff, and F1 for both sentence and document tasks
        epoch_pk_sentence, epoch_windiff_sentence, threshold_sentence = acc_sentence.calc_accuracy()
        epoch_pk_document, epoch_windiff_document, threshold_document = acc_document.calc_accuracy()

        logger.info('Validating Epoch: {}, sentence accuracy: {:.4}, sentence Pk: {:.4}, sentence Windiff: {:.4}, sentence F1: {:.4}'.format(
            epoch + 1,
            preds_stats.get_accuracy(),  
            epoch_pk_sentence,
            epoch_windiff_sentence,
            preds_stats.get_f1()  
        ))

        logger.info('Validating Epoch: {}, document accuracy: {:.4}, document Pk: {:.4}, document Windiff: {:.4}, document F1: {:.4}'.format(
            epoch + 1,
            preds_stats.get_accuracy(), 
            epoch_pk_document,
            epoch_windiff_document,
            preds_stats.get_f1() 
        ))

        preds_stats.reset()

        return epoch_pk_sentence, threshold_sentence, epoch_pk_document, threshold_document


def test(model, args, epoch, dataset, logger, threshold):
    model.eval()
    with tqdm(desc='Testing', total=len(dataset)) as pbar:
        # Add this line at the start of the main function to open a file for writing segment details
        segment_output_file = open('segment_output.txt', 'w')
        acc_sentence = accuracy.Accuracy()
        acc_document = accuracy.Accuracy()

        for i, (data, segment_targets, document_targets, paths) in enumerate(dataset):
            pbar.update()

            try:
                # Forward pass: Get predictions for both tasks
                sentence_output, document_output = model(data)

                # Apply softmax to both outputs
                sentence_output_softmax = F.softmax(sentence_output, 1)
                document_output_softmax = F.softmax(document_output, 1)

                # Convert model outputs and targets to numpy arrays for metric computation
                output_seg = sentence_output.data.cpu().numpy().argmax(axis=1)
                output_doc = document_output.data.cpu().numpy().argmax(axis=1)

                segment_targets_var = Variable(maybe_cuda(torch.cat(segment_targets, 0), args.cuda), requires_grad=False)
                document_targets_var = Variable(maybe_cuda(torch.cat(document_targets, 0), args.cuda), requires_grad=False)

                target_seg = segment_targets_var.data.cpu().numpy()
                target_doc = document_targets_var.data.cpu().numpy()

                # Update accuracy and other metrics for sentence and document segmentation
                preds_stats.add(output_seg, target_seg)
                preds_stats.add(output_doc, target_doc)
                current_idx = 0
                # # Additional processing for document-level segmentation
                # current_idx = 0
                # for k, doc_target in enumerate(document_targets):
                #     document_sentence_count = len(doc_target)
                #     to_idx = int(current_idx + document_sentence_count)

                #     output = ((document_output_softmax.data.cpu().numpy()[current_idx: to_idx, :])[:, 1] > threshold)
                #     h = np.append(output, [1])
                #     tt = np.append(doc_target, [1])

                #     acc_document.update(h, tt)

                #     current_idx = to_idx

            # Process segment targets
                for k, seg_target in enumerate(segment_targets):
                    document_sentence_count = len(seg_target)
                    to_idx = int(current_idx + document_sentence_count)

                    # Process segment-level predictions
                    seg_output = ((sentence_output_softmax.data.cpu().numpy()[current_idx: to_idx, :])[:, 1] > threshold)
                    seg_h = np.append(seg_output, [1])
                    seg_tt = np.append(seg_target, [1])

                    acc_sentence.update(seg_h, seg_tt)

                    segment_output_file.write(f'Batch {i}, Document {k}, File Path: {paths[k]}:\n')  # Include the file path
                    segment_output_file.write(f'Segments: {seg_h}\n')
                    segment_output_file.write(f'Target Segments: {seg_tt}\n\n')

                    current_idx = to_idx

                current_idx = 0

                # Process document targets
                for k, doc_target in enumerate(document_targets):
                    document_segment_count = len(doc_target)
                    to_idx = int(current_idx + document_segment_count)

                    # Process document-level predictions
                    doc_output = ((document_output_softmax.data.cpu().numpy()[current_idx: to_idx, :])[:, 1] > threshold)
                    doc_h = np.append(doc_output, [1])
                    doc_tt = np.append(doc_target, [1])

                    acc_document.update(doc_h, doc_tt)
                    segment_output_file.write(f'Batch {i}, Document {k}, File Path: {paths[k]}:\n') 
                    segment_output_file.write(f'Large Segment: {doc_h}\n')
                    segment_output_file.write(f'Target Large Segments: {doc_tt}\n\n')

                    current_idx = to_idx

            except RuntimeError as e:
                if 'Length of all samples has to be greater than 0' in str(e):
                    logger.warning(f"Skipping batch due to empty sequence. Error: {str(e)}")
                    continue
                else:
                    raise e

        # Compute accuracy, Pk, Windiff, and F1 for both sentence and document tasks
        epoch_pk_sentence, epoch_windiff_sentence = acc_sentence.calc_accuracy()
        epoch_pk_document, epoch_windiff_document = acc_document.calc_accuracy()

        logger.debug('Testing Epoch: {}, sentence accuracy: {:.4}, sentence Pk: {:.4}, sentence Windiff: {:.4}, sentence F1: {:.4}'.format(
            epoch + 1,
            preds_stats.get_accuracy(),  # assuming you want sentence-level accuracy
            epoch_pk_sentence,
            epoch_windiff_sentence,
            preds_stats.get_f1()  # assuming you want sentence-level F1
        ))

        logger.debug('Testing Epoch: {}, document accuracy: {:.4}, document Pk: {:.4}, document Windiff: {:.4}, document F1: {:.4}'.format(
            epoch + 1,
            preds_stats.get_accuracy(),  # assuming you want document-level accuracy
            epoch_pk_document,
            epoch_windiff_document,
            preds_stats.get_f1()  # assuming you want document-level F1
        ))

        preds_stats.reset()

        return epoch_pk_sentence, epoch_pk_document


def main(args):
    sys.path.append(str(Path(__file__).parent))
    # multiprocessing.set_start_method('spawn')
    checkpoint_path = Path(args.checkpoint_dir)
    checkpoint_path.mkdir(exist_ok=True)

    logger = utils.setup_logger(__name__, os.path.join(args.checkpoint_dir, 'train.log'))

    utils.read_config_file(args.config)
    utils.config.update(args.__dict__)
    logger.debug('Running with config %s', utils.config)

    configure(os.path.join('runs', args.expname))

    if not args.test:
        word2vec = gensim.models.KeyedVectors.load_word2vec_format(utils.config['word2vecfile'], binary=True)
    else:
        word2vec = None

    if not args.infer:
        if args.wiki:
            # dataset_path = Path(utils.config['half-wikidataset'])
            # dataset_path = Path(utils.config['wikidataset'])
            # dataset_path = Path(utils.config['10_concanted_documents_small'])
            # dataset_path = Path(utils.config['1_concanted_document_mini'])
            dataset_path = Path(utils.config[args.path])
            train_dataset = WikipediaDataSet(dataset_path / 'train', word2vec=word2vec,
                                             high_granularity=args.high_granularity)
            dev_dataset = WikipediaDataSet(dataset_path / 'dev', word2vec=word2vec, high_granularity=args.high_granularity)
            test_dataset = WikipediaDataSet(dataset_path / 'test', word2vec=word2vec,
                                            high_granularity=args.high_granularity)
            
        else:
            dataset_path = utils.config['choidataset']
            train_dataset = ChoiDataset(dataset_path, word2vec)
            dev_dataset = ChoiDataset(dataset_path, word2vec)
            test_dataset = ChoiDataset(dataset_path, word2vec)

        train_dl = DataLoader(train_dataset, batch_size=args.bs, collate_fn=collate_fn_2_head, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
        dev_dl = DataLoader(dev_dataset, batch_size=args.test_bs, collate_fn=collate_fn_2_head, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
        test_dl = DataLoader(test_dataset, batch_size=args.test_bs, collate_fn=collate_fn_2_head, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    assert bool(args.model) ^ bool(args.load_from)  # exactly one of them must be set

    if args.model:
        model = import_model(args.model)
    elif args.load_from:
        with open(args.load_from, 'rb') as f:
            model = torch.load(f)

    model.train()
    model = maybe_cuda(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    if not args.infer:
        best_val_pk_sentence = 1.0
        best_val_pk_document = 1.0
        no_improvement_epochs = 0  
        for j in range(args.epochs):
            train(model, args, j, train_dl, logger, optimizer)
            with (checkpoint_path / 'model{:03d}.t7'.format(j)).open('wb') as f:
                torch.save(model, f)

            val_pk_sentence, val_threshold_sentence, val_pk_document, val_threshold_document = validate(model, args, j, dev_dl, logger)
            # Check if there's an improvement in sentence-level segmentation
            print(f"val_pk_sentence: {val_pk_sentence}, best_val_pk_sentence: {best_val_pk_sentence}")
            print(f"val_pk_document: {val_pk_document}, best_val_pk_document: {best_val_pk_document}")
            print(f"{val_threshold_sentence}, {val_threshold_document}")
        if val_pk_sentence < best_val_pk_sentence and val_pk_document < best_val_pk_document:
            best_val_pk_sentence = val_pk_sentence
            best_val_pk_document = val_pk_document
            no_improvement_epochs = 0
            
            # Test using the best thresholds
            test_pk_sentence, test_pk_document = test(model, args, j, test_dl, logger, val_threshold_sentence)

            logger.debug(
                colored(
                    'Current best model from epoch {} with sentence p_k {} and document p_k {}, thresholds {}, {}'.format(
                        j, test_pk_sentence, test_pk_document, val_threshold_sentence, val_threshold_document),
                    'green'))

            # Save the best model for both sentence and document segmentation
            with (checkpoint_path / 'best_model.t7').open('wb') as f:
                torch.save(model, f)

        else:
            no_improvement_epochs += 1

        # Early stopping if no improvement in both tasks for consecutive epochs
        if args.early_stops & no_improvement_epochs >= args.early_stops:
            print(f"Stopping training after {j + 1} epochs due to no improvement in p_k.")

    else:
        test_dataset = WikipediaDataSet(args.infer, word2vec=word2vec,
                                        high_granularity=args.high_granularity)
        test_dl = DataLoader(test_dataset, batch_size=args.test_bs, collate_fn=collate_fn_2_head, shuffle=False,
                             num_workers=args.num_workers)
        print(test(model, args, 0, test_dl, logger, 0.4))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cuda', help='Use cuda?', action='store_true')
    parser.add_argument('--test', help='Test mode? (e.g fake word2vec)', action='store_true')
    parser.add_argument('--bs', help='Batch size', type=int, default=8)
    parser.add_argument('--test_bs', help='Batch size', type=int, default=5)
    parser.add_argument('--epochs', help='Number of epochs to run', type=int, default=10)
    parser.add_argument('--model', help='Model to run - will import and run')
    parser.add_argument('--load_from', help='Location of a .t7 model file to load. Training will continue')
    parser.add_argument('--expname', help='Experiment name to appear on tensorboard', default='exp1')
    parser.add_argument('--checkpoint_dir', help='Checkpoint directory', default='checkpoints')
    parser.add_argument('--stop_after', help='Number of batches to stop after', default=None, type=int)
    parser.add_argument('--config', help='Path to config.json', default='config.json')
    parser.add_argument('--wiki', help='Use wikipedia as dataset?', action='store_true')
    parser.add_argument('--num_workers', help='How many workers to use for data loading', type=int, default=0)
    parser.add_argument('--high_granularity', help='Use high granularity for wikipedia dataset segmentation', action='store_true')
    parser.add_argument('--infer', help='inference_dir', type=str)
    parser.add_argument('--path', help='Path for the datasets', type=str, default='./datasets')
    parser.add_argument('--early_stops', help='Help to stop training early after a certain amounts of epochs does not improt p_k', type=int)

    main(parser.parse_args())
