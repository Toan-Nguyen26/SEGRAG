from torch.utils.data import Dataset
from text_manipulation import word_model
from text_manipulation import extract_sentence_words
from new_wiki_utils import read_concated_wiki_file, get_sections, process_json_file, concat_document, get_old_sections
from pathlib2 import Path
import logging
import re
import wiki_utils
import os
import json
import utils
import spacy
import random

logger = utils.setup_logger(__name__, 'train.log')

section_delimiter = "========"

# Configure logging
logging.basicConfig(filename='logger.txt', 
                    filemode='w',  # 'w' to overwrite each time; use 'a' to append
                    format='%(asctime)s - %(levelname)s - %(message)s', 
                    level=logging.INFO)

logger = logging.getLogger()

def get_files(path):
    all_objects = Path(path).glob('**/*')
    files = [str(p) for p in all_objects if p.is_file()]
    return files


def get_cache_path(wiki_folder):
    cache_file_path = wiki_folder / 'paths_cache'
    return cache_file_path


def cache_wiki_filenames(wiki_folder):
    files = Path(wiki_folder).glob('*/*/*/*')
    cache_file_path = get_cache_path(wiki_folder)

    with cache_file_path.open('w+') as f:
        for file in files:
            f.write(str(file) + u'\n')


def clean_section(section):
    cleaned_section = section.strip('\n')
    return cleaned_section


def get_scections_from_text(txt, high_granularity=True):
    sections_to_keep_pattern = wiki_utils.get_seperator_foramt() if high_granularity else wiki_utils.get_seperator_foramt(
        (1, 2))
    if not high_granularity:
        # if low granularity required we should flatten segments within segemnt level 2
        pattern_to_ommit = wiki_utils.get_seperator_foramt((3, 999))
        txt = re.sub(pattern_to_ommit, "", txt)

        #delete empty lines after re.sub()
        sentences = [s for s in txt.strip().split("\n") if len(s) > 0 and s != "\n"]
        txt = '\n'.join(sentences).strip('\n')


    all_sections = re.split(sections_to_keep_pattern, txt)
    non_empty_sections = [s for s in all_sections if len(s) > 0]

    return non_empty_sections


def read_wiki_file(path, word2vec, remove_preface_segment=True, ignore_list=False, remove_special_tokens=False,
                   return_as_sentences=False, high_granularity=True,only_letters = False):
    data = []
    targets = []
    all_sections = get_old_sections(path, high_granularity)
    required_sections = all_sections[1:] if remove_preface_segment and len(all_sections) > 0 else all_sections
    required_non_empty_sections = [section for section in required_sections if len(section) > 0 and section != "\n"]
    # for section in required_non_empty_sections:
    #     print(section)
    #     print("\n")
    # # print("Required sections:", required_non_empty_sections)
    # # print("\n")
    # print(path)
    # print("=====================================")

    for section in required_non_empty_sections:
        sentences = section.split('\n')
        if sentences:
            for sentence in sentences:
                is_list_sentence = wiki_utils.get_list_token() + "." == sentence.encode('utf-8')
                if ignore_list and is_list_sentence:
                    continue
                if not return_as_sentences:
                    sentence_words = extract_sentence_words(sentence, remove_special_tokens=remove_special_tokens)
                    if 1 <= len(sentence_words):
                        data.append([word_model(word, word2vec) for word in sentence_words])
                    else:
                        #raise ValueError('Sentence in wikipedia file is empty')
                        logger.info('Sentence in wikipedia file is empty')
                else:  # for the annotation. keep sentence as is.
                    if (only_letters):
                        sentence = re.sub('[^a-zA-Z0-9 ]+', '', sentence)
                        data.append(sentence)
                    else:
                        data.append(sentence)
            if data:
                targets.append(len(data) - 1)
    # print(targets)
    return data, targets, path


class WikipediaDataSet(Dataset):
    def __init__(self, root, word2vec, train=True, manifesto=False, folder=False, high_granularity=False, is_json=False, json_data_path=None):
        self.documents = []
        self.train = train
        self.root = root
        self.word2vec = word2vec
        self.high_granularity = high_granularity
        self.is_json = is_json
        self.json_data_path = json_data_path

        if is_json and json_data_path:
            # Load all JSON files in the directory
            json_directory = Path(json_data_path)
            if json_directory.is_dir():
                json_files = list(json_directory.glob('*.json'))
                for json_path in json_files:
                    print(json_path)
                    with open(json_path, "r", encoding="utf-8") as file:
                        documents_in_file = json.load(file)
                        self.documents.append(documents_in_file)
            else:
                raise RuntimeError(f"JSON directory not found: {json_directory}")

            if len(self.documents) == 0:
                raise RuntimeError(f'Found 0 documents in the JSON files within: {json_directory}')
        else:
            if (manifesto):
                self.textfiles = list(Path(root).glob('*'))
            else:
                if (folder):
                    self.textfiles = get_files(root)
                else:
                    root_path = Path(root)
                    cache_path = get_cache_path(root_path)
                    if not cache_path.exists():
                        cache_wiki_filenames(root_path)
                    else:
                        print('Found cache file: {}'.format(cache_path))
                    self.textfiles = cache_path.read_text().splitlines()
                    print("Number of files: ", len(self.textfiles))
            if len(self.textfiles) == 0:
                raise RuntimeError('Found 0 images in subfolders of: {}'.format(root))
            
            # concat_num_files_min = 4
            # concat_num_files_max = 7
            # num_files = len(self.textfiles)
            # current_index = 0
            # document_index = 1

            # while current_index < num_files:
            #     # Randomly select the number of files to concatenate between 4 and 7
            #     concat_num_files = random.randint(concat_num_files_min, concat_num_files_max)
                
            #     # Calculate the start and end index for the documents to concatenate
            #     start_idx = current_index
            #     end_idx = min(current_index + concat_num_files, num_files)  # Ensure we don't go out of bounds

            #     # Initialize an empty string to store the concatenated document
            #     concatenated_document = ""

            #     # Loop through the paths and concatenate the contents
            #     for i in range(start_idx, end_idx):
            #         path = self.textfiles[i]
            #         concatenated_document += concat_document(Path(path)) + "===============\n"  # Add a separator between documents

            #     # Determine the directory structure
            #     batch_num = document_index // 1000
            #     sub_dir_1 = f"{batch_num:02d}"
            #     sub_dir_2 = f"{(document_index % 1000) // 100:02d}"

            #     output_directory = os.path.join(root, "concat", sub_dir_1, sub_dir_2)
            #     os.makedirs(output_directory, exist_ok=True)  # Create directory if it doesn't exist

            #     # Define the output file path
            #     output_path = os.path.join(output_directory, f"concatenated_document_{document_index}.txt")

            #     # Write the concatenated document to a file
            #     with open(output_path, 'w', encoding='utf-8') as output_file:
            #         output_file.write(concatenated_document.strip())

            #     print(f"Concatenated document {document_index} written to: {output_path}")

            #     # Update the current index and document index
            #     current_index = end_idx
            #     document_index += 1
                            

    def __getitem__(self, index):
        if self.is_json:
             # Retrieve the document at the given index
            document = self.documents[index]
            return process_json_file(document, self.word2vec, remove_special_tokens=True)
        else:
            path = self.textfiles[index]
            return read_concated_wiki_file(Path(path), self.word2vec, ignore_list=True, remove_special_tokens=True,
                                high_granularity=self.high_granularity)
            # return read_wiki_file(Path(path), self.word2vec, ignore_list=True, remove_special_tokens=True,
            #                     high_granularity=self.high_granularity)
    def __len__(self):
        if self.is_json:
            return len(self.documents)
        else:
            return len(self.textfiles)