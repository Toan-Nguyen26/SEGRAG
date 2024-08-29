from torch.utils.data import Dataset
from text_manipulation import word_model
from text_manipulation import extract_sentence_words
from pathlib2 import Path
import logging
import re
import wiki_utils
import os
import json
import utils
import spacy

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


def get_sections(path, high_granularity=True):
    file = open(str(path), "r")
    # with open(path, 'r', encoding='utf-8') as file:
    #     raw_content = file.read()
    # file.close()

    with open(path, 'r', encoding='utf-8') as file:
        raw_content = file.read()
    file.close()
    clean_txt = raw_content.strip()
    # print("Raw content:", raw_content, "Clean text:", clean_txt)
    sections = [clean_section(s) for s in get_scections_from_text(clean_txt, high_granularity)]
    for section in sections:
        print("Section: \n", section)
        print("=====================================")
    # Debugging Information
    logger.info(f"File: {path}, Number of sections: {len(sections)}")
    for i, section in enumerate(sections):
        logger.info(f"Section {i+1} length: {len(section)}")

    return sections


def read_wiki_file(path, word2vec, remove_preface_segment=True, ignore_list=False, remove_special_tokens=False,
                   return_as_sentences=False, high_granularity=True,only_letters = False):
    data = []
    targets = []
    all_sections = get_sections(path, high_granularity)
    required_sections = all_sections[1:] if remove_preface_segment and len(all_sections) > 0 else all_sections
    required_non_empty_sections = [section for section in required_sections if len(section) > 0 and section != "\n"]
    print(f"{len(all_sections)} with non-required {len(required_non_empty_sections)} and required {len(required_sections)} for {path}")
    # print("\n")
    # print("Required sections:", required_non_empty_sections)
    # print("\n")
    # print("Required sections:", required_sections)
    i = 0
    for section in required_non_empty_sections:
        sentences = section.split('\n')
        if sentences:
            for sentence in sentences:
                # print("Sentence:", sentence,"on path:", path, "index:", i)
                # print("=====================================")
                i = i + 1
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
    print(targets)
    return data, targets, path

def split_text_into_sentences(text):
    # Load the Spacy model
    nlp = spacy.load("en_core_web_sm")  # Ensure you have this model installed or use a different one
    
    # Process the text with Spacy
    doc = nlp(text)
    
    # Extract sentences
    sentences = [sent.text.strip() for sent in doc.sents]
    
    return sentences

def process_json_file(document, word2vec, remove_special_tokens=False, return_as_sentences=False, only_letters=False):
    data = []
    targets = []
    content = document['content']
    doc_id = document['id']
    sentences = split_text_into_sentences(content)
    # i = 0
    for sentence in sentences:
        # print("Sentence:", sentence ,"index:", i)
        # print("=====================================")
        # i = i + 1
        is_list_sentence = wiki_utils.get_list_token() + "." == sentence.encode('utf-8')
        if is_list_sentence:
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
    return data, targets, doc_id

class WikipediaDataSet(Dataset):
    def __init__(self, root, word2vec, train=True, manifesto=False, folder=False, high_granularity=False, is_json=False, json_file=None):

        if is_json and json_file:
            with open(json_file, "r", encoding="utf-8") as file:
                self.documents = json.load(file)
            if len(self.documents) == 0:
                raise RuntimeError('Found 0 documents in the JSON file: {}'.format("RAG\data\squad\concatenated_documents.json"))
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

            if len(self.textfiles) == 0:
                raise RuntimeError('Found 0 images in subfolders of: {}'.format(root))
        self.train = train
        self.root = root
        self.word2vec = word2vec
        self.high_granularity = high_granularity
        self.is_json = is_json
        self.json_file = json_file

    def __getitem__(self, index):
        if self.is_json:
             # Retrieve the document at the given index
            document = self.documents[index]
            return process_json_file(document, self.word2vec, remove_special_tokens=True)
        else:
            path = self.textfiles[index]
            print(Path(path))
            return read_wiki_file(Path(path), self.word2vec, ignore_list=True, remove_special_tokens=True,
                                high_granularity=self.high_granularity)

        # return process_json_file(self.word2vec, remove_special_tokens=True)
    def __len__(self):
        if self.is_json:
            return len(self.documents)
        else:
            return len(self.textfiles)

class InMemoryWikipediaDataSet(WikipediaDataSet):
    def __init__(self, root, word2vec, train=True, manifesto=False, folder=False, high_granularity=False):
        # Initialize the parent class to handle file paths and configurations
        super().__init__(root, word2vec, train, manifesto, folder, high_granularity)
        
        self.data = []
        self.targets = []
        self.paths = []
        
        # Preload all data into memory
        for path in self.textfiles:
            data, target, path = read_wiki_file(Path(path), self.word2vec, ignore_list=True, remove_special_tokens=True,
                                                high_granularity=self.high_granularity)
            self.data.append(data)
            self.targets.append(target)
            self.paths.append(path)

    def __getitem__(self, index):
        # Return preloaded data and target from memory
        return self.data[index], self.targets[index], self.paths[index]

    def __len__(self):
        return len(self.data)
