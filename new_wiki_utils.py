# def add_documents:
#     # Loop through files in groups of 1000
#     num_files = len(self.textfiles)
#     num_batches = (num_files + 9) // 10  # Calculate how many batches of 10 files

#     for index in range(num_batches):
#         # Calculate the start and end index for the 10 documents to concatenate
#         start_idx = index * 10
#         end_idx = min(start_idx + 10, num_files)  # Ensure we don't go out of bounds

#         # Initialize an empty string to store the concatenated document
#         concatenated_document = ""

#         # Loop through the paths and concatenate the contents
#         for i in range(start_idx, end_idx):
#             path = self.textfiles[i]
#             concatenated_document += concat_document(Path(path)) + "===============\n"  # Add a separator between documents

#         # Determine the directory structure
#         # e.g., datasets/half-wikidataset/train/concat/00/00/concatenated_document_1.txt
#         batch_num = index // 1000
#         sub_dir_1 = f"{batch_num:02d}"
#         sub_dir_2 = f"{(index % 1000) // 100:02d}"
        
#         output_directory = os.path.join(root, "concat", sub_dir_1, sub_dir_2)
#         os.makedirs(output_directory, exist_ok=True)  # Create directory if it doesn't exist

#         # Define the output file path
#         output_path = os.path.join(output_directory, f"concatenated_document_{index + 1}.txt")

#         # Write the concatenated document to a file
#         with open(output_path, 'w', encoding='utf-8') as output_file:
#             output_file.write(concatenated_document.strip())

#         print(f"Concatenated document {index + 1} written to: {output_path}")
import wiki_utils
from text_manipulation import extract_sentence_words
from text_manipulation import word_model
import logging
import re
import spacy
import utils

logger = utils.setup_logger(__name__, 'train.log')


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

def concat_document(path):
    concatenated_document = ""  # Initialize an empty string to store the concatenated document

    file = open(str(path), "r")

    with open(path, 'r', encoding='utf-8') as file:
        raw_content = file.read()
    file.close()
    
    # Define a pattern to match and remove the entire preface section, including its marker
    preface_pattern = r'========,1,preface\..*?(?========,\d+,[^\n]+\.\n)'
    
    # Remove the preface section
    modified_content = re.sub(preface_pattern, '', raw_content, flags=re.DOTALL)
    return modified_content

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

def clean_section(section):
    cleaned_section = section.strip('\n')
    return cleaned_section

def get_sections(path, high_granularity=True):
    file = open(str(path), "r")
    with open(path, 'r', encoding='utf-8') as file:
        raw_content = file.read()
    file.close()
    documents = split_documents(raw_content)
    # return sections
    all_sections = []
    for doc in documents:
        clean_txt = doc.strip()
        sections = [clean_section(s) for s in get_scections_from_text(clean_txt, high_granularity)]
        all_sections.append(sections)

        # logger.info(f"Document processed, Number of sections: {len(sections)}")
        # for i, section in enumerate(sections):
        #     logger.info(f"Section {i+1} length: {len(section)}")

    return all_sections

def split_documents(content):
    """
    Split the content into multiple documents based on the document separator.
    """
    documents = content.split("===============")
    return [doc.strip() for doc in documents if doc.strip()]

def read_concated_wiki_file(path, word2vec, remove_preface_segment=False, ignore_list=False, remove_special_tokens=False,
                   return_as_sentences=False, high_granularity=True,only_letters = False):
    data = []
    segment_targets = []
    document_targets = []
    # Get all sections for all documents
    all_documents_sections = get_sections(path, high_granularity)
    # Should have another for loop to handle the concated documents
    for document in all_documents_sections:
        required_sections = document[1:] if remove_preface_segment and len(document) > 0 else document
        required_non_empty_sections = [section for section in required_sections if len(section) > 0 and section != "\n"]
        for i, section in enumerate(required_non_empty_sections):
            sentences = section.split('\n')

            # Skip the first sentence in the section. Data processing sucks
            sentences = sentences[1:]
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
                        if only_letters:
                            sentence = re.sub('[^a-zA-Z0-9 ]+', '', sentence)
                            data.append(sentence)
                        else:
                            data.append(sentence)

                # Print the last sentence in the sentences list
                if i < len(required_non_empty_sections) - 1:
                    if data:
                        segment_targets.append(len(data) - 1)
                else:
                    if data:
                        document_targets.append(len(data) - 1)
    # right now, target is an array of the index of the endsentence (so like [5,17])
    # But now, we should also return additional array for the end of the document
    # print(f"Path: {path}")
    # print(f"Segments are: {segment_targets}")
    # print("\n")
    # print(f"Documents are: {document_targets}")
    # print("------------------")
    return data, segment_targets, segment_targets, path

def split_text_into_sentences(text):
    # Load the Spacy model
    nlp = spacy.load("en_core_web_sm")  # Ensure you have this model installed or use a different one
    
    # Process the text with Spacy
    doc = nlp(text)
    
    # Extract sentences
    sentences = [sent.text.strip() for sent in doc.sents]
    
    return sentences

