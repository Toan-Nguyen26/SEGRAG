from choiloader import ChoiDataset, collate_fn
from wiki_loader import WikipediaDataSet
from torch.utils.data import DataLoader
from pathlib2 import Path
import utils
import gensim
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("output.log"),
                        logging.StreamHandler()  # This will output to the console as well
                    ])

logger = logging.getLogger()

def main():
    utils.read_config_file("config.json")
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(utils.config['word2vecfile'], binary=True)
    dataset_path = Path(utils.config['snippets'])

    train_dataset = WikipediaDataSet(dataset_path / 'train', word2vec=word2vec,
                                        high_granularity="store_true")
    dev_dataset = WikipediaDataSet(dataset_path / 'dev', word2vec=word2vec, high_granularity="store_true")
    test_dataset = WikipediaDataSet(dataset_path / 'test', word2vec=word2vec,
                                    high_granularity="store_true")

    print("WikipediaDataset loaded")

    train_dl = DataLoader(train_dataset, batch_size=8, collate_fn=collate_fn, shuffle=True,
                            num_workers=0)
    dev_dl = DataLoader(dev_dataset, batch_size=8, collate_fn=collate_fn, shuffle=False,
                        num_workers=0)
    test_dl = DataLoader(test_dataset, batch_size=8, collate_fn=collate_fn, shuffle=False,
                            num_workers=0)

    print("DataLoader loaded")

    for i, data in enumerate(train_dl):
        logger.info(f"The file is on {i}")
        doc_sizes = [len(doc) for doc in data]
        if any(size == 0 for size in doc_sizes):
            logger.warning(f"Found empty sequence {i} in {data}")

    print("Done")

if __name__ == '__main__':
    main()