from argparse import ArgumentParser
from data_preprocessing.narrativeqa import create_individual_documents_narrativeqa_json
from data_preprocessing.qasper import create_concantenated_documents_qasper_json
from data_preprocessing.quality import create_concatenated_documents_quality_json
from data_preprocessing.squad import create_concatenated_documents_squad_json

def main(args):
    if args.dataset == 'squad':
        create_concatenated_documents_squad_json(num_files=args.num_files)
    elif args.dataset == 'narrativeqa':
        create_individual_documents_narrativeqa_json(num_files=args.num_files)
    elif args.dataset == 'quality':
        create_concatenated_documents_quality_json(num_files=args.num_files)
    elif args.dataset == 'qasper':
        create_concantenated_documents_qasper_json(num_files=args.num_files)
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}. Please choose 'squad' or 'narrativeqa' or 'quality'.")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', help='whenever it is squad or narrative_qa or quality',  required=True, type=str, default="squad")
    parser.add_argument('--num_files', help='total documents wanting to have', type=int, default=10)
    # By default the command to run is:
    # python document_concat.py --dataset squad --num_files 10

    main(parser.parse_args())