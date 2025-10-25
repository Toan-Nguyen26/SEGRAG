from argparse import ArgumentParser
from data_preprocessing.narrativeqa import create_individual_documents_narrativeqa_json
from data_preprocessing.qasper import create_concantenated_documents_qasper_json
from data_preprocessing.quality import create_concatenated_documents_quality_json
from data_preprocessing.squad import create_concatenated_documents_squad_json
from data_preprocessing.vietlaw import create_concatenated_documents_viet_law_jsonl

def main(args):
    if args.dataset == 'squad':
        create_concatenated_documents_squad_json(num_files=args.num_files)
    elif args.dataset == 'narrativeqa':
        create_individual_documents_narrativeqa_json(num_files=args.num_files)
    elif args.dataset == 'quality':
        create_concatenated_documents_quality_json(num_files=args.num_files)
    elif args.dataset == 'qasper':
        create_concantenated_documents_qasper_json(num_files=args.num_files)
    elif args.dataset == 'viet_law':
        create_concatenated_documents_viet_law_jsonl(
            txt_folder=args.txt_folder,
            qa_json_path=args.qa_json_path
        )
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}. Please choose 'squad', 'narrativeqa', 'quality', 'qasper', or 'viet_law'.")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', help='Dataset to process: squad, narrativeqa, quality, qasper, or viet_law', required=True, type=str, default="squad")
    parser.add_argument('--num_files', help='Total documents wanting to have (for squad/narrativeqa/quality/qasper)', type=int, default=10)
    
    # Vietnamese law specific arguments
    parser.add_argument('--txt_folder', help='Path to folder containing .txt files (for viet_law)', type=str, default='data/vietnamese_law/chapters')
    parser.add_argument('--qa_json_path', help='Path to Q&A JSON file (for viet_law)', type=str, default='data/vietnamese_law/hanhchinh.json')
    
    # By default the command to run is:
    # python document_concat.py --dataset squad --num_files 10
    # For Vietnamese law:
    # python document_concat.py --dataset viet_law --txt_folder data/vietlaw/chapters --qa_json_path data/vietlaw/hanhchinh.json

    main(parser.parse_args())