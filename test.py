from pathlib import Path
import utils

# dataset_path = Path('datasets/wikidataset/train')

# files = list(dataset_path.glob('*/*/*/*'))
# print(f'Found {len(files)} files.')

half_dataset_path = Path('datasets/half-wikidataset/train')

helf_files = list(half_dataset_path.glob('*/*/*/*'))
print(f'Found {len(helf_files)} files.')

# dataset_path_test = Path('datasets/wikidataset/test')

# files = list(dataset_path_test.glob('*/*/*/*'))
# print(f'Found {len(files)} files.')

half_dataset_path_test = Path('datasets/half-wikidataset/test')

helf_files = list(half_dataset_path_test.glob('*/*/*/*'))
print(f'Found {len(helf_files)} files.')


# dataset_folders = [Path(utils.config['wikidataset']) / 'test']
# # if (args.wiki_folder):
# #     dataset_folders = []
# #     dataset_folders.append(args.wiki_folder)
# print(f'Found {len(dataset_folders)} test files in wikidataset.')
# print('running on wikipedia')

# half_dataset_folders = [Path(utils.config['half-wikidataset']) / 'test']
# # if (args.wiki_folder):
# #     dataset_folders = []
# #     dataset_folders.append(args.wiki_folder)
# print(f'Found {len(half_dataset_folders)} test files half-wikidataset.')
# print('running on wikipedia')