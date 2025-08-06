import json

# // trong so do dai =  test do dai khac nhau
# // he so 
jsondata = {
#     "word2vecfile": "/home/omri/datasets/word2vec/GoogleNews-vectors-negative300.bin",
#     "choidataset": "/home/omri/code/text-segmentation-2017/data/choi",
#     "wikidataset": "/home/omri/datasets/wikipedia/process_dump_r",

      "word2vecfile": "./datasets/GoogleNews-vectors-negative300.bin",
      "wiki-test-50k": "./datasets/wiki_test_50",
      "wikidataset": "./datasets/wikidataset",
      "half-wikidataset": "./datasets/half-wikidataset",
      "snippets": "./datasets/snippets",
      "10_concanted_documents": "./datasets/10_concanted_document",
      "10_concanted_documents_small": "./datasets/10_concanted_document_small",
      "5_concanted_document_mini": "./datasets/5_concanted_document_mini",
      "1_concanted_document_mini": "./datasets/1_concanted_document_mini",
      "4_7_concanted_document_mini": "./datasets/4_7_concanted_document_mini",
      "4_7_concanted_document": "./datasets/4_7_concanted_document",
      "4_7_concanted_document_small": "./datasets/4_7_concanted_document_small",

       # Ruunning on VSCode
    #   "word2vecfile": "./burr/GoogleNews-vectors-negative300.bin",
    #   "wiki-test-50k": "./burr/wiki_test_50",
    #   "wikidataset": "./burr/wikidataset",
    #   "half-wikidataset": "./burr/half-wikidataset",
    #   "snippets": "./burr/snippets",
    #   "10_concanted_documents": "./burr/10_concanted_document",
    #   "10_concanted_documents_small": "./burr/10_concanted_document_small",
    #   "5_concanted_document_mini": "./burr/5_concanted_document_mini",
    #   "1_concanted_document_mini": "./burr/1_concanted_document_mini",
    #   "4_7_concanted_document_mini": "./burr/4_7_concanted_document_mini",
    #   "4_7_concanted_document": "./burr/4_7_concanted_document",
    #   "4_7_concanted_document_small": "./burr/4_7_concanted_document_small",

       # Ruunning on colab
    #   "word2vecfile": "./my_datasets/GoogleNews-vectors-negative300.bin",
    #   "wiki-test-50k": "./my_datasets/wiki_test_50",
    #   "wikidataset": "./my_datasets/wikidataset",
    #   "half-wikidataset": "./my_datasets/half-wikidataset",
    #   "snippets": "./my_datasets/snippets",
    # "10_concanted_documents": "./my_datasets/10_concanted_document",
    #   "10_concanted_documents_small": "./my_datasets/10_concanted_document_small",
    #   "5_concanted_document_mini": "./my_datasets/5_concanted_document_mini",
    #   "1_concanted_document_mini": "./my_datasets/1_concanted_document_mini",

    #   "word2vecfile": "./1_20_dataset/GoogleNews-vectors-negative300.bin",
    #   "wiki-test-50k": "./1_20_dataset/wiki_test_50",
    #   "wikidataset": "./1_20_dataset/wikidataset",
    #   "half-wikidataset": "./1_20_dataset/half-wikidataset",
    #   "snippets": "./1_20_dataset/snippets",

      
}

with open('config.json', 'w') as f:
    json.dump(jsondata, f)

print("Configuration file generated successfully!")