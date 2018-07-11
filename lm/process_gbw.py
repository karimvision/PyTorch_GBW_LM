import os
import sys
import pickle

import numpy as np
import torch
from torch.utils.serialization import load_lua
from util import load_np

def build(tensor_path,lua_load=False):
    """ Convert data (sentence id, word id) into a list of sentences """
    tensor = load_lua(tensor_path).long() if lua_load else load_np(tensor_path,item=False)
    num_words = tensor.size()[0]

    print("Processing words to find sentences")
    sentences = dict()
    current_sentence_id = tensor[0, 0]
    start_idx = 0
    for idx, value in enumerate(tensor):
        if (idx % 100000) == 0:
            print(idx, num_words)

        sentence_id, word_id = value
        if current_sentence_id != sentence_id:
            length = idx - start_idx
            sentences[current_sentence_id] = (start_idx, length)

            start_idx = idx
            current_sentence_id = sentence_id

    print("Processing sentences - Building SID Tensor")
    num_sentences = len(sentences)
    data = np.empty((num_sentences, 2), dtype=np.int32)
    for idx, item in enumerate(sentences.items()):
        if (idx % 100000) == 0:
            print(idx, num_sentences)

        key, value = item
        start_idx, length = value
        data[idx, 0] = start_idx
        data[idx, 1] = length
    return data

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise Exception('Boolean value expected.')
        
assert(len(sys.argv) == 4)
dataset_file = sys.argv[1]
filename = sys.argv[2]
lua = str2bool(sys.argv[3]) # in case of th7 gbw dataset, else make it false if you have a custom numpy dataset

print("dataset file:", dataset_file)
print("Output:", filename)
print("lua",lua)

data = build(dataset_file,lua_load=lua)
print("Build Sentence ID Tensor")

with open(filename, 'wb') as f:
    np.savez(f, item=data)
print("Saved Sentence ID Tensor")
