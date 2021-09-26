import os
import sys
import time
import yaml
from pickle import load
from tqdm import tqdm
from collections import defaultdict

import MeCab
import boto3

from src.utils import load_doc, save_descriptions


# Load the config file
with open('config.yml', 'r') as yml:
    config = yaml.safe_load(yml)

# Path to the train files
TRAIN_PATH = config['train_path']

# Path to the images
IMAGES_PATH = config['images_path']
DESCS_PATH_ORI = config['descs_path_ori']
DESCS_PATH_JA = config['descs_path_ja']
DESCS_PATH_MECAB = config['descs_path_mecab']


"""
##  Morphological transformation
"""


# Load the descriptions
descs = load_doc(DESCS_PATH_JA)
descs = descs.split("\n")
mapping = defaultdict(list)
for desc in descs:
    filename = desc.split("\t")[0]
    text = desc.split("\t")[1]
    mapping[filename].append(text)

# Instance of MeCab
m = MeCab.Tagger("-d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd")

# MeCab
for key, desc_list in mapping.items():
    for i in range(len(desc_list)):
        desc = desc_list[i]

        for char in ['、', '。', '"', '「', '」']:
            desc = desc.replace(char, '')

        desc = m.parse(desc).split('\n')

        lists = list()
        for word in desc:
            lists.append(word.split('\t')[0])
        desc = ' '.join(lists)
        desc = desc.replace(" EOS ", "")
        desc_list[i] = desc.strip()

# Save the descriptions
save_descriptions(mapping, DESCS_PATH_MECAB)
