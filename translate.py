import os
import sys
import time
import yaml
from pickle import load
from tqdm import tqdm

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
##  Converting texts from english to japanese
"""


# Load the english texts
doc = load_doc(DESCS_PATH_ORI)

# Prepare the text mapping
mapping = dict()

# Instance of Amazon Translate
translate = boto3.client(service_name='translate', region_name='ap-northeast-1', use_ssl=True)

# Translate texts from english to japanese
for line in tqdm(doc.split('\n')):
    tokens = line.split('\t')
    image_id, image_desc = tokens[0], tokens[1:]
    image_id = image_id.split('#')[0]
    image_desc = ' '.join(image_desc)
    
    # Google Translate (Free, but the number of requests is limited.)
    #from googletrans import Translator
    #translator = Translator(service_urls=['translate.googleapis.com'])
    #image_desc_ja = translator.translate(image_desc, dest='ja').text
    
    # Amazon Translate
    result = translate.translate_text(Text=image_desc, SourceLanguageCode="en", TargetLanguageCode="ja")
    image_desc_ja = result.get('TranslatedText')

    if image_id not in mapping:
        mapping[image_id] = list()
    mapping[image_id].append(image_desc_ja)
    
# Save the descriptions
save_descriptions(mapping, DESCS_PATH_JA)
