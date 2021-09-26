import sys
import os
import re
import yaml
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import efficientnet
from tensorflow.keras.layers import TextVectorization


with open('config.yml', 'r') as yml:
    config = yaml.safe_load(yml)

TRAIN_PATH = config['train_path']

# Path to the images
IMAGES_PATH = config['images_path']
DESCS_PATH = config['descs_path_mecab']

# Desired image dimensions
IMAGE_SIZE = (config['img_size_w'], config['img_size_h'])

# Vocabulary size
VOCAB_SIZE = config['vocab_size']

# Fixed length allowed for any sequence
SEQ_LENGTH = config['seq_length']

# Dimension for the image embeddings and token embeddings
EMBED_DIM = config['embed_dim']

# Per-layer units in the feed-forward network
FF_DIM = config['ff_dim']

# Other training parameters
BATCH_SIZE = config['batch_size']
EPOCHS = config['epochs']


"""
## Preparing the dataset
"""


#テキストをロードする関数
def load_doc(filename):
    with open(filename, "r") as file:
        text = file.read()
    return text

#テキストをファイルに保存する関数
def save_descriptions(descriptions, filename):
    lines = list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            #lines.append(key + ' ' + desc)
            lines.append(key + '\t' + desc)
    data = '\n'.join(lines)
    with open(filename, "w") as file:
        file.write(data)

def load_captions_data(filename):
    """Loads captions (text) data and maps them to corresponding images.
    Args:
        filename: Path to the text file containing caption data.
    Returns:
        caption_mapping: Dictionary mapping image names and the corresponding captions
        text_data: List containing all the available captions
    """

    with open(filename) as caption_file:
        caption_data = caption_file.readlines()
        caption_mapping = {}
        text_data = []
        images_to_skip = set()

        for line in caption_data:
            line = line.rstrip("\n")
            # Image name and captions are separated using a tab
            img_name, caption = line.split("\t")
            #img_name, caption = line.split(" ")

            # Each image is repeated five times for the five different captions.
            # Each image name has a suffix `#(caption_number)`
            #img_name = img_name.split("#")[0]
            img_name = os.path.join(IMAGES_PATH, img_name.strip())

            # We will remove caption that are either too short to too long
            tokens = caption.strip().split()

            #if len(tokens) < 5 or len(tokens) > SEQ_LENGTH:
            #    images_to_skip.add(img_name)
            #    continue

            #if img_name.endswith("jpg") and img_name not in images_to_skip:
            #    # We will add a start and an end token to each caption
            #    caption = "<start> " + caption.strip() + " <end>"
            #    text_data.append(caption)

            #    if img_name in caption_mapping:
            #        caption_mapping[img_name].append(caption)
            #    else:
            #        caption_mapping[img_name] = [caption]

            # We will add a start and an end token to each caption
            caption = "<start> " + caption.strip() + " <end>"
            text_data.append(caption)

            if img_name in caption_mapping:
                caption_mapping[img_name].append(caption)
            else:
                caption_mapping[img_name] = [caption]

        #for img_name in images_to_skip:
        #    if img_name in caption_mapping:
        #        del caption_mapping[img_name]
        
        return caption_mapping, text_data


def train_val_split(caption_data, train_size=0.8, shuffle=True):
    """Split the captioning dataset into train and validation sets.
    Args:
        caption_data (dict): Dictionary containing the mapped caption data
        train_size (float): Fraction of all the full dataset to use as training data
        shuffle (bool): Whether to shuffle the dataset before splitting
    Returns:
        Traning and validation datasets as two separated dicts
    """

    # 1. Get the list of all image names
    all_images = list(caption_data.keys())

    # 2. Shuffle if necessary
    if shuffle:
        np.random.shuffle(all_images)

    # 3. Split into training and validation sets
    train_size = int(len(caption_data) * train_size)

    training_data = {
        img_name: caption_data[img_name] for img_name in all_images[:train_size]
    }
    validation_data = {
        img_name: caption_data[img_name] for img_name in all_images[train_size:]
    }

    # 4. Return the splits
    return training_data, validation_data


"""
## Vectorizing the text data
We'll use the `TextVectorization` layer to vectorize the text data,
that is to say, to turn the
original strings into integer sequences where each integer represents the index of
a word in a vocabulary. We will use a custom string standardization scheme
(strip punctuation characters except `<` and `>`) and the default
splitting scheme (split on whitespace).
"""


def custom_standardization(input_string):
    strip_chars = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
    strip_chars = strip_chars.replace("<", "")
    strip_chars = strip_chars.replace(">", "")
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")

vectorization = TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=SEQ_LENGTH,
    standardize=custom_standardization,
)

# Data augmentation for image data
image_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomContrast(0.3),
    ]
)


"""
## Building a `tf.data.Dataset` pipeline for training
We will generate pairs of images and corresponding captions using a `tf.data.Dataset` object.
The pipeline consists of two steps:
1. Read the image from the disk
2. Tokenize all the five captions corresponding to the image
"""


def decode_and_resize(img_path, size=IMAGE_SIZE):
    img = tf.io.read_file(img_path) #+".jpg")
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    return img


def read_train_image(img_path, size=IMAGE_SIZE):
    img = decode_and_resize(img_path)
    img = image_augmentation(tf.expand_dims(img, 0))[0]
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img


def read_valid_image(img_path, size=IMAGE_SIZE):
    img = decode_and_resize(img_path)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img


def make_dataset(images, captions, vectorization, split="train"):

    if split == "train":
        img_dataset = tf.data.Dataset.from_tensor_slices(images).map(
            read_train_image, num_parallel_calls=tf.data.AUTOTUNE
        )
    else:
        img_dataset = tf.data.Dataset.from_tensor_slices(images).map(
            read_valid_image, num_parallel_calls=tf.data.AUTOTUNE
        )

    cap_dataset = tf.data.Dataset.from_tensor_slices(captions).map(
        vectorization, num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = tf.data.Dataset.zip((img_dataset, cap_dataset))
    dataset = dataset.batch(BATCH_SIZE).shuffle(256).prefetch(tf.data.AUTOTUNE)
    return dataset
