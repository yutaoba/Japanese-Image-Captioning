import sys
import os
import re
import yaml
import pickle
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import TextVectorization

from src.utils import load_captions_data, train_val_split, vectorization, decode_and_resize, read_train_image, read_valid_image, make_dataset
from src.models import build_caption_model, LRSchedule

seed = 111
np.random.seed(seed)
tf.random.set_seed(seed)


# Load the config file
with open('config.yml', 'r') as yml:
    config = yaml.safe_load(yml)

# Path to the train files
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

# Load the dataset
captions_mapping, text_data = load_captions_data(DESCS_PATH)

# Split the dataset into training and validation sets
train_data, valid_data = train_val_split(captions_mapping)
print("Number of training samples: ", len(train_data))
print("Number of validation samples: ", len(valid_data))


"""
## Vectorizing the text data
We'll use the `TextVectorization` layer to vectorize the text data,
that is to say, to turn the
original strings into integer sequences where each integer represents the index of
a word in a vocabulary. We will use a custom string standardization scheme
(strip punctuation characters except `<` and `>`) and the default
splitting scheme (split on whitespace).
"""

# Adapt the vectorization from text data
vectorization.adapt(text_data)

# Pickle the vectorization weights
pickle.dump({'weights': vectorization.get_weights()}, open(os.path.join(TRAIN_PATH, "tv_layer.pkl"), "wb"))


"""
## Building a `tf.data.Dataset` pipeline for training
We will generate pairs of images and corresponding captions using a `tf.data.Dataset` object.
The pipeline consists of two steps:
1. Read the image from the disk
2. Tokenize all the five captions corresponding to the image
"""


# Pass the list of images and the list of corresponding captions
train_dataset = make_dataset(
    list(train_data.keys()), list(train_data.values()), vectorization, split="train"
)

valid_dataset = make_dataset(
    list(valid_data.keys()), list(valid_data.values()), vectorization, split="valid"
)


"""
## Building the model
Our image captioning architecture consists of three models:
1. A CNN: used to extract the image features
2. A TransformerEncoder: The extracted image features are then passed to a Transformer
                    based encoder that generates a new representation of the inputs
3. A TransformerDecoder: This model takes the encoder output and the text data
                    (sequences) as inputs and tries to learn to generate the caption.
"""


# Build the caption model
caption_model = build_caption_model()


"""
## Model training
"""

# Specify checkpoint path
checkpoint_path = os.path.join(TRAIN_PATH, "cp-{epoch:04d}")
checkpoint_dir = os.path.dirname(checkpoint_path)

# Define the model checkpoint per one epoch
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    period=1)

# Define the loss function
cross_entropy = keras.losses.SparseCategoricalCrossentropy(
    from_logits=False, reduction="none"
)

# EarlyStopping criteria
early_stopping = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

# Create a learning rate schedule
num_train_steps = len(train_dataset) * EPOCHS
num_warmup_steps = num_train_steps // 15
lr_schedule = LRSchedule(post_warmup_learning_rate=1e-4, warmup_steps=num_warmup_steps)

# Compile the model
caption_model.compile(optimizer=keras.optimizers.Adam(lr_schedule), loss=cross_entropy)

# Fit the model
caption_model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=valid_dataset,
    callbacks=[cp_callback],
    #callbacks=[early_stopping],
)

# Save the model weights
caption_model.save_weights(os.path.join(TRAIN_PATH, 'model'))
