import sys
import os
import yaml
import pickle
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from src.utils import load_captions_data, train_val_split, vectorization, read_valid_image 
from src.models import build_caption_model


seed = 111
np.random.seed(seed)
tf.random.set_seed(seed)

# Load the config file
with open('config.yml', 'r') as yml:
    config = yaml.safe_load(yml)

# Path to the training files
TRAIN_PATH = config['train_path']

# Path to the images and descriptions
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

# Caption model path 
MODEL_PATH = os.path.join(TRAIN_PATH, config['model_name'])


"""
## Testing image captioning
"""


# Load the dataset
captions_mapping, text_data = load_captions_data(DESCS_PATH)

# Split the dataset into training and validation sets
train_data, valid_data = train_val_split(captions_mapping)
print("Number of training samples: ", len(train_data))
print("Number of validation samples: ", len(valid_data))

# Build the caption model and load weights
caption_model = build_caption_model()
caption_model.load_weights(MODEL_PATH)

# Build the vectorization and load weights
vector_weights = pickle.load(open(os.path.join(TRAIN_PATH, "tv_layer.pkl"), "rb"))['weights']
vectorization.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
vectorization.set_weights(vector_weights)

# Prepare index lookup, max decoded length, and valid images
vocab = vectorization.get_vocabulary()
index_lookup = dict(zip(range(len(vocab)), vocab))
max_decoded_sentence_length = SEQ_LENGTH - 1
valid_images = list(valid_data.keys())


def generate_caption():
    # Select a random image from the validation dataset
    sample_img = np.random.choice(valid_images)

    # Read the image from the disk
    sample_img = read_valid_image(sample_img)
    img = sample_img.numpy().clip(0, 255).astype(np.uint8)
    plt.imshow(img)
    plt.show()

    # Pass the image to the CNN
    img = tf.expand_dims(sample_img, 0)
    img = caption_model.cnn_model(img)

    # Pass the image features to the Transformer encoder
    encoded_img = caption_model.encoder(img, training=False)

    # Generate the caption using the Transformer decoder
    decoded_caption = "<start> "
    for i in range(max_decoded_sentence_length):
        tokenized_caption = vectorization([decoded_caption])[:, :-1]
        mask = tf.math.not_equal(tokenized_caption, 0)
        predictions = caption_model.decoder(
            tokenized_caption, encoded_img, training=False, mask=mask
        )
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = index_lookup[sampled_token_index]
        if sampled_token == " <end>":
            break
        decoded_caption += " " + sampled_token

    decoded_caption = decoded_caption.replace("<start> ", "")
    decoded_caption = decoded_caption.replace(" <end>", "").strip()
    print("Predicted Caption: ", decoded_caption)


# Predict 
generate_caption()
