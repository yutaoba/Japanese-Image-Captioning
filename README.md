# Japanese-Image-Captioning
Generate japanese captions that describe the contents of images

## Git clone

    git clone https://github.com/yutaoba-san/Japanese-Image-Captioning.git

## Download Flickr 8k Dataset

    cd data && ./download.sh && cd ../

## Configuration
Please decide parameters for translation, morphological tranformation, model training and prediction.
- Checkpoint directory path
- Image directory path
- Description directory path
- Image dimensions
- Vocabulary size
- Sequence length
- Embedding dimension
- Feed forward network dimension
- Batch size
- Epochs
- Testing model name

## Translate from english to japanese
### Option 1 : Amazon Translate (default)
Amazon Translate is a neural machine translation service. 
You have to pay-as-you-go based on the number of characters of text that you processed. 
Translate up to 2M characters monthly - free for the first 12 months.
To use Amazon Translate, you have to configure aws account.

### Option 2 : Googletrans
Googletrans is a free python library that implemented Google Translate API. 
This uses the Google Translate Ajax API to make calls to such methods as detect and translate. 
However, your access will be denied if you too many requests in a short time.

    python3 translate.py

## Morphological transformation

    python3 mecab.py

## Train image captioning model

    python3 train.py

## Test image captioning model
    
    python3 test.py
