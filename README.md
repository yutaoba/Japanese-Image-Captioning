# Japanese-Image-Captioning
Generate japanese captions that describe the contents of images

## Build environment

    git clone https://github.com/yutaoba-san/Japanese-Image-Captioning.git
    cd Japanese-Image-Captioning
    ./build.sh Dockerfile image-caption
    ./run.sh image-caption

## Download Flickr 8k Dataset

    cd data && ./download.sh && cd ../

## Configuration
You need to decide parameters on `config.yml` for translation, morphological tranformation, model training and prediction.
- Image filepath
- Description filepath
- Image dimensions
- Vocabulary size
- Sequence length
- Embedding dimension
- Feed forward network dimension
- Batch size
- Epochs
- Checkpoint dirpath
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
However, your access will be denied if you make too many requests in a short time.

    python3 translate.py

## Morphological transformation
MeCab is an open-source morphological transformation engine for japanese.
As a dictionary, you should use mecab-ipadic-NEologd including many neologisms (new word), which are extracted from many language resources on the Web.

    python3 mecab.py

## Train image captioning model
Now, you have images and its japanese descriptions.
The image captioning model consists of CNN and Transformer.
After the training, the text vectorization and the image captioning model weights are saved into the files.

    python3 train.py

## Test image captioning model
You load the trained weights and build the text vectorization and the image captioning model.
You can specify an image filepath with `-i` opition.
Otherwise, the filepath is choosed at random from validation datasets.

### Specific image
    python3 test.py -i <image filepath>
    
### Random image
    python3 test.py

![alt_text](https://github.com/yutaoba/Japanese-Image-Captioning/tree/main/sample/278007543_99f5a91a3e.jpg?rwa=true)
