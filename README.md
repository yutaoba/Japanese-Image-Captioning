# Japanese-Image-Captioning
Generate japanese captions that describe the contents of images

# Git clone

    $ git clone https://github.com/yutaoba-san/Japanese-Image-Captioning.git

# Download Flickr 8k Dataset

    $ cd data
    $ ./download.sh
    $ cd ../

# Translate from english to japanese
    
    $ python3 translate.py

# Morphological transformation

    $ python3 mecab.py

# Train image captioning model

    $ python3 train.py

# Test image captioning model
    
    $ python3 test.py
