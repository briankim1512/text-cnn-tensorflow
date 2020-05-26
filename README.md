# Text CNN Using Tensorflow

This code implements [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882) models.

## About

This CNN takes the review of something given by the user and tries to discern what rating follows the text.
The returned rating depends on what the user trains the CNN on.

[Original Repo](https://github.com/DongjunLee/text-cnn-tensorflow) by DongjunLee

## Requirements

- Python 3.6
- TensorFlow 1.4
- [hb-config](https://github.com/hb-research/hb-config) (Singleton Config)
- tqdm
- requests

## Installation

To install requirements, simply run the following command within the project folder

```pip install -r requirements.txt```

Although conda would have been preferred, conda unfortunately does not have a 1.4 version of tensorflow, 
nor does it have hb_config. Pip will be used in this case.

## Usage

To preprocess the dataset and train it. You can modify the config for the NN in `config` folder.

```
chmod 755 scripts/prepare_kaggle_movie_reviews.sh
sh scripts/prepare_kaggle_movie_reviews.sh
```

To train the CNN, run the following command

```
python main.py --config kaggle_movie_review --mode train_and_evaluate
```

After training, you can try typing the sentences what you want using `predict.py`.

```python python predict.py --config kaggle_movie_review```

Predict example

```
python predict.py --config kaggle_movie_review
Setting max_seq_length to Config : 62
load vocab ...
Typing anything :)

> good
1
> bad
0
```

## Config

Configuration can be done by passing the arguments through `main.py` or through configuring the config file within the `config` folder

The arguments that can be passed to `main.py` are as follows

```
  --config CONFIG       
                        config file name (default: config)
  --mode MODE           
                        Mode (train/test/train_and_evaluate) (default: train)
  --batchsize BATCHSIZE
                        Batch size (default: 0)
  --embeddim EMBEDDIM   
                        Embedded dimensions (default: 0)
  --numfilters NUMFILTERS
                        Number of filters (default: 0)
  --dropout DROPOUT     
                        Dropout rate (default: 0)
  --learnrate LEARNRATE
                        Learn rate (default: 0)
  --trainsteps TRAINSTEPS
                        Training steps (default: 0)
  --savecheck SAVECHECK
                        Number of steps before model is saved (default: 0)
  --evalcheck EVALCHECK
                        Number of steps before model is evaluated (default: 0)
```

### Note
When the default value `0` is passed, the values for that respective setting will be taken from the config file instead. 

The config files within the `config` folder follow this type of format

example: kaggle\_movie\_review.yml

```yml
data:
  type: 'kaggle_movie_review'
  base_path: 'data/'
  raw_data_path: 'kaggle_movie_reviews/'
  processed_path: 'kaggle_processed_data'
  testset_size: 25000
  num_classes: 5
  PAD_ID: 0

model:
  batch_size: 64
  embed_type: 'rand'     #(rand, static, non-static, multichannel)
  pretrained_embed: "" 
  embed_dim: 300
  num_filters: 256
  filter_sizes:
    - 2
    - 3
    - 4
    - 5
  dropout: 0.5

train:
  learning_rate: 0.00005
  
  train_steps: 100000
  model_dir: 'logs/kaggle_movie_review'
  
  save_checkpoints_steps: 1000
  loss_hook_n_iter: 1000
  check_hook_n_iter: 1000
  min_eval_frequency: 1000
  
slack:
  webhook_url: ""   # unused for this project
```


## Reference

- [Implementing a CNN for Text Classification in TensorFlow](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/) by Denny Britz
- [Paper - Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882) (2014) by Y Kim
- [Paper - A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1510.03820.pdf) (2015) Y Zhang
