# Text CNN Using Tensorflow

This code implements [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882) models.

## Requirements

- Python 3.6
- TensorFlow 1.4
- [hb-config](https://github.com/hb-research/hb-config) (Singleton Config)
- tqdm
- requests

All of these requirements are in the requirements .txt, therefore you can just run `pip` to install these. More on this later

## Project Structure

init Project by [hb-base](https://github.com/hb-research/hb-base)

    .
    ├── config                  # Config files (.yml, .json) using with hb-config
    ├── data                    # dataset path
    ├── notebooks               # Prototyping with numpy or tf.interactivesession
    ├── scripts                 # download or prepare dataset using shell scripts
    ├── text-cnn                # text-cnn architecture graphs (from input to logits)
        ├── __init__.py             # Graph logic
    ├── data_loader.py          # raw_date -> precossed_data -> generate_batch (using Dataset)
    ├── hook.py                 # training or test hook feature (eg. print_variables)
    ├── main.py                 # define experiment_fn
    ├── model.py                # define EstimatorSpec
    └── predict.py              # test trained model       

Reference : [hb-config](https://github.com/hb-research/hb-config), [Dataset](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_generator), [experiments_fn](https://www.tensorflow.org/api_docs/python/tf/contrib/learn/Experiment), [EstimatorSpec](https://www.tensorflow.org/api_docs/python/tf/estimator/EstimatorSpec)

- Dataset : [rt-polarity](https://github.com/yoonkim/CNN_sentence), [Sentiment Analysis on Movie Reviews](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/data)

## Config

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


## Usage

Install requirements.

```pip install -r requirements.txt```

Then, prepare dataset and train it.

```
chmod 755 scripts/prepare_kaggle_movie_reviews.sh
sh scripts/prepare_kaggle_movie_reviews.sh
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

## Reference

- [Implementing a CNN for Text Classification in TensorFlow](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/) by Denny Britz
- [Paper - Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882) (2014) by Y Kim
- [Paper - A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1510.03820.pdf) (2015) Y Zhang
