#-- coding: utf-8 -*-

import argparse
import atexit
import logging

from hbconfig import Config
import tensorflow as tf

import data_loader
import hook
from model import Model
import utils


def experiment_fn(run_config, params):

    model = Model()
    estimator = tf.estimator.Estimator(
            model_fn=model.model_fn,
            model_dir=Config.train.model_dir,
            params=params,
            config=run_config)

    vocab = data_loader.load_vocab("vocab")
    Config.data.vocab_size = len(vocab)

    train_data, test_data = data_loader.make_train_and_test_set()
    train_input_fn, train_input_hook = data_loader.make_batch(train_data,
                                                              batch_size=Config.model.batch_size,
                                                              scope="train")
    test_input_fn, test_input_hook = data_loader.make_batch(test_data,
                                                            batch_size=Config.model.batch_size,
                                                            scope="test")

    train_hooks = [train_input_hook]
    if Config.train.print_verbose:
        train_hooks.append(hook.print_variables(
            variables=['train/input_0'],
            rev_vocab=get_rev_vocab(vocab),
            every_n_iter=Config.train.check_hook_n_iter))
        train_hooks.append(hook.print_target(
            variables=['train/target_0', 'train/pred_0'],
            every_n_iter=Config.train.check_hook_n_iter))
    if Config.train.debug:
        train_hooks.append(tf_debug.LocalCLIDebugHook())

    eval_hooks = [test_input_hook]
    if Config.train.debug:
        eval_hooks.append(tf_debug.LocalCLIDebugHook())

    experiment = tf.contrib.learn.Experiment(
        estimator=estimator,
        train_input_fn=train_input_fn,
        eval_input_fn=test_input_fn,
        train_steps=Config.train.train_steps,
        min_eval_frequency=Config.train.min_eval_frequency,
        train_monitors=train_hooks,
        eval_hooks=eval_hooks
    )
    return experiment


def get_rev_vocab(vocab):
    if vocab is None:
        return None
    return {idx: key for key, idx in vocab.items()}

def re_config(args):
    # Model Re Config
    if args.batchsize != 0:
        Config.model.batch_size = args.batchsize
    if args.embeddim != 0:
        Config.model.embed_dim = args.embeddim
    if args.numfilters != 0:
        Config.model.num_filters = args.numfilters
    if args.dropout != 0:
        Config.model.dropout = args.dropout
    # Train Re Config
    if args.learnrate != 0:
        Config.train.learning_rate = args.learnrate
    if args.trainsteps != 0:
        Config.train.train_steps = args.trainsteps
    if args.savecheck != 0:
        Config.train.save_checkpoints_steps = args.savecheck
    if args.evalcheck != 0:
        Config.train.min_eval_frequency = args.evalcheck

def main(mode):
    params = tf.contrib.training.HParams(**Config.model.to_dict())

    run_config = tf.contrib.learn.RunConfig(
            model_dir=Config.train.model_dir,
            save_checkpoints_steps=Config.train.save_checkpoints_steps)

    tf.contrib.learn.learn_runner.run(
        experiment_fn=experiment_fn,
        run_config=run_config,
        schedule=mode,
        hparams=params
    )


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Basic Arguments
    parser.add_argument('--config', type=str, default='config',
                        help='config file name')
    parser.add_argument('--mode', type=str, default='train',
                        help='Mode (train/test/train_and_evaluate)')
    # Model Arguments
    parser.add_argument('--batchsize', type=int, default=0,
                        help="Batch size")
    parser.add_argument('--embeddim', type=int, default=0,
                        help="Embedded dimensions")
    parser.add_argument('--numfilters', type=int, default=0,
                        help="Number of filters")
    parser.add_argument('--dropout', type=int, default=0,
                        help="Dropout rate")
    # Training Arguments
    parser.add_argument('--learnrate', type=int, default=0,
                        help="Learn rate")
    parser.add_argument('--trainsteps', type=int, default=0,
                        help="Training steps")
    parser.add_argument('--savecheck', type=int, default=0,
                        help="Number of steps before model is saved")
    parser.add_argument('--evalcheck', type=int, default=0,
                        help="Number of steps before model is evaluated")
    args = parser.parse_args()

    tf.logging._logger.setLevel(logging.INFO)

    # Print Config setting
    Config(args.config)
    re_config(args)
    print("Config: ", Config)
    if Config.get("description", None):
        print("Config Description")
        for key, value in Config.description.items():
            print(f" - {key}: {value}")

    # After terminated Notification to Slack
    atexit.register(utils.send_message_to_slack, config_name=args.config)

    main(args.mode, args)
