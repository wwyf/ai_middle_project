#!/usr/bin/python3
import os
import sklearn
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import multiprocessing
import utils
from cnn import CNN, CNN2


def run_tensorflow(args, train_data=None, test_data=None):
    # create a tensorflow session
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    row_shapes = []
    if test_data is not None:
        row_shapes += [h.shape[0] for h in test_data]
    if train_data is not None:
        row_shapes += [h.shape[0] for h in train_data]
    max_rows = np.max(row_shapes)
    with tf.Session(config=run_config) as sess:
        if args.debug:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        if not args.use_cnn2:
            network = CNN(
                sess,
                args.vector_size,
                args.output_dim,
                max_rows,
                args.conv_dim,
                args.conv_fmaps,
                args.hidden_units,
                args.hidden_activation,
                args.dropout_rates,
                args.batch_enabled,
                args.batch_size,
                args.conv_activation,
                args.checkpoint_dir,
                args.summary_dir
            )
        else:
            network = CNN2(
                sess,
                args.vector_size,
                args.output_dim,
                max_rows,
                enable_batch=args.batch_enabled,
                batch_size=args.batch_size,
                checkpoint_dir=args.checkpoint_dir,
                summary_dir=args.summary_dir
            )
        network.build_model()
        print("\n".join([str(v) for v in tf.global_variables()]))
        if args.train_cnn:
            network.train(train_data, train_labels, args.learning_rate, args.beta1, args.epochs, 500)
            utils.logger.info("training finished")
            utils.logger.info("start train data prediction test")
            predicted_labels = network.predict(train_data)
            utils.logger.info("scores: f1: %.4f, recall: %.4f, precision: %.4f" % (
                sklearn.metrics.f1_score(train_labels, predicted_labels, average='weighted'),
                sklearn.metrics.recall_score(train_labels, predicted_labels, average='weighted'),
                sklearn.metrics.precision_score(train_labels, predicted_labels, average='weighted')
            ))
        if args.validate_cnn:
            if not args.train_cnn:
                if args.checkpoint_file is not None:
                    if not network.load_file(args.checkpoint_file)[0]:
                        utils.logger.error("specified checkpoint_file is not valid")
                        exit(-1)
                elif not network.load(args.checkpoint_dir)[0]:
                    utils.logger.error("No model available in checkpoint directory, train one first")
                    exit(-1)
            utils.logger.info("start validatation on train data")
            predicted_labels = network.predict(train_data)
            utils.logger.info("scores: f1: %.4f, recall: %.4f, precision: %.4f" % (
                sklearn.metrics.f1_score(train_labels, predicted_labels, average='weighted'),
                sklearn.metrics.recall_score(train_labels, predicted_labels, average='weighted'),
                sklearn.metrics.precision_score(train_labels, predicted_labels, average='weighted')
            ))
        if args.predict_cnn:
            if not args.train_cnn:
                if args.checkpoint_file is not None:
                    if not network.load_file(args.checkpoint_file)[0]:
                        utils.logger.error("specified checkpoint_file is not valid")
                        exit(-1)
                elif not network.load(args.checkpoint_dir)[0]:
                    utils.logger.error("No model available in checkpoint directory, train one first")
                    exit(-1)
            result = network.predict(test_data)
            utils.save_labels(args.o, result)
            utils.logger.info("labels predicted and saved")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-di", help="doc2vec input file path", type=str, action="append")
    parser.add_argument("-dm", help="doc2vec model path", type=str)
    parser.add_argument("-dd", help="doc2vec processed data file path", type=str)
    parser.add_argument("-dt", help="doc2vec processed test file path", type=str)
    parser.add_argument("-wi", help="word2vec input file path", type=str, action="append")
    parser.add_argument("-wm", help="word2vec model path", type=str)
    parser.add_argument("-wd", help="word2vec processed data file path", type=str)
    parser.add_argument("-wt", help="word2vec processed test file path", type=str)

    parser.add_argument("-l", help="label file path", type=str)
    parser.add_argument("-d", help="data file path", type=str)
    parser.add_argument("-t", help="test file path", type=str)
    parser.add_argument("-o", help="test label output path", type=str)
    parser.add_argument("--vector_size", help="doc2vec or word2vec vector size", type=int)
    parser.add_argument("--use_w2v", help="use word2vec in CNN training", action="store_true")
    parser.add_argument("--use_d2v", help="use doc2vec in CNN training", action="store_true")
    parser.add_argument("--debug", help="enable tensorflow debugger", action="store_true")
    parser.add_argument("--conv_dim", help="CNN convolution dimensions", default=[3, 4, 5], nargs='+', type=int)
    parser.add_argument("--output_dim", help="CNN output dimensions", default=2, type=int)
    parser.add_argument("--conv_fmaps", help="CNN convolution featuremap numbers",
                        default=[50, 50, 50], nargs='+', type=int)
    parser.add_argument("--hidden_units", help="CNN FC layer hidden units number", default=[256], nargs='+', type=int)
    parser.add_argument("--hidden_activation", help="CNN FC layer hidden units activation function", default=["tanh"],
                        nargs='+', type=str)
    parser.add_argument("--dropout_rates", help="CNN FC layer dropout rates", default=[0], nargs='+', type=int)
    parser.add_argument("--batch_enabled", help="enable batch normalization layer in CNN", action="store_true")
    parser.add_argument("--batch_size", help="batch normalization size", type=int, default=64)
    parser.add_argument("--conv_activation", help="convolution layer activation function", type=str, default="relu")
    parser.add_argument("--checkpoint_dir", help="checkpoint directory used in training", type=str, default="./checkpoint")
    parser.add_argument("--checkpoint_file", help="checkpoint file used in training", type=str)
    parser.add_argument("--summary_dir", help="summary directory used in training", type=str, default="./summary")
    parser.add_argument("--beta1", help="adam optimizer parameter", type=float, default=0.5)
    parser.add_argument("--learning_rate", help="adam optimizer parameter", type=float, default=0.0001)
    parser.add_argument("--epochs", help="training epochs", type=int, default=50)
    parser.add_argument("--use_cnn2", action="store_true")
    parser.add_argument("--train_d2v", help="train doc2vec model from input file", action="store_true")
    parser.add_argument("--train_w2v", help="train word2vec model from input file", action="store_true")
    parser.add_argument("--train_cnn", help="train CNN model from input labels and data", action="store_true")
    parser.add_argument("--predict_cnn", help="use CNN model to predict labels for test data", action="store_true")
    parser.add_argument("--validate_cnn", action="store_true")

    args = parser.parse_args()
    if args.train_d2v:
        text = []
        for i in args.di:
            text += utils.read_text(i)
        utils.logger.info("text reading finished")
        tokens = utils.tokenize_paragraph_d2v(text)
        utils.logger.info("text tokenizing finished")
        utils.compute_paragraph_doc2vec(tokens,
                                        vector_size=args.vector_size,
                                        epochs=25,
                                        workers=multiprocessing.cpu_count(),
                                        model_path=args.dm)
        utils.logger.info("doc2vec training finished")
    elif args.train_w2v:
        text = []
        for i in args.wi:
            text += utils.read_text(i)
        utils.logger.info("text reading finished")
        tokens = utils.tokenize_paragraph_w2v(text)
        utils.logger.info("text tokenizing finished")
        utils.compute_paragraph_word2vec(tokens,
                                         vector_size=args.vector_size,
                                         workers=multiprocessing.cpu_count(),
                                         model_path=args.wm)
        utils.logger.info("word2vec training finished")

    elif args.train_cnn or args.predict_cnn or args.validate_cnn:
        train_data = None
        test_data = None
        if args.train_cnn or args.validate_cnn:
            loaded = False
            train_labels = utils.read_labels(args.l)
            utils.logger.info("labels reading finished")

            if args.use_d2v and utils.utils.is_path_accessible(args.dd):
                utils.logger.info("try to load from doc2vec train data from specified path")
                train_data = utils.read_array(args.dd)
                utils.logger.info("load doc2vec train data successfully")
                loaded = True
            elif args.use_w2v and utils.is_path_accessible(args.wd):
                utils.logger.info("try to load from word2vec train data from specified path")
                train_data = utils.read_array(args.wd)
                utils.logger.info("load word2vec train data successfully")
                loaded = True

            if not loaded:
                train_text = utils.read_text(args.d)
                utils.logger.info("train text reading finished")

                if args.use_d2v:
                    train_tokens = utils.tokenize_paragraph_d2v(train_text)
                    utils.logger.info("train text tokenizing finished")
                    train_data = utils.compute_paragraph_doc2vec(train_tokens,
                                                                 vector_size=args.vector_size,
                                                                 model_path=args.dm,
                                                                 load_model=True,
                                                                 predict=True)
                    utils.logger.info("train data doc2vec computing finished")
                    if utils.is_path_creatable(args.dd):
                        utils.save_array(args.dd, train_data)
                    utils.logger.info("save doc2vec train data successfully")

                elif args.use_w2v:
                    train_tokens = utils.tokenize_paragraph_w2v(train_text)
                    utils.logger.info("train text tokenizing finished")
                    train_data = utils.compute_paragraph_word2vec(train_tokens,
                                                                  vector_size=args.vector_size,
                                                                  model_path=args.wm,
                                                                  load_model=True,
                                                                  predict=True)
                    utils.logger.info("train data word2vec computing finished")
                    if utils.is_path_creatable(args.wd):
                        utils.save_array(args.wd, train_data)
                    utils.logger.info("save word2vec train data successfully")

        if args.predict_cnn:
            loaded = False

            if args.use_d2v and utils.is_path_accessible(args.dt):
                utils.logger.info("try to load from doc2vec train data from specified path")
                test_data = utils.read_array(args.dt)
                utils.logger.info("load doc2vec test data successfully")
                loaded = True
            elif args.use_w2v and utils.is_path_accessible(args.wt):
                utils.logger.info("try to load from word2vec train data from specified path")
                test_data = utils.read_array(args.wt)
                utils.logger.info("load word2vec test data successfully")
                loaded = True

            if not loaded:
                if args.use_d2v:
                    test_text = utils.read_text(args.t)
                    utils.logger.info("test text reading finished")
                    test_tokens = utils.tokenize_paragraph_d2v(test_text)
                    utils.logger.info("test text tokenizing finished")
                    test_data = utils.compute_paragraph_doc2vec(test_tokens,
                                                                vector_size=args.vector_size,
                                                                model_path=args.dm,
                                                                load_model=True,
                                                                predict=True)
                    utils.logger.info("test data doc2vec computing finished")
                    if utils.is_path_creatable(args.dt):
                        utils.save_array(args.dt, test_data)
                    utils.logger.info("save doc2vec test data successfully")
                elif args.use_w2v:
                    test_text = utils.read_text(args.t)
                    utils.logger.info("test text reading finished")
                    test_tokens = utils.tokenize_paragraph_w2v(test_text)
                    utils.logger.info("test text tokenizing finished")
                    test_data = utils.compute_paragraph_word2vec(test_tokens,
                                                                 vector_size=args.vector_size,
                                                                 model_path=args.wm,
                                                                 load_model=True,
                                                                 predict=True)
                    utils.logger.info("test data word2vec computing finished")
                    if utils.is_path_creatable(args.wt):
                        utils.save_array(args.wt, test_data)
                    utils.logger.info("save word2vec test data successfully")
        # create directories
        if not os.path.exists(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)

        p = multiprocessing.Process(target=run_tensorflow, args=[args, train_data, test_data])
        p.start()
        p.join()
        utils.logger.info("tensorflow process terminated")


