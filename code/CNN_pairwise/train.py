from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import logging

import numpy as np
#from six.moves import range  # pylint: disable=redefined-builtin
import tensorflow as tf

import data_utils
import argparse
import copy
import collections
from gensim.models import Word2Vec
from model import CNN
from gensim.models import KeyedVectors

FLAGS = None

def add_arguments(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument("--data_dir", type=str, default="/home/wangtm/code/DBQA", help="Data directory")
    parser.add_argument("--model_dir", type=str, default="/S1/LCWM/wangtm/model/", help="Model directory")
    parser.add_argument("--out_dir", type=str, default="/S1/LCWM/wangtm/output/", help="Out directory")
    parser.add_argument("--train_dir", type=str, default="DBQA/cnn/", help="Training directory")
    parser.add_argument("--gpu_device", type=str, default="2", help="which gpu to use")
    parser.add_argument("--train_question", type=str, default="train_q",
                        help="Training data_src path")
    parser.add_argument("--train_answer", type=str, default="train_a",
                        help="Training data_dst path")
    parser.add_argument("--train_label", type=str, default="train_l",
                        help="Training data path")

    parser.add_argument("--valid_question", type=str, default="valid_q_tokens",
                        help="Training data_src path")
    parser.add_argument("--valid_answer", type=str, default="valid_a_tokens",
                        help="Training data_dst path")
    parser.add_argument("--valid_label", type=str, default="valid_labels",
                        help="Training data path")

    parser.add_argument("--test_question", type=str, default="test_q_tokens",
                        help="Training data_src path")
    parser.add_argument("--test_answer", type=str, default="test_a_tokens",
                        help="Training data_dst path")
    parser.add_argument("--test_label", type=str, default="test_labels",
                        help="Training data path")

    parser.add_argument("--from_vocab", type=str, default="vocab_60000",
                        help="from vocab path")
    parser.add_argument("--to_vocab", type=str, default="vocab_60000",
                        help="to vocab path")
    parser.add_argument("--output_dir", type=str, default="DBQA/cnn/")
    parser.add_argument("--cnn_ckpt_dir", type=str, default="DBQA/cnn/",
                        help="model checkpoint directory")
    parser.add_argument("--gan_ckpt_dir", type=str, default="DBQA/cnn/",
                        help="model checkpoint directory")
    parser.add_argument("--dis_ckpt_dir", type=str, default="DBQA/cnn/")
    parser.add_argument("--d_ckpt_dir", type=str, default="DBQA/cnn/",
                        help="model checkpoint directory")


    parser.add_argument("--max_train_data_size", type=int, default=0, help="Limit on the size of training data (0: no limit)")
    parser.add_argument("--attention", type=str, default="", help="""\
          luong | scaled_luong | bahdanau | normed_bahdanau or set to "" for no
          attention\
          """)
    parser.add_argument("--from_vocab_size", type=int, default=60000, help="NormalWiki vocabulary size")
    parser.add_argument("--to_vocab_size", type=int, default=60000, help="SimpleWiki vocabulary size")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers in the model")
    parser.add_argument("--num_units", type=int, default=256, help="Size of each model layer")
    parser.add_argument("--emb_dim", type=int, default=300, help="Dimension of word embedding")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size to use during training")
    parser.add_argument("--max_gradient_norm", type=float, default=5.0, help="Clip gradients to this norm")
    parser.add_argument("--learning_rate_decay_factor", type=float, default=0.5, help="Learning rate decays by this much")
    parser.add_argument("--learning_rate", type=float, default=1, help="Learning rate")
    parser.add_argument("--num_buckets", type=int, default=1, help="Number of buckets")
    parser.add_argument("--src_max_len", type=int, default=500, help="Maximum length of source sentence")
    parser.add_argument("--tgt_max_len", type=int, default=200, help="Maximum length of target sentence")
    parser.add_argument("--input_keep_prob", type=float, default=0.7, help="Dropout input keep prob")
    parser.add_argument("--output_keep_prob", type=float, default=1.0, help="Dropout output keep prob")
    parser.add_argument("--epoch_num", type=int, default=100, help="Number of epoch")
    parser.add_argument("--lambda1", type=float, default=0.5)
    parser.add_argument("--lambda2", type=float, default=0.5)


    parser.add_argument("--num_train_epoch", type=int, default=100, help="Number of epoch for training")
    parser.add_argument("--steps_per_eval", type=int, default=2000, help="How many training steps to do per eval/checkpoint")

_filter_sizes = [1, 2, 3]
_num_filters = [100, 100, 100]



def safe_exp(value):
  """Exponentiation with catching of overflow error."""
  try:
    ans = math.exp(value)
  except OverflowError:
    ans = float("inf")
  return ans

def get_config_proto(log_device_placement=False, allow_soft_placement=True):
  # GPU options:
  # https://www.tensorflow.org/versions/r0.10/how_tos/using_gpu/index.html
  config_proto = tf.ConfigProto(
      log_device_placement=log_device_placement,
      allow_soft_placement=allow_soft_placement)
  config_proto.gpu_options.allow_growth = True
  return config_proto



class TrainModel(
    collections.namedtuple("TrainModel",
                           ("graph", "model"))):
  pass

class EvalModel(
    collections.namedtuple("EvalModel",
                           ("graph", "model"))):
  pass

class InferModel(
    collections.namedtuple("InferModel",
                           ("graph", "model"))):
  pass

def create_model(hparams, model):
    print("Creating generator model...")
    train_graph = tf.Graph()
    with train_graph.as_default():
        train_model = model(hparams, tf.contrib.learn.ModeKeys.TRAIN)

    eval_graph = tf.Graph()
    with eval_graph.as_default():
        eval_model = model(hparams, tf.contrib.learn.ModeKeys.EVAL)

    infer_graph = tf.Graph()
    with infer_graph.as_default():
        infer_model = model(hparams, tf.contrib.learn.ModeKeys.INFER)

    return TrainModel(graph=train_graph, model=train_model), EvalModel(graph=eval_graph, model=eval_model), InferModel(
        graph=infer_graph, model=infer_model)

def read_data(q_path, a_path, l_path, vocab_path):
    data_set = []
    max_length1, max_length2 = 0, 0
    pos = []
    neg = []
    last_q = []
    from_vocab, rev_from_vocab = data_utils.initialize_vocabulary(vocab_path)
    with tf.gfile.GFile(q_path, mode="r") as q_file:
        with tf.gfile.GFile(a_path, mode="r") as a_file:
            with tf.gfile.GFile(l_path, mode="r") as l_file:
                    q, a, l = q_file.readline().rstrip("\n"), a_file.readline().rstrip("\n"), l_file.readline().rstrip(
                        "\n")
                    counter = 0
                    while q or a or l:
                        # print(q)
                        # print(a)
                        q_ids = []
                        a_ids = []
                        if len(q.split()) >= max_length1:
                            max_length1 = len(q.split())
                        if len(a.split()) >= max_length2:
                            max_length2 = len(a.split())
                        for x in q.split():
                            if x == "":
                                continue
                            if x in from_vocab:
                                q_ids.append(from_vocab[x])
                            else:
                                q_ids.append(data_utils.UNK_ID)
                        for x in a.split():
                            if x == "":
                                continue
                            if x in from_vocab:
                                a_ids.append(from_vocab[x])
                            else:
                                a_ids.append(data_utils.UNK_ID)
                        try:
                            label = int(l)
                        except:
                            label = 0
                        # print(q_ids)
                        if len(a_ids) >= 50:
                            a_ids = a_ids[:50]
                        if q_ids != last_q and last_q != []:
                            # print(len(pos), len(neg))
                            for p in pos:
                                for n in neg:
                                    data_set.append([last_q, p, n])
                                    counter += 1
                                    if counter % 10000 == 0:
                                        print("counter %d" % counter)
                            last_q = q_ids
                            pos = []
                            neg = []
                            if label == 1:
                                pos.append(a_ids)
                            else:
                                neg.append(a_ids)
                        else:
                            last_q = q_ids
                            if label == 1:
                                pos.append(a_ids)
                            else:
                                neg.append(a_ids)
                        q, a, l = q_file.readline().rstrip("\n"), a_file.readline().rstrip(
                            "\n"), l_file.readline().rstrip(
                            "\n")
    for p in pos:
        for n in neg:
            data_set.append([last_q, p, n])
            counter += 1
    print(counter)
    print(max_length1, max_length2)
    return data_set

def read_infer_data(q_path, a_path, l_path, vocab_path):
    data_set = []
    max_length1, max_length2 = 0, 0
    pos = []
    neg = []
    last_q = []
    labels = []
    from_vocab, rev_from_vocab = data_utils.initialize_vocabulary(vocab_path)
    with tf.gfile.GFile(q_path, mode="r") as q_file:
        with tf.gfile.GFile(a_path, mode="r") as a_file:
            with tf.gfile.GFile(l_path, mode="r") as l_file:
                    q, a, l = q_file.readline().rstrip("\n"), a_file.readline().rstrip("\n"), l_file.readline().rstrip(
                        "\n")
                    counter = 0
                    while q or a or l :
                        # print(q)
                        # print(a)
                        q_ids = []
                        a_ids = []
                        if len(q.split()) >= max_length1:
                            max_length1 = len(q.split())
                        if len(a.split()) >= max_length2:
                            max_length2 = len(a.split())
                        for x in q.split():
                            if x == "":
                                continue
                            if x in from_vocab:
                                q_ids.append(from_vocab[x])
                            else:
                                q_ids.append(data_utils.UNK_ID)
                        for x in a.split():
                            if x == "":
                                continue
                            if x in from_vocab:
                                a_ids.append(from_vocab[x])
                            else:
                                a_ids.append(data_utils.UNK_ID)
                        try:
                            label = int(l)
                        except:
                            label = 0
                        if len(q_ids) >= 40:
                            q_ids = q_ids[:40]
                        if len(a_ids) >= 50:
                            a_ids = a_ids[:50]
                        counter += 1
                        data_set.append([q_ids, a_ids, a_ids])
                        labels.append(label)
                        q, a, l = q_file.readline().rstrip("\n"), a_file.readline().rstrip(
                            "\n"), l_file.readline().rstrip(
                            "\n")
    print(counter)
    print(max_length1, max_length2)
    return data_set, labels

def train(hparams, train=True, interact=False):
    # model_creator = adversarial_nets.Seq2Seq
    embeddings = init_embedding(hparams)
    hparams.add_hparam(name="embeddings", value=embeddings)



    hparams.add_hparam(name="cnn_ckpt_path", value=os.path.join(hparams.cnn_ckpt_dir, "cnn.ckpt"))


    train_set = read_data("%s/%s" % (hparams.data_dir, hparams.train_question),
                          "%s/%s" % (hparams.data_dir, hparams.train_answer),
                          "%s/%s" % (hparams.data_dir, hparams.train_label),

                          "%s/%s" % (hparams.data_dir, hparams.from_vocab))
    valid_set = read_data("%s/%s" % (hparams.data_dir, hparams.valid_question),
                          "%s/%s" % (hparams.data_dir, hparams.valid_answer),
                          "%s/%s" % (hparams.data_dir, hparams.valid_label),

                          "%s/%s" % (hparams.data_dir, hparams.from_vocab))
    test_set, test_label = read_infer_data("%s/%s" % (hparams.data_dir, hparams.test_question),
                          "%s/%s" % (hparams.data_dir, hparams.test_answer),
                          "%s/%s" % (hparams.data_dir, hparams.test_label),

                          "%s/%s" % (hparams.data_dir, hparams.from_vocab))
    # for i in range(35):
    #     print(valid_set[i])
    # print()
    # for i in range(36):
    #     print(test_set[i])
    #     print(test_label[i])
    # print()
    hparams.add_hparam(name="filter_sizes", value=_filter_sizes)
    hparams.add_hparam(name="num_filters", value=_num_filters)

    ######################################### D train   ###############################################################
    # train_model, eval_model, infer_model = create_model(hparams, RNN_CNN)
    # train_model, eval_model, infer_model = create_model(hparams, RNN)
    train_model, eval_model, infer_model = create_model(hparams, CNN)
    config = get_config_proto(
        log_device_placement=False)
    train_sess = tf.Session(config=config, graph=train_model.graph)
    eval_sess = tf.Session(config=config, graph=eval_model.graph)
    infer_sess = tf.Session(config=config, graph=infer_model.graph)
    ckpt = tf.train.get_checkpoint_state(hparams.cnn_ckpt_dir)
    with train_model.graph.as_default():
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            train_model.model.saver.restore(train_sess, ckpt.model_checkpoint_path)
            eval_model.model.saver.restore(eval_sess, ckpt.model_checkpoint_path)
            infer_model.model.saver.restore(infer_sess, ckpt.model_checkpoint_path)
            global_step = train_model.model.global_step.eval(session=train_sess)
        else:
            train_sess.run(tf.global_variables_initializer())
            global_step = 0
    # outfile = open("%s/test_%d" % (hparams.output_dir, global_step), "w",
    #                encoding="utf-8")
    # total_acc = 0
    # acc = 0
    # all = 0
    # for id in range(0, int(len(test_set) / hparams.batch_size)):
    #     predict, step_acc = eval_model.model.infer_step(eval_sess, test_set, id * hparams.batch_size, no_random=True)
    #     for i in range(0, hparams.batch_size):
    #         _, _, _, answer = test_set[id * hparams.batch_size + i]
    #         outfile.write(str(predict[i][1]) + "\n")
    #         if answer == 1:
    #             all += 1
    #             if predict[i][1] >= 0.5:
    #                 acc += 1
    #     total_acc += step_acc
    #
    # outfile.close()
    # print("infer acc %.3f  check acc %.3f" % (total_acc / int(len(test_set) / hparams.batch_size), acc * 1.0 / all))
    step_loss, step_time, total_acc, total_loss, total_time, avg_loss, avg_time = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    vocab_path = "%s/%s" % (hparams.data_dir, hparams.from_vocab)
    last_ppl = 100000
    to_vocab, rev_to_vocab = data_utils.initialize_vocabulary(vocab_path)


    while global_step <= 100000:
        start_time = time.time()
        step_loss, global_step, step_acc = train_model.model.train_step(train_sess, train_set)
        total_loss += step_loss
        total_time += (time.time() - start_time)
        total_acc += step_acc
        if global_step % 100 == 0:
            avg_loss = total_loss / 100
            avg_time = total_time / 100
            avg_acc = total_acc / 100
            total_loss, total_acc, total_time = 0.0, 0.0, 0.0
            print("global step %d   step-time %.2fs  loss %.3f   acc %.3f" % (global_step, avg_time, avg_loss, avg_acc))

        if global_step % 1000 == 0:
            train_model.model.saver.save(train_sess, hparams.cnn_ckpt_path, global_step=global_step)
            ckpt = tf.train.get_checkpoint_state(hparams.cnn_ckpt_dir)
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                eval_model.model.saver.restore(eval_sess, ckpt.model_checkpoint_path)
                infer_model.model.saver.restore(infer_sess, ckpt.model_checkpoint_path)
                print("load eval model.")
            else:
                raise ValueError("ckpt file not found.")
            total_loss, total_acc, total_time = 0.0, 0.0, 0.0
            for i in range(0, int(len(valid_set) / hparams.batch_size)):
                step_loss, step_acc = eval_model.model.eval_step(eval_sess, valid_set, i * hparams.batch_size, no_random=True)
                total_loss += step_loss
                total_acc += step_acc
            avg_loss = total_loss / (len(valid_set) / hparams.batch_size)
            avg_acc = total_acc / (len(valid_set) / hparams.batch_size)
            total_loss, total_acc, total_time = 0.0, 0.0, 0.0
            print("eval  loss %.3f  acc %.3f" % (avg_loss, avg_acc))

            outfile = open("%s/predict_%d" % (hparams.output_dir, global_step), "w",
                       encoding="utf-8")

            total_acc = 0
            acc = 0
            all = 0
            for id in range(0, int(len(test_set) / hparams.batch_size)):
                predict, step_acc = eval_model.model.infer_step(eval_sess, test_set, id * hparams.batch_size, no_random=True)
                for i in range(0, hparams.batch_size):
                    answer = test_label[id * hparams.batch_size + i]
                    outfile.write(str(predict[i]) + "\n")
                    if answer == 1:
                        all += 1
                        if predict[i] >= 0.5:
                            acc += 1
                total_acc += step_acc

            outfile.close()
            print("infer acc %.3f  check acc %.3f" % (total_acc/ int(len(test_set) / hparams.batch_size), acc * 1.0 / all))
            total_acc = 0
            print("infer done.")

    rnnlm_train = (train_model, train_sess)
    rnnlm_eval = (eval_model, eval_sess)
    rnnlm_infer = (infer_model, infer_sess)
    rnnlm = (rnnlm_train, rnnlm_eval, rnnlm_infer)
    return rnnlm




def init_embedding(hparams):
    f = open("vocab_60000", "r", encoding="utf-8")
    vocab = []
    for line in f:
        vocab.append(line.rstrip("\n"))
    # word_vectors = KeyedVectors.load_word2vec_format("news_12g_baidubaike_20g_novel_90g_embedding_64.bin", binary=True)
    lines_num, dim = 0, 0
    vectors = {}
    with open("sgns.baidubaike.bigram-char", encoding='utf-8', errors='ignore') as f:
        first_line = True
        for line in f:
            if first_line:
                first_line = False
                dim = int(line.rstrip().split()[1])
                continue
            lines_num += 1
            tokens = line.rstrip().split(' ')
            vectors[tokens[0]] = np.asarray([float(x) for x in tokens[1:]]).astype(np.float32)

    # model = Word2Vec(sentences=sent, sg=1, size=256, window=5, min_count=3, hs=1)
    # model.save("word2vec")
    emb = []
    ct = 0
    for i in range(0, len(vocab)):
        word = vocab[i]
        if word in vectors:
            emb.append(vectors[word])
            ct += 1
        else:
            emb.append((np.random.random([hparams.emb_dim]) * 2 - 1.0).astype(np.float32))
    # emb = []
    # pad = np.random.random([hparams.emb_dim]) - 0.5
    # emb.append(pad.astype(np.float32))
    # go = np.random.random([hparams.emb_dim]) - 0.5
    # emb.append(go.astype(np.float32))
    # eos = np.random.random([hparams.emb_dim]) - 0.5
    # emb.append(eos.astype(np.float32))
    # unk = np.random.random([hparams.emb_dim]) - 0.5
    # emb.append(unk.astype(np.float32))
    # model = Word2Vec.load("word2vec")
    # for i in range(4, len(vocab)):
    #     word = vocab[i]
    #     # if word == "fiasco":
    #     #     continue
    #     emb.append(model.wv[word].astype(np.float32))
    print(ct)
    print(" init embedding finished")
    emb = np.array(emb)
    print(emb.shape)
    return emb



def create_hparams(flags):
    return tf.contrib.training.HParams(
        # dir path
        data_dir=flags.data_dir,
        train_dir=flags.train_dir,
        cnn_ckpt_dir=flags.cnn_ckpt_dir,
        dis_ckpt_dir=flags.dis_ckpt_dir,
        gan_ckpt_dir=flags.gan_ckpt_dir,
        output_dir=flags.output_dir,

        # data params
        batch_size=flags.batch_size,
        from_vocab_size=flags.from_vocab_size,
        to_vocab_size=flags.to_vocab_size,
        GO_ID=data_utils.GO_ID,
        EOS_ID=data_utils.EOS_ID,
        PAD_ID=data_utils.PAD_ID,
        emb_dim=flags.emb_dim,
        max_train_data_size=flags.max_train_data_size,
        num_train_epoch=flags.num_train_epoch,
        steps_per_eval=flags.steps_per_eval,
        train_question=flags.train_question,
        train_answer=flags.train_answer,
        train_label=flags.train_label,

        valid_question=flags.valid_question,
        valid_answer=flags.valid_answer,
        valid_label=flags.valid_label,

        test_question=flags.test_question,
        test_answer=flags.test_answer,
        test_label=flags.test_label,

        from_vocab=flags.from_vocab,
        to_vocab=flags.to_vocab,
        share_vocab=True,

        # model params
        input_keep_prob=flags.input_keep_prob,
        output_keep_prob=flags.output_keep_prob,
        init_weight=0.1,
        num_buckets=flags.num_buckets,
        num_units=flags.num_units,
        num_layers=flags.num_layers,
        learning_rate=flags.learning_rate,
        clip_value=flags.max_gradient_norm,
        decay_factor=flags.learning_rate_decay_factor,
        src_max_len=flags.src_max_len,
        tgt_max_len=flags.tgt_max_len,
        max_seq_length=20,
        #train params
        epoch_num=flags.epoch_num,
        epoch_step=0,
        lambda1=flags.lambda1,
        lambda2=flags.lambda2
    )

def main(_):

    hparams = create_hparams(FLAGS)
    train(hparams)
    #encode_story(hparams)

if __name__ == "__main__":
    my_parser = argparse.ArgumentParser()
    add_arguments(my_parser)
    FLAGS, remaining = my_parser.parse_known_args()
    FLAGS.cnn_ckpt_dir = FLAGS.model_dir + FLAGS.cnn_ckpt_dir
    FLAGS.d_ckpt_dir = FLAGS.model_dir + FLAGS.d_ckpt_dir
    FLAGS.dis_ckpt_dir = FLAGS.model_dir + FLAGS.dis_ckpt_dir
    FLAGS.gan_ckpt_dir = FLAGS.model_dir + FLAGS.gan_ckpt_dir
    FLAGS.train_dir = FLAGS.model_dir + FLAGS.train_dir
    FLAGS.output_dir = FLAGS.out_dir + FLAGS.output_dir
    print(FLAGS)
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_device
    tf.app.run()