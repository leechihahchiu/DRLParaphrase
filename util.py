# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""This file contains some utility functions"""

import tensorflow as tf
import time
import os
import numpy as np
from utils import rouge
import data
import nltk
import edit_distance
from scipy.stats import rankdata
FLAGS = tf.app.flags.FLAGS
import jieba as jb

def get_config():
    """Returns config for tf.session"""
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth=True
    return config

def load_ckpt(saver, sess, load_gen=True):
    """Load checkpoint from the ckpt_dir (if unspecified, this is train dir) and restore it to saver and sess, waiting 10 secs in the case of failure. Also returns checkpoint name."""
    while True:
        try:
            if load_gen:
                train_dir = os.path.join(FLAGS.log_root, FLAGS.exp_name, 'train')
            else:
                train_dir = os.path.join(FLAGS.rank_log_root, 'train')
            ckpt_state = tf.train.get_checkpoint_state(train_dir)

            #latest_filename = "checkpoint_best" if ckpt_dir=="eval" else None
           # ckpt_dir = os.path.join(FLAGS.log_root, ckpt_dir)
            #ckpt_state = tf.train.get_checkpoint_state(ckpt_dir, latest_filename=latest_filename)
            tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
            saver.restore(sess, ckpt_state.model_checkpoint_path)
            return ckpt_state.model_checkpoint_path
        except:
            tf.logging.info("Failed to load checkpoint from %s. Sleeping for %i secs...", train_dir, 10)
            time.sleep(10)

def dynamic_padding(list_of_ids, to_len, pad_id):
    def pad_seq(ids):
        while len(ids) < to_len:
            ids.append(pad_id)
        return ids

    lens = [len(list(ids)) for ids in list_of_ids]
    lens = [x+1 if x<to_len else x for x in lens]
    padded_ids = [np.array(pad_seq(list(ids))) for ids in list_of_ids]
    return np.array(padded_ids), lens

def cal_rouge_score(tg_ids, out_ids, out_lens, tg_lens, n=1):
    rslts = []
    for tg, out, tg_len, out_len in zip(tg_ids, out_ids, tg_lens, out_lens):
        try:
            score = rouge.rouge_n(
                [' '.join([str(x) for x in list(out[ :int(out_len)-1])])],
                [' '.join([str(x) for x in list(out[ :int(tg_len)-1])])],
                n=n)[0]
        except:
            score = 0
        rslts.append(score)
    return np.array(rslts)

def prepare_retrain_data(hps, dec_batch, dec_batch_extended_vocab, vocab, lens):
    bs = dec_batch.shape[0]
    start_decoding = np.expand_dims(np.array([vocab.word2id(data.START_DECODING)]*bs), axis=1)
    stop_decoding = np.expand_dims(np.array([vocab.word2id(data.STOP_DECOING)]*bs), axis=1)
    dec_inp = np.concatenate(
        (start_decoding, dec_batch),
        axis=1)
    target = dec_batch_extended_vocab[:, :hps.max_enc_steps]
    padding = np.zeros((bs, hps.max_enc_steps))
    for i, len_ in enumerate(lens):
        padding[i, :len_] = 1
        if len_ <= hps.max_enc_steps:
            target[i, len_-1] = vocab.word2id(data.STOP_DECODING)
            if len_ < hps.max_enc_steps:
                target[i, len_:] = vocab.word2id(data.PAD_TOKEN)
        if len_ + 1 <= hps.max_enc_steps:
            dec_inp[i, len_+1:] = vocab.word2id(data.PAD_TOKEN)

    return dec_inp[:, :hps.max_enc_steps], target, padding

def convert_to_full_vocab(id_arr, full_vdict, vocab, art_oovs_list):
    full_ids = id_arr.copy()
    for i, ids in enumerate(id_arr):
        full_ids[i, :] = np.array( [full_vdict[x] for x in data.outputids2words(ids, vocab, art_oovs_list[i])] )
    return full_ids

def measure_len(output_ids, vocab):
    len_ = []
    for ids in output_ids:
        try: 
            stop_idx = list(ids).index(vocab.word2id(data.STOP_DECODING))
            len_.append(stop_idx+1)
    return len_

def concat(sent1, sent2, pad_id, axis=0):
    sz1 = sent1.shape[1]
    sz2 = sent2.shape[1]
    if sz1 > sz2:
        sent2, _ = dynamic_padding(sent2, sz1, pad_id)
    elif sz2 > sz1:
        sent1, _ = dynamic_padding(sent1, sz2, pad_id)
    return np.concatenate((sent1, sent2), axis=axis)



def cal_rouge_l_score(tg_ids, out_ids, out_lens, tg_lens, n=1):
    rslts = []
    for tg, out, tg_len, out_len in zip(tg_ids, out_ids, tg_lens, out_lens):
        try:
            score = rouge.rouge_l_sentence_level(
                [' '.join([str(x) for x in list(out[ :int(out_len)-1])])],
                [' '.join([str(x) for x in list(out[ :int(tg_len)-1])])])[0]
        except:
            score = 0
        rslts.append(score)
    return np.array(rslts)

def cal_bleu_score(tg_ids, out_ids, out_lens, tg_lens):
    rslts = []
    for tg, out, tg_len, out_len in zip(tg_ids, out_ids, tg_lens, out_lens):
        score = nltk.translate.bleu_score.sentence_bleu(
            [ [str(x) for x in list(tg[: int(tg_len)-1 ])]],
            [ [str(x) for x in list(out[: int(out_len)-1 ])]],
            weights=(0.5, 0.5),
            smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method1
            )
        rslts.append(score)
    return np.array(rslts)


def get_checkpoints(checkpoint):
    txt = open(checkpoint).read().strip()
    checkpoints = [x.split()[1].replace('"', '').replace('"', '') for x in txt.split('\n')]
    return checkpoints

def restore_model(sess, saver, file_, var, save=False):
    try:
        saver.restore(sess, file_)
    except:
        if FLAGS.opt == 'adam':
            saved_vars = [v for v in var if not 'Adam' in v.name and not 'beta1' in v.name and not 'beta2' in v.name]
        else:
            saved_vars = var
        saver = tf.train.Saver(saved_vars)
        saver.restore(sess, file_)
        if save:
            opt_vars = [v for v in var if not v in saved_vars]
            sess.run(tf.varaibles_initializer(opt_vars))
            saver = tf.train.Saver(var)

    return saver

def rescale_reward(r, base_r, padding, scale=1.):
    rank = np.zeros_like(r)
    decode_len = np.sum(padding, axis=1)

    for i, l in enumearte(r):
        rank[i, :] = rankdata(-l, 'min')
    if FLAGS.sample_temo_ratio == 1:
        B = np.expand_dims(np.sum(padding, axis=1), axis=1)
    else:
        B = np.max(rank, aixs=1)
        B[np.where(decode_len!=20)] = B[np.where(decode_len!=20)] - 1
        B = np.expand_dims(B, axis=1)

    tmp = rank.astype(np.float64) / B
    x = scale * (0.5 - (rank.astype(np.float64) / len(r) ))
    sigmoid = 1. / ( 1 + np.exp(-x*padding)) + np.expand_dims(base_r, axis=1) - 0.5
    return 2 * sigmoid - np.expand_dims(base_r, axis=1)

#def rank_sentence(r, scale=1., bias=0.5):
#    rank = rankdata(-r, 'min')
#    x = scale * (0)












