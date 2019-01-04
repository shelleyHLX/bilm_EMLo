
import argparse

import numpy as np

from bilm_model.training import train, load_options_latest_checkpoint, load_vocab
from bilm_model.data import BidirectionalLMDataset


def main(args):
    # load the vocab
    vocab = load_vocab(args.vocab_file, None)

    # define the options
    batch_size = 256  # batch size for each GPU
    n_gpus = 1

    # number of tokens in training data (this for 1B Word Benchmark)
    n_train_tokens = 14273184  #  100000*16

    options = {
     'bidirectional': True,

     # 'char_cnn': {'activation': 'relu',
     #  'embedding': {'dim': 200},
     #  'filters': [[1, 32],
     #   [2, 32],
     #   [3, 64],
     #   [4, 128],
     #   [5, 256],
     #   [6, 512],
     #   [7, 1024]],
     #  'max_characters_per_token': 50,
     #  'n_characters': 261,
     #  'n_highway': 2},
    
     'dropout': 0.1,
    
     'lstm': {
      'cell_clip': 3,
      'dim': 4096,
      'n_layers': 2,
      'proj_clip': 3,
      'projection_dim': 200,  # 512
      'use_skip_connections': True},
    
     'all_clip_norm_val': 10.0,
    
     'n_epochs': 3,
     'n_train_tokens': n_train_tokens,
     'batch_size': batch_size,
     'n_tokens_vocab': vocab.size,
     'unroll_steps': 20,  # 5
     'n_negative_samples_batch': 8192,
    }

    prefix = args.train_prefix  # '../corpus_me/wd_fact_cut.txt'
    data = BidirectionalLMDataset(prefix, vocab, test=False, shuffle_on_load=True)
    print('load data BidirectionalLMDataset')
    tf_save_dir = args.save_dir
    tf_log_dir = args.save_dir
    # 加载模并训练
    train(options, data, n_gpus, tf_save_dir=tf_save_dir, tf_log_dir=tf_log_dir, restart_ckpt_file=args.save_dir)
    # 训练模型
    # train(options, data, n_gpus, tf_save_dir=tf_save_dir, tf_log_dir=tf_log_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', help='Location of checkpoint files', default='../try4/')
    parser.add_argument('--vocab_file', help='Vocabulary file', default='../corpus_me/vocab_elmo.txt')
    parser.add_argument('--train_prefix', help='Prefix for train files',
                        default='../corpus_me/wd_fact_cut.txt')  # ')
    # 2800000.txt  wd_fact_cut  100000/128*10 33450

    args = parser.parse_args()
    main(args)

""" step 1
    export CUDA_VISIBLE_DEVICES=4
    python bin/train_elmo.py \
    --train_prefix='../corpus_me/wd_fact_cut.txt' \
    --vocab_file ../corpus_me/vocab_elmo.txt \
    --save_dir ../try4/ >bilm_out.txt 2>&1 &
    参数         值       控制台输出写入文件
"""
""" step 2
    python  bin/dump_weights.py  \
    --save_dir ../try4/  \
    --outfile ../try4/weights.hdf5 >outfile.txt 2>&1 &
"""
""" step 3
测试 usage_token.py
"""