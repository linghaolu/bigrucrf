from __future__ import print_function

import math, os
import numpy as np
import paddle
import paddle.dataset.conll05 as conll05
import paddle.fluid as fluid
import six
import time
import argparse

import dataset
import nets

np.set_printoptions(suppress=True)

with_gpu = os.getenv('WITH_GPU', '0') != '0'

word_dict, label_dict, idx_word, idx_label = dataset.build_dict()
word_dict_len = len(word_dict)
label_dict_len = len(label_dict)

base_learning_rate = 0.001
PASS_NUM = 10
BATCH_SIZE = 128

def parse_args():
    parser = argparse.ArgumentParser("label_semantic_roles")
    parser.add_argument(
        '--enable_ce',
        action='store_true',
        help="If set, run the task with continuous evaluation logs.")
    parser.add_argument(
        '--use_gpu', type=int, default=0, help="Whether to use GPU or not.")
    parser.add_argument(
        '--num_epochs', type=int, default=100, help="number of epochs.")
    args = parser.parse_args()
    return args


def train(use_cuda, save_dirname=None, is_local=True):
    # define data layers
    word = fluid.data(
        name='word_data', shape=[None, 1], dtype='int64', lod_level=1)

    if args.enable_ce:
        fluid.default_startup_program().random_seed = 90
        fluid.default_main_program().random_seed = 90

    # define network topology
    target = fluid.layers.data(
        name='target', shape=[1], dtype='int64', lod_level=1)

    avg_cost, crf_decode = nets.lex_net(word, args, word_dict_len, label_dict_len, for_infer=False, target=target)

    optimizer = fluid.optimizer.Adam(learning_rate=base_learning_rate)
    optimizer.minimize(avg_cost)

    if args.enable_ce:
        train_data = paddle.batch(
            dataset.train(), batch_size=BATCH_SIZE)
    else:
        train_data = paddle.batch(
            paddle.reader.shuffle(dataset.train(), buf_size=1024),
            batch_size=BATCH_SIZE)

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

    feeder = fluid.DataFeeder(
        feed_list=[
            word, target
        ],
        place=place)
    exe = fluid.Executor(place)

    def train_loop(main_program):
        exe.run(fluid.default_startup_program())

        start_time = time.time()
        batch_id = 0
        for pass_id in six.moves.xrange(PASS_NUM):
            print("------")
            for data in train_data():
                cost = exe.run(
                    main_program, feed=feeder.feed(data), fetch_list=[avg_cost])
                cost = cost[0]

                if batch_id % 10 == 0:
                    print("avg_cost:" + str(cost))
                    if batch_id != 0:
                        print("second per batch: " + str((
                            time.time() - start_time) / batch_id))
                    # Set the threshold low to speed up the CI test
                    if float(cost) < 60.0:
                        if args.enable_ce:
                            print("kpis\ttrain_cost\t%f" % cost)

                        if save_dirname is not None:
                            # TODO(liuyiqun): Change the target to crf_decode
                            fluid.io.save_inference_model(save_dirname, ['word_data'], 
                                    crf_decode, exe)
                        #return

                batch_id = batch_id + 1

    train_loop(fluid.default_main_program())


def main(use_cuda, is_local=True):
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return

    # Directory for saving the trained model
    save_dirname = "output"

    train(use_cuda, save_dirname, is_local)


if __name__ == '__main__':
    args = parse_args()
    use_cuda = args.use_gpu
    PASS_NUM = args.num_epochs
    main(use_cuda)
