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

mark_dict_len = 2
word_dim = 32
mark_dim = 5
hidden_dim = 512
depth = 8
mix_hidden_lr = 1e-3
base_learning_rate = 0.001

IS_SPARSE = True
PASS_NUM = 10
BATCH_SIZE = 128

embedding_name = 'emb'


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



def db_lstm(word):
    # 8 features
    emb_lr=30.0
    emb = fluid.layers.embedding(
                input=word,
                size=[word_dict_len, word_dim],
                param_attr=fluid.ParamAttr(learning_rate=emb_lr))

    hidden_0 = fluid.layers.fc(input=emb, size=hidden_dim, act='tanh')
    lstm_0 = fluid.layers.dynamic_lstm(
        input=hidden_0,
        size=hidden_dim,
        candidate_activation='relu',
        gate_activation='sigmoid',
        cell_activation='sigmoid')

    # stack L-LSTM and R-LSTM with direct edges
    input_tmp = [hidden_0, lstm_0]

    for i in range(1, depth):
        mix_hidden = fluid.layers.sums(input=[
            fluid.layers.fc(input=input_tmp[0], size=hidden_dim, act='tanh'),
            fluid.layers.fc(input=input_tmp[1], size=hidden_dim, act='tanh')
        ])

        lstm = fluid.layers.dynamic_lstm(
            input=mix_hidden,
            size=hidden_dim,
            candidate_activation='relu',
            gate_activation='sigmoid',
            cell_activation='sigmoid',
            is_reverse=((i % 2) == 1))

        input_tmp = [mix_hidden, lstm]

    feature_out = fluid.layers.sums(input=[
        fluid.layers.fc(input=input_tmp[0], size=label_dict_len, act='tanh'),
        fluid.layers.fc(input=input_tmp[1], size=label_dict_len, act='tanh')
    ])

    return feature_out


def train(use_cuda, save_dirname=None, is_local=True):
    # define data layers
    word = fluid.data(
        name='word_data', shape=[None, 1], dtype='int64', lod_level=1)

    if args.enable_ce:
        fluid.default_startup_program().random_seed = 90
        fluid.default_main_program().random_seed = 90

    # define network topology
    #feature_out = db_lstm(word)
    target = fluid.layers.data(
        name='target', shape=[1], dtype='int64', lod_level=1)
    #crf_cost = fluid.layers.linear_chain_crf(
    #    input=feature_out,
    #    label=target,
    #    param_attr=fluid.ParamAttr(name='crfw', learning_rate=mix_hidden_lr))

    #avg_cost = fluid.layers.mean(crf_cost)

    avg_cost, crf_decode = nets.lex_net(word, args, word_dict_len, label_dict_len, for_infer=False, target=target)

    #sgd_optimizer = fluid.optimizer.SGD(
    #    learning_rate=fluid.layers.exponential_decay(
    #        learning_rate=0.01,
    #        decay_steps=100000,
    #        decay_rate=0.5,
    #        staircase=True))

    #sgd_optimizer.minimize(avg_cost)

    optimizer = fluid.optimizer.Adam(learning_rate=base_learning_rate)
    optimizer.minimize(avg_cost)

    #crf_decode = fluid.layers.crf_decoding(
    #    input=feature_out, param_attr=fluid.ParamAttr(name='crfw'))

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


def infer(use_cuda, save_dirname=None):
    if save_dirname is None:
        return

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)

    inference_scope = fluid.core.Scope()
    with fluid.scope_guard(inference_scope):
        # Use fluid.io.load_inference_model to obtain the inference program desc,
        # the feed_target_names (the names of variables that will be fed
        # data using feed operators), and the fetch_targets (variables that
        # we want to obtain data from using fetch operators).
        #[inference_program, feed_target_names,
        # fetch_targets] = fluid.io.load_inference_model(save_dirname, exe)
        inference_program = fluid.default_main_program()
        word = fluid.data(name='word_data', shape=[None, 1], dtype='int64', lod_level=1)
        crf_decode = nets.lex_net(word, None, word_dict_len, label_dict_len, for_infer=True, target=None)
        fluid.io.load_persistables(exe, save_dirname, inference_program)

        # Setup inputs by creating LoDTensors to represent sequences of words.
        # Here each word is the basic element of these LoDTensors and the shape of
        # each word (base_shape) should be [1] since it is simply an index to
        # look up for the corresponding word vector.
        # Suppose the length_based level of detail (lod) info is set to [[3, 4, 2]],
        # which has only one lod level. Then the created LoDTensors will have only
        # one higher level structure (sequence of words, or sentence) than the basic
        # element (word). Hence the LoDTensor will hold data for three sentences of
        # length 3, 4 and 2, respectively.
        # Note that lod info should be a list of lists.
        # The range of random integers is [low, high]
        testdata = dataset.train()().next()
	data = testdata[0]
        print(np.array(data).shape)
	lod = []
	lod.append(data)
	base_shape = [[len(c) for c in lod]]
	words = fluid.create_lod_tensor(lod, base_shape, place)
        # Construct feed as a dictionary of {feed_target_name: feed_target_data}
        # and results will contain a list of data corresponding to fetch_targets.

        results = exe.run(
            inference_program,
            feed={
                "word_data": words,
            },
            fetch_list=crf_decode,
            return_numpy=False)
        print(results[0].lod())
        np_data = np.array(results[0])
        print("Inference Shape: ", np_data.shape)


def main(use_cuda, is_local=True):
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return

    # Directory for saving the trained model
    save_dirname = "output"

    train(use_cuda, save_dirname, is_local)
    #infer(use_cuda, save_dirname)


if __name__ == '__main__':
    args = parse_args()
    use_cuda = args.use_gpu
    PASS_NUM = args.num_epochs
    main(use_cuda)
