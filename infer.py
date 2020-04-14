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
        '--num_epochs', type=int, default=10, help="number of epochs.")
    args = parser.parse_args()
    return args



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
        words = []
        for i in range(len(testdata[0])):
            idx = testdata[0][i]
            words.append(idx_word[idx] + idx_label[testdata[1][i]])
        print(" ".join(words))

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
        labels = []
        for idx in np_data:
            label = idx_label[idx[0]]
            labels.append(label)
        print(" ".join(labels))


def main(use_cuda, is_local=True):
    if use_cuda and not fluid.core.is_compiled_with_cuda():
        return

    # Directory for saving the trained model
    save_dirname = "output"

    infer(use_cuda, save_dirname)


if __name__ == '__main__':
    args = parse_args()
    use_cuda = args.use_gpu
    PASS_NUM = args.num_epochs
    main(use_cuda)
