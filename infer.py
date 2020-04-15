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
        inference_program = fluid.default_main_program()
        word = fluid.data(name='word_data', shape=[None, 1], dtype='int64', lod_level=1)
        crf_decode = nets.lex_net(word, None, word_dict_len, label_dict_len, for_infer=True, target=None)
        fluid.io.load_persistables(exe, save_dirname, inference_program, filename="__params__")

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
    main(use_cuda)
