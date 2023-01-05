import numpy as np
from mltu.metrics import CERMetric, WERMetric

from mltu.utils.text_utils import get_wer as wer

import cv2
import typing
import numpy as np
import tensorflow as tf

if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm
)

  

    # sentences_true = ['helo love', 'helo home', 'helo world']
    # sentences_pred = ['helo python', 'helo home', 'helo python here']

    # def to_embeddings(sentences, vocab):
    #     embeddings, max_len = [], 0

    #     for sentence in sentences:
    #         embedding = []
    #         for character in sentence:
    #             embedding.append(vocab.index(character))
    #         embeddings.append(embedding)
    #         max_len = max(max_len, len(embedding))
    #     return embeddings, max_len

    # vocab = set()
    # for sen in sentences_true + sentences_pred:
    #     for character in sen:
    #         vocab.add(character)
    # vocab = "".join(vocab)

    # sen1, max_len = to_embeddings(sentences_true, vocab)
    # sen2, _ = to_embeddings(sentences_pred, vocab)

    # sen_true = [np.pad(sen, (0, max_len - len(sen)), 'constant', constant_values=len(vocab)) for sen in sen1]
    # sen_pred = [np.pad(sen, (0, 24 - len(sen)), 'constant', constant_values=-1) for sen in sen2]


    # tf_vocab = tf.constant(list(vocab))

    # distance = WERMetric.get_wer(sen_pred, sen_true, vocab=tf_vocab)

    # d = wer(sentences_pred, sentences_true)

    # print(list(distance.numpy()))
    # print(d)


    word_true = [
        [1, 2, 3, 4, 5, 6, 1],
        [2, 3, 4, 5, 6, 1, 1]
    ]
    word_pred = [
        [1, 2, 3, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [2, 3, 4, 5, 6,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    ]
    vocabulary = tf.constant(list("abcdefg"))

    distance = CERMetric.get_cer(word_pred, word_true, vocabulary)