import unittest
import numpy as np
from mltu.tensorflow.metrics import CERMetric, WERMetric

import numpy as np
import tensorflow as tf

class TestMetrics(unittest.TestCase):

    def to_embeddings(self, sentences, vocab):
        embeddings, max_len = [], 0

        for sentence in sentences:
            embedding = []
            for character in sentence:
                embedding.append(vocab.index(character))
            embeddings.append(embedding)
            max_len = max(max_len, len(embedding))
        return embeddings, max_len

    def setUp(self) -> None:
        true_words = ["Who are you", "I am a student", "I am a teacher", "Just different sentence length"]
        pred_words = ["Who are you", "I am a ztudent", "I am A reacher", "Just different length"]

        vocab = set()
        for sen in true_words + pred_words:
            for character in sen:
                vocab.add(character)
        self.vocab = "".join(vocab)

        sentence_true, max_len_true = self.to_embeddings(true_words, self.vocab)
        sentence_pred, max_len_pred = self.to_embeddings(pred_words, self.vocab)

        max_len = max(max_len_true, max_len_pred)
        padding_length = 64

        self.sen_true = [np.pad(sen, (0, max_len - len(sen)), "constant", constant_values=len(self.vocab)) for sen in sentence_true]
        self.sen_pred = [np.pad(sen, (0, padding_length - len(sen)), "constant", constant_values=-1) for sen in sentence_pred]

    def test_CERMetric(self):
        vocabulary = tf.constant(list(self.vocab))
        cer = CERMetric.get_cer(self.sen_true, self.sen_pred, vocabulary).numpy()

        self.assertTrue(np.array_equal(cer, np.array([0.0, 0.071428575, 0.14285715, 0.42857143], dtype=np.float32)))

    def test_WERMetric(self):
        vocabulary = tf.constant(list(self.vocab))
        wer = WERMetric.get_wer(self.sen_true, self.sen_pred, vocabulary).numpy()

        self.assertTrue(np.array_equal(wer, np.array([0., 0.25, 0.5, 0.33333334], dtype=np.float32)))

if __name__ == "__main__":
    unittest.main()