import tensorflow as tf

class CWERMetric(tf.keras.metrics.Metric):
    """ A custom TensorFlow metric to compute the Character and Word Error Rate (CWER)
    """
    def __init__(self, name='CWER', **kwargs):
        super(CWERMetric, self).__init__(name=name, **kwargs)
        self.cer_accumulator = tf.Variable(0.0, name="cer_accumulator", dtype=tf.float32)
        self.wer_accumulator = tf.Variable(0.0, name="wer_accumulator", dtype=tf.float32)
        self.counter = tf.Variable(0, name="counter", dtype=tf.int32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        input_shape = tf.keras.backend.shape(y_pred)

        input_length = tf.ones(shape=input_shape[0], dtype='int32') * tf.cast(input_shape[1], 'int32')

        decode, log = tf.keras.backend.ctc_decode(y_pred, input_length, greedy=True)

        decode = tf.keras.backend.ctc_label_dense_to_sparse(decode[0], input_length)
        y_true_sparse = tf.cast(tf.keras.backend.ctc_label_dense_to_sparse(y_true, input_length), "int64")

        decode = tf.sparse.retain(decode, tf.not_equal(decode.values, -1))
        distance = tf.edit_distance(decode, y_true_sparse, normalize=True)

        correct_words_amount = tf.reduce_sum(tf.cast(tf.not_equal(distance, 0), tf.float32))

        self.wer_accumulator.assign_add(correct_words_amount)
        self.cer_accumulator.assign_add(tf.reduce_sum(distance))
        self.counter.assign_add(len(y_true))

    def result(self):
        return {
                "CER": tf.math.divide_no_nan(self.cer_accumulator, tf.cast(self.counter, tf.float32)),
                "WER": tf.math.divide_no_nan(self.wer_accumulator, tf.cast(self.counter, tf.float32))
        }

class CERMetric(tf.keras.metrics.Metric):
    """A custom TensorFlow metric to compute the Character Error Rate (CER).
    
    Args:
        padding_token: An integer representing the padding token in the input data.
        name: (Optional) string name of the metric instance.
        **kwargs: Additional keyword arguments.
    """
    def __init__(self, padding_token, name='CER', **kwargs):
        # Initialize the base Metric class
        super(CERMetric, self).__init__(name=name, **kwargs)
        
        # Initialize variables to keep track of the cumulative character error rate and counter
        self.cumulative_error_rate = tf.Variable(0.0, name="cumulative_error_rate", dtype=tf.float32)
        self.batch_counter = tf.Variable(0, name="batch_counter", dtype=tf.int32)
        
        # Store the padding token as an attribute
        self.padding_token = padding_token

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Updates the state variables of the metric.

        Args:
            y_true: A tensor of true labels with shape (batch_size, sequence_length).
            y_pred: A tensor of predicted labels with shape (batch_size, sequence_length, num_classes).
            sample_weight: (Optional) a tensor of weights with shape (batch_size, sequence_length).
        """
        # Get the input shape and length
        input_shape = tf.keras.backend.shape(y_pred)
        input_length = tf.ones(shape=input_shape[0], dtype='int32') * tf.cast(input_shape[1], 'int32')

        # Decode the predicted labels using greedy decoding
        decode_predicted, log = tf.keras.backend.ctc_decode(y_pred, input_length, greedy=True)

        # Convert the dense decode tensor to a sparse tensor
        predicted_labels_sparse = tf.keras.backend.ctc_label_dense_to_sparse(decode_predicted[0], input_length)
        
        # Convert the dense true labels tensor to a sparse tensor and cast to int64
        true_labels_sparse = tf.cast(tf.keras.backend.ctc_label_dense_to_sparse(y_true, input_length), "int64")

        # Retain only the non-padding elements in the predicted labels tensor
        predicted_labels_sparse = tf.sparse.retain(predicted_labels_sparse, tf.not_equal(predicted_labels_sparse.values, -1))
        
        # Retain only the non-padding elements in the true labels tensor
        true_labels_sparse = tf.sparse.retain(true_labels_sparse, tf.not_equal(true_labels_sparse.values, self.padding_token))

        # Calculate the normalized edit distance between the predicted labels and true labels tensors
        distance = tf.edit_distance(predicted_labels_sparse, true_labels_sparse, normalize=True)

        # Add the sum of the distance tensor to the cumulative_error_rate variable
        self.cumulative_error_rate.assign_add(tf.reduce_sum(distance))
        
        # Increment the batch_counter by the batch size
        self.batch_counter.assign_add(len(y_true))

    def result(self):
        """Computes and returns the metric result.

        Returns:
            The average character error rate (CER) over all batches seen so far.
        """
        # Return the cumulative character error rate divided by the batch_counter
        return tf.math.divide_no_nan(self.cumulative_error_rate, tf.cast(self.batch_counter, tf.float32))