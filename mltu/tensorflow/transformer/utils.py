import tensorflow as tf


class MaskedLoss(tf.keras.losses.Loss):
    """ Masked loss function for Transformer.

    Args:
        mask_value (int, optional): Mask value. Defaults to 0.
        reduction (str, optional): Reduction method. Defaults to 'none'.
    """
    def __init__(self, mask_value: int=0, reduction: str='none') -> None:
        super(MaskedLoss, self).__init__()
        self.mask_value = mask_value
        self.reduction = reduction
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=reduction)

    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight=None) -> tf.Tensor:
        """ Calculate masked loss.
        
        Args:
            y_true (tf.Tensor): True labels.
            y_pred (tf.Tensor): Predicted labels.

        Returns:
            tf.Tensor: Masked loss.
        """
        mask = y_true != self.mask_value
        loss = self.loss_object(y_true, y_pred)

        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask

        loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
        return loss


class MaskedAccuracy(tf.keras.metrics.Metric):
    """ Masked accuracy metric for Transformer.

    Args:
        mask_value (int, optional): Mask value. Defaults to 0.
        name (str, optional): Name of the metric. Defaults to 'masked_accuracy'.
    """
    def __init__(self, mask_value: int=0, name: str='masked_accuracy') -> None:
        super(MaskedAccuracy, self).__init__(name=name)
        self.mask_value = mask_value
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    @tf.function
    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight=None):
        """ Update state of the metric.

        Args:
            y_true (tf.Tensor): True labels.
            y_pred (tf.Tensor): Predicted labels.
        """
        pred = tf.argmax(y_pred, axis=2)
        label = tf.cast(y_true, pred.dtype)
        match = label == pred

        mask = label != self.mask_value

        match = match & mask

        match = tf.cast(match, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        match = tf.reduce_sum(match)
        mask = tf.reduce_sum(mask)

        self.total.assign_add(match)
        self.count.assign_add(mask)

    def result(self) -> tf.Tensor:
        """ Calculate masked accuracy.

        Returns:
            tf.Tensor: Masked accuracy.
        """
        return self.total / self.count
    

class CERMetric(tf.keras.metrics.Metric):
    """A custom TensorFlow metric to compute the Character Error Rate (CER).
    
    Args:
        vocabulary: A string of the vocabulary used to encode the labels.
        name: (Optional) string name of the metric instance.
        **kwargs: Additional keyword arguments.
    """
    def __init__(self, end_token, padding_token: int=0, name="CER", **kwargs):
        # Initialize the base Metric class
        super(CERMetric, self).__init__(name=name, **kwargs)
        
        # Initialize variables to keep track of the cumulative character/word error rates and counter
        self.cer_accumulator = tf.Variable(0.0, name="cer_accumulator", dtype=tf.float32)
        self.batch_counter = tf.Variable(0, name="batch_counter", dtype=tf.int32)
        
        self.padding_token = padding_token
        self.end_token = end_token

    def get_cer(self, pred, y_true, padding=-1):
        """ Calculates the character error rate (CER) between the predicted labels and true labels for a batch of input data.

        Args:
            pred(tf.Tensor): The predicted labels, with dtype=tf.int32, usually output from tf.keras.backend.ctc_decode
            y_true (tf.Tensor): The true labels, with dtype=tf.int32
            padding (int, optional): The padding token when converting to sparse tensor. Defaults to -1.

        Returns:
            tf.Tensor: The CER between the predicted labels and true labels
        """
        # find index where end token is
        equal = tf.equal(pred, self.end_token)
        equal_int = tf.cast(equal, tf.int64)
        end_token_index = tf.argmax(equal_int, axis=1)

        # mask out everything after end token
        new_range = tf.range(tf.shape(pred)[1], dtype=tf.int64)
        range_matrix = tf.tile(new_range[None, :], [tf.shape(pred)[0], 1])

        mask = range_matrix <= tf.expand_dims(end_token_index, axis=1)
        masked_pred = tf.where(mask, pred, padding)

        # Convert the valid predicted labels tensor to a sparse tensor
        sparse_pred = tf.RaggedTensor.from_tensor(masked_pred, padding=padding).to_sparse()

        # Convert the valid true labels tensor to a sparse tensor
        sparse_true = tf.RaggedTensor.from_tensor(y_true, padding=padding).to_sparse()

        # Calculate the normalized edit distance between the sparse predicted labels tensor and sparse true labels tensor
        distance = tf.edit_distance(sparse_pred, sparse_true, normalize=True)

        return distance

    # @tf.function
    def update_state(self, y_true, y_pred, sample_weight=None):
        """Updates the state variables of the metric.

        Args:
            y_true: A tensor of true labels with shape (batch_size, sequence_length).
            y_pred: A tensor of predicted labels with shape (batch_size, sequence_length, num_classes).
            sample_weight: (Optional) a tensor of weights with shape (batch_size, sequence_length).
        """
        pred = tf.argmax(y_pred, axis=2)

        # Calculate the normalized edit distance between the predicted labels and true labels tensors
        distance = self.get_cer(pred, y_true, self.padding_token)

        # Add the sum of the distance tensor to the cer_accumulator variable
        self.cer_accumulator.assign_add(tf.reduce_sum(distance))
        
        # Increment the batch_counter by the batch size
        self.batch_counter.assign_add(len(y_true))

    def result(self):
        """ Computes and returns the metric result.

        Returns:
            A TensorFlow float representing the CER (character error rate).
        """
        return tf.math.divide_no_nan(self.cer_accumulator, tf.cast(self.batch_counter, tf.float32))