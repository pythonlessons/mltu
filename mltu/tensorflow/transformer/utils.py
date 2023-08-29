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