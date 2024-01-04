# older version but the other one wasn't improving training 

import tensorflow as tf
from tensorflow.keras import losses

class WeightedSCCE(losses.Loss):
    def __init__(self, class_weight, from_logits=False, name='weighted_scce'):
        if class_weight is None or all(v == 1. for v in class_weight):
            self.class_weight = None
        else:
            self.class_weight = tf.convert_to_tensor(class_weight, dtype=tf.float32)

        self.name = name
        self.from_logits = from_logits
        self.reduction = losses.Reduction.NONE
        self.unreduced_scce = losses.SparseCategoricalCrossentropy(
            from_logits=from_logits, name=name, reduction=self.reduction)
        
    def __call__(self, y_true, y_pred, sample_weight=None):

        loss = self.unreduced_scce(y_true, y_pred, sample_weight)
    
        if self.class_weight is not None:
            weight_mask = tf.gather(self.class_weight, y_true)
            loss = tf.math.multiply(loss, weight_mask)
    
        # Compute the mean loss across batch dimensions
        return tf.reduce_mean(loss)