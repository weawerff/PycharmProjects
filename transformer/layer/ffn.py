from ospark import Layer
from typing import Optional
from ospark.utility.weight_initializer import glorot_uniform, zeros
import tensorflow as tf


class FeedForward(Layer):

    def __init__(self,
                 obj_name: str,
                 embedding_size: int,
                 scale_rate:int,
                 is_training: Optional[bool] = None):
        super().__init__(obj_name=obj_name, is_training=is_training)
        self._embedding_size = embedding_size

        self.w_1 = glorot_uniform(obj_name="w_1_weight", shape=[embedding_size, scale_rate * embedding_size], trainable=is_training)
        self.w_2 = glorot_uniform(obj_name="w_2_weight", shape=[scale_rate * embedding_size, embedding_size], trainable=is_training)
        self.b_1= zeros(obj_name="w_1", shape=[1, scale_rate * embedding_size], trainable=is_training)
        self.b_2 = zeros(obj_name="w_2", shape=[1, embedding_size], trainable=is_training)

    @property
    def embedding_size(self) -> int:
        return self._embedding_size

    def pipeline(self, input_data:tf.Tensor) ->tf.Tensor:
        matmul_output = tf.matmul(input_data, self.w_1) + self.b_1
        relu_output   = tf.nn.relu(matmul_output)
        output       = tf.matmul(relu_output, self.w_2) + self.b_2
        return output