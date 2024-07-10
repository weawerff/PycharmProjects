from ospark import Layer
from typing import Optional, Tuple
from ospark.utility.weight_initializer import glorot_uniform, zeros
import tensorflow as tf

class Attention(Layer):
    def __init__(self,
                 obj_name: str,
                 embedding_size: int,
                 head_number: int,
                 use_look_ahead: Optional[bool] = None,
                 is_training: Optional[bool] = None) -> object:
        super().__init__(obj_name=obj_name, is_training=is_training)
        self._embedding_size = embedding_size
        self._head_number    = head_number
        self._use_look_ahead = use_look_ahead if use_look_ahead is not None else False
        assert embedding_size % head_number == 0
        self._depth          = int(embedding_size / head_number)


        self._q_weight = glorot_uniform(obj_name="q_weight", shape=[embedding_size, embedding_size], trainable=is_training)
        self._v_weight = glorot_uniform(obj_name="v_weight", shape=[embedding_size, embedding_size], trainable=is_training)
        self._k_weight = glorot_uniform(obj_name="k_weight", shape=[embedding_size, embedding_size], trainable=is_training)

        self._linear_weight = glorot_uniform(obj_name="linear_weight", shape=[embedding_size, embedding_size], trainable=is_training)

        self._q_bias = zeros(obj_name="q_bias", shape=[1, embedding_size], trainable=is_training)
        self._k_bias = zeros(obj_name="k_bias", shape=[1, embedding_size], trainable=is_training)
        self._v_bias = zeros(obj_name="v_bias", shape=[1, embedding_size], trainable=is_training)

        self._linear_bias = zeros(obj_name="linear_bias", shape=[1, embedding_size], trainable=is_training)

    @property
    def embedding_size(self) -> int:
        return self._embedding_size

    @property
    def use_look_ahead(self) -> bool:
        return self._use_look_ahead

    @property
    def head_number(self) -> int:
        return self._head_number

    @property
    def sequence_length(self) -> tf.Tensor:
        return self._sequence_length

    @property
    def look_ahead_matrix(self) -> tf.Tensor:
        return 1 - tf.linalg.band_part(tf.ones(shape=[self.sequence_length, self.sequence_length]), num_lower=-1, num_upper=0)

    def pipeline(self, input_data: tf.Tensor) -> tf.Tensor:
        # input_data -> [B, L, D]
        batch_size = tf.shape(input_data)[0]

        q, k, v = self.process_qkv(input_data, batch_size)

        # [B, H, L, d] -> d = 512 / H

        # Q * transpose(K) -> [B, H, L, d] * [B, H, d, L] -> [B, H, L, L]
        qk_att = tf.matmul(q, tf.transpose(k, [0, 1, 3, 2])) / tf.sqrt(tf.cast(self.embedding_size, tf.float32))
        if self.use_look_ahead:
            self._sequence_length = tf.shape(q)[-2]
            mask = self.look_ahead_matrix * tf.constant(-1e9)
            qk_att += mask

        qk_att = tf.nn.softmax(qk_att, axis=-1)
        # [B, H, L, L] * [B, H, L, d] -> [B, H, L, d]
        att_result = tf.matmul(qk_att, v)

        # [B, H, L, d] -> [B, L, D]
        result = self.concat_att_result(att_result=att_result, batch_size=batch_size)
        result = tf.matmul(result, self._linear_weight) + self._linear_bias
        return result

    def process_qkv(self, input_data: tf.Tensor, batch_size: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        # input_data -> [B, L, D]

        q = tf.matmul(input_data, self._q_weight) + self._q_bias
        k = tf.matmul(input_data, self._k_weight) + self._k_bias
        v = tf.matmul(input_data, self._v_weight) + self._v_bias
        q = self.split_head(input_data=q, batch_size=batch_size)
        k = self.split_head(input_data=k, batch_size=batch_size)
        v = self.split_head(input_data=v, batch_size=batch_size)
        return q, k, v

    def split_head(self, input_data: tf.Tensor, batch_size: tf.Tensor) -> tf.Tensor:
        split_result = tf.reshape(input_data, [batch_size, -1, self.head_number, self._depth]) # [B, L, H, d]
        split_result = tf.transpose(split_result, [0, 2, 1, 3])  # [B, L, H, d] -> [B, H, L, d]
        return split_result

    def concat_att_result(self, att_result: tf.Tensor, batch_size: tf.Tensor) -> tf.Tensor:
        # input_data -> [B, H, L, d] -> [B, L, H, d]
        result = tf.transpose(att_result, [0, 2, 1, 3])
        # [B, L, H, d] -> [B, L, D] D = self.embedding_size
        result = tf.reshape(result, [batch_size, -1, self.embedding_size])
        return result

class EncoderDecoderAttention(Attention):

    def pipeline(self, input_data: tf.Tensor, encoder_output: tf.Tensor) -> tf.Tensor:
        # input_data -> [B, L, D]
        batch_size = tf.shape(input_data)[0]

        q, k, v = self.process_qkv(input_data, encoder_output, batch_size)

        # [B, H, L, d] -> d = 512 / H

        # Q * transpose(K) -> [B, H, L, d] * [B, H, d, L] -> [B, H, L, L]

        qk_att = tf.nn.softmax(tf.matmul(q, tf.transpose(k, [0, 1, 3, 2])), axis=-1) / tf.sqrt(tf.cast(self.embedding_size, tf.float32))
        # [B, H, L, L] * [B, H, L, d] -> [B, H, L, d]
        att_result = tf.matmul(qk_att, v)

        # [B, H, L, d] -> [B, L, D]
        result = self.concat_att_result(att_result=att_result, batch_size=batch_size)
        result = tf.matmul(result, self._linear_weight) + self._linear_bias
        return result

    def process_qkv(self, input_data: tf.Tensor, encoder_output: tf.Tensor, batch_size: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        # input_data -> [B, L, D]

        q = tf.matmul(input_data, self._q_weight) + self._q_bias
        k = tf.matmul(encoder_output, self._k_weight) + self._k_bias
        v = tf.matmul(encoder_output, self._v_weight) + self._v_bias
        q = self.split_head(input_data=q, batch_size=batch_size)
        k = self.split_head(input_data=k, batch_size=batch_size)
        v = self.split_head(input_data=v, batch_size=batch_size)
        return q, k, v