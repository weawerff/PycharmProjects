from typing import Optional, NoReturn
from ospark import Block
from ospark.nn.layers.normalization import LayerNormalization
from transformer.layer.attention import Attention
import tensorflow as tf
from transformer.layer.ffn import FeedForward

class EncoderBlock(Block):
    def __init__(self,
                obj_name:str,
                head_number:int,
                scale_rate:int,
                embedding_size:int,
                is_training: Optional[bool] = None):
        super(EncoderBlock,self).__init__(obj_name = obj_name, is_training = is_training)
        self.norm = LayerNormalization(layer_dimension = embedding_size)
        self.attention = Attention(obj_name = "attention_layer",
                                   embedding_size = embedding_size,
                                   head_number = head_number,
                                   is_training = is_training)
        self.ffn = FeedForward(obj_name = "feedforward_layer",
                               embedding_size = embedding_size,
                               scale_rate = scale_rate,
                               is_training = is_training)

    def pipeline(self, input_data:tf.Tensor) -> tf.Tensor:
        att_output = self.attention.pipeline(input_data)
        add_norm = self.norm(att_output + input_data)
        ffn_output = self.ffn.pipeline(add_norm)
        add_norm = self.norm(ffn_output + add_norm)
        return add_norm