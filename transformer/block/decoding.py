from typing import Optional , NoReturn
from ospark import Block
from ospark.nn.layers.normalization import LayerNormalization
from transformer.layer.attention import Attention, EncoderDecoderAttention
import tensorflow as tf
from transformer.layer.ffn import FeedForward


class DecoderBlock(Block):

    def __init__(self,
                 obj_name: str,
                 head_number: int,
                 scale_rate: int,
                 embedding_size: int,
                 is_training: Optional[bool] = None):
        super(DecoderBlock, self).__init__(obj_name=obj_name, is_training=is_training)
        self._mask_attention = Attention(obj_name="mask_attention",
                                         embedding_size=embedding_size,
                                         head_number=head_number,
                                         use_look_ahead=True,
                                         is_training=is_training)
        self._encode_decode_attention = EncoderDecoderAttention(obj_name="encode_decode_attention",
                                                                embedding_size=embedding_size,
                                                                head_number=head_number,
                                                                use_look_ahead=False,
                                                                is_training=is_training)
        self._ffn = FeedForward(obj_name="ffn_layer",
                                embedding_size=embedding_size,
                                scale_rate=scale_rate,
                                is_training=is_training)

        self._layer_norm = LayerNormalization(layer_dimension=embedding_size)

    def pipeline(self, input_data: tf.Tensor, encoder_output: tf.Tensor) -> tf.Tensor:
        mask_att_output = self._mask_attention.pipeline(input_data=input_data)
        add_norm = self._layer_norm(input_data + mask_att_output)
        encode_decode_output = self._encode_decode_attention.pipeline(input_data=add_norm, encoder_output=encoder_output)
        add_norm = self._layer_norm(add_norm + encode_decode_output)
        ffn_output = self._ffn.pipeline(input_data=add_norm)
        add_norm = self._layer_norm(ffn_output + add_norm)
        return add_norm
