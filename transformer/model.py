from ospark import Model
from ospark.nn.layers.embedding_layer import EmbeddingLayer
from ospark.utility.weight_initializer import Glorot
from transformer.block.decoding import DecoderBlock
from transformer.block.encoding import EncoderBlock
from typing import Optional
import tensorflow as tf
import numpy as np
from functools import reduce

class TransformerModel(Model):
    def __init__(self,
                 obj_name: str,
                 block_number: int,
                 embedding_size: int,
                 head_number: int,
                 scale_rate: int,
                 max_length: int,
                 encoder_corpus_size: Optional[int] = None,
                 decoder_corpus_size: Optional[int] = None,
                 is_training: Optional[bool]=None,
                 trained_weights: Optional[dict]=None):
        super().__init__(obj_name=obj_name, is_training=is_training, trained_weights=trained_weights)
        self._max_length     = max_length
        self._embedding_size = embedding_size
        self._encoder_blocks = []
        self._decoder_blocks = []
        self._output_layer   = Glorot(obj_name="output_layer",
                                      shape=[embedding_size, decoder_corpus_size+2])
        self._encoder_embedding_layer = EmbeddingLayer(obj_name = "encoder_embedding_layer",
                                                       embedding_dimension = embedding_size,
                                                       corpus_size = encoder_corpus_size+2)
        self._decoder_embedding_layer = EmbeddingLayer(obj_name = "decoder_embedding_layer",
                                                       embedding_dimension = embedding_size,
                                                       corpus_size = decoder_corpus_size+2)

        self._positional_encoding_table = self.create_positional_encoding_table()

        for i in range(block_number):
            encoder_block = EncoderBlock(obj_name=f"encoder_block_{i}",
                                         head_number=head_number,
                                         scale_rate=scale_rate,
                                         embedding_size=embedding_size)
            decoder_block = DecoderBlock(obj_name=f"decoder_block_{i}",
                                         head_number=head_number,
                                         scale_rate=scale_rate,
                                         embedding_size=embedding_size)
            self._encoder_blocks.append(encoder_block)
            self._decoder_blocks.append(decoder_block)

    @property
    def embedding_size(self) -> int:
        return self._embedding_size

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def positional_encoding_table(self) -> tf.Tensor:
        return self._positional_encoding_table

    def create_positional_encoding_table(self) -> tf.Tensor:
        basic_table = np.zeros(shape=[self.max_length, self.embedding_size])
        position    = np.arange(self.max_length).reshape([-1, 1])
        denominator = np.power(10000, np.arange(0, self.embedding_size, 2) / self.embedding_size)
        basic_table[:, 0::2] = np.sin(position / denominator)
        basic_table[:, 1::2] = np.cos(position / denominator)
        return tf.convert_to_tensor(basic_table, dtype=tf.float32)[tf.newaxis, :, :]

    def pipeline(self, encoder_input: tf.Tensor, decoder_input: tf.Tensor) -> tf.Tensor:
        # encoder_input -> [B, L, D]
        # positional_encoding_table -> [1, max_length, D]

        encoder_input = self._encoder_embedding_layer.pipeline(encoder_input)
        decoder_input = self._decoder_embedding_layer.pipeline(decoder_input)
        encoder_input += self.positional_encoding_table[:, tf.shape(encoder_input)[1], :]
        decoder_input += self.positional_encoding_table[:, tf.shape(decoder_input)[1], :]
        encoder_output = reduce(lambda input_data, block: block.pipeline(input_data),
                                self._encoder_blocks,
                                encoder_input)
        decoder_output = reduce(lambda input_data, block: block.pipeline(input_data, encoder_output),
                                self._decoder_blocks,
                                decoder_input)
        output = tf.matmul(decoder_output, self._output_layer)
        result = tf.nn.softmax(output)
        return result


if __name__ == "__main__":
    encoder_input = tf.random.normal(shape = [4,22,128])
    decoder_input = tf.random.truncated_normal(shape = [4,50,128])

    model = TransformerModel(obj_name="transformer",
                             block_number= 4,
                             embedding_size=128,
                             head_number=8,
                             scale_rate=4,
                             max_length=2000,
                             corpus_size=6000
                             )

    model.create()
    print(model.pipeline(encoder_input=encoder_input,decoder_input=decoder_input))