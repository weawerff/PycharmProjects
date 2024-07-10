import tensorflow_datasets as tfds
from transformer.data_generator.data_generator import TranslateDataGenerator
from ospark.nn.loss_function.sparse_categorical_cross_entropy import SparseCategoricalCrossEntropy
from ospark.nn.optimizer.learning_rate_schedule import TransformerWarmup
import tensorflow as tf
from transformer.model import TransformerModel
from transformer.trainer import Trainer

folder_path = "C:/Users/Lola/PycharmProjects/Transformer_model"
dataset_name = "wmt14_translate/de-en"
train_vocab_file = "de_vocab_file"
target_vocab_file = "en_vocab_file"
vocabulary = 40000
scale_rate = 4

# with open(os.path.join(weight_path, file_name), "r") as fp:
#   weights = json.load(fp)

ds = tfds.load(data_dir=folder_path,
               name=dataset_name,
               as_supervised=True)

training_dataset = ds["train"]
print(training_dataset, ds)

# train_data, target_data = None, None
# train_data, target_data = zip(*[[train_data.numpy(), target_data.numpy()]
#                                            for i, (train_data, target_data)
#                                           in enumerate(training_dataset)])

dataset = []
for i, (train_data, target_data) in enumerate(training_dataset):
    if i < 100:
        dataset.append([train_data.numpy(), target_data.numpy()])
    else:
        break
train_data, target_data = zip(*[dataset[0], dataset[1]])

data_generator = TranslateDataGenerator.create_from_dataset(train_data=train_data,
                                                            target_data=target_data,
                                                            target_data_vocab_size=40000,
                                                            train_data_vocab_size=40000,
                                                            batch_size=1)

learning_rate = TransformerWarmup(model_dimension=128, warmup_step=4000.)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.98, beta_2=0.9)
loss_function = SparseCategoricalCrossEntropy()

model = TransformerModel(obj_name="transformer",
                         block_number=4,
                         embedding_size=128,
                         head_number=8,
                         scale_rate=4,
                         max_length=2000,
                         encoder_corpus_size=data_generator.data_encoder.train_data_encoder.vocab_size,
                         decoder_corpus_size=data_generator.data_encoder.label_data_encoder.vocab_size)

trainer = Trainer(model = model,
                 data_generator = data_generator,
                 epoch_number =50,
                 optimizer = optimizer,
                 loss_function = loss_function,
                 save_times = 4,
                 save_path = folder_path+"/weight.json",
                 use_auto_graph = True,
                 use_multi_gpu = None,
                 devices = None)


trainer.start()

