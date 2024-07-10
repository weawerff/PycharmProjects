from transformer.data_generator.data_generator import TranslateDataGenerator
from typing import List
import numpy as np


train_data: List[str] = ["不經歷風雨，怎麼遇見彩虹", "相聚有時，後會無期"]
target_data: List[str] = ["No cross, no crown.", "Sometime ever, sometime never."]
data_generator = TranslateDataGenerator.create_from_dataset(train_data =train_data,
                                                            target_data = target_data,
                                                            target_data_vocab_size = 265,
                                                            train_data_vocab_size = 265,
                                                            batch_size = 2)

for data in data_generator:
    print(data)
    print(data.training_data, data.target_data, data.length)
    for data in data.training_data:
        print(data)
        print(data_generator.data_encoder.train_data_encoder.decode(np.array(data)[1:-1]))