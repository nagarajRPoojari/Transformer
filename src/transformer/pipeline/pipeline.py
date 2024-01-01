
import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt 
import tensorflow_datasets as tfds 
from transformer.components.data_embedding import *
from transformer.components.Attention import *
from transformer.components.encoder_decoder import *
from transformer.components.model import *
from transformer.components.data_preperation import *


class Translator:
    def __init__(self):
        num_layers = 4
        d_model = 128
        dff = 512
        num_heads = 8
        dropout_rate = 0.1
        learning_rate = CustomSchedule(d_model)
    
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)
        data_prepare=DataPreperation()
        tokenizers=data_prepare.tokenizers
        transformer = Transformer(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            input_vocab_size=tokenizers.pt.get_vocab_size().numpy(),
            target_vocab_size=tokenizers.en.get_vocab_size().numpy(),
            dropout_rate=dropout_rate)