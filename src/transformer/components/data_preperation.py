import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt 
import tensorflow_datasets as tfds 


class DataPreperation:
    def __init__(self) -> None:
        model_name = 'ted_hrlr_translate_pt_en_converter'
        tf.keras.utils.get_file(
            f'{model_name}.zip',
            f'https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip',
            cache_dir='.', cache_subdir='', extract=True
        )
        self.BUFFER_SIZE = 20000
        self.BATCH_SIZE = 64
        self.tokenizers = tf.saved_model.load(model_name)
        
        self.MAX_TOKENS=128
    def prepare_batch(self,pt, en):
        pt = self.tokenizers.pt.tokenize(pt)     
        pt = pt[:, :self.MAX_TOKENS]    
        pt = pt.to_tensor()  
        en = self.tokenizers.en.tokenize(en)
        en = en[:, :(self.MAX_TOKENS+1)]
        en_inputs = en[:, :-1].to_tensor() 
        en_labels = en[:, 1:].to_tensor()  
        return (pt, en_inputs), en_labels
    
    
        
        
        
    def make_batches(self,ds):
        return (
      ds
      .shuffle(self.BUFFER_SIZE)
      .batch(self.BATCH_SIZE)
      .map(self.prepare_batch, tf.data.AUTOTUNE)
      .prefetch(buffer_size=tf.data.AUTOTUNE))