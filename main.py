from transformers import pipeline
import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt 
import tensorflow_datasets as tfds 
import argparse
from transformer.logging import logger
import streamlit as st
import warnings
from contextlib import contextmanager


    
@contextmanager
def suppress_streamlit_warnings():
    original_showwarning = warnings.showwarning
    warnings.showwarning = lambda *args, **kwargs: None
    yield
    warnings.showwarning = original_showwarning
    
    
@st.cache(allow_output_mutation=True)
def load_model():
    predictor =  pipeline("translation", model="fubuki119/opus-mt-en-hi")
    return predictor


model = load_model()
input_text = st.text_area("Enter your text:")



if st.button("Generate Summary"):
    if input_text:
        t = model(input_text)
        st.write(t[0]['translation_text'])

    else:
        st.warning("Please enter text or upload a text or PDF file for summarization.")