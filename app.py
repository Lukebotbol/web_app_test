from random import choice
import streamlit as st
import pickle
import datasets
import pandas as pd
from fastai.text.all import *
from transformers import *
from blurr.data.all import *
from blurr.modeling.all import *
import pdb
from fastcore.test import *
from fastai import __version__ as fa_version
from torch import __version__ as pt_version
from transformers import __version__ as hft_version
import pathlib

# NLP

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
summarizer = load_learner(
    fname='C:\workings\Dissertation Project\pegasus_model.pkl')


def main():

    st.title('Summarization of Terms and Conditions')

    activities = ['Summarize']
    choice = st.sidebar.selectbox('Select Activity', activities)

    if choice == 'Summarize':
        st.subheader('Summary with NLP')
        raw_text = st.text_area('Enter Text Here', 'Type Here')
        if st.button("Summarize"):
            summary_choice = st.selectbox('Summary Choice', ['PEGASUS'])
            if summary_choice == 'PEGASUS':
                summary_result = summarizer.blurr_generate(raw_text)

            st.success('Generated Summary: \n {}'.format(summary_result[0]))


if __name__ == '__main__':
    main()
