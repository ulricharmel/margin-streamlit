from helper import *

#importing all the helper fxn from helper.py which we will create later

import streamlit as st
import os
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile

sns.set_theme(style="darkgrid")

sns.set()

from PIL import Image

st.title('Road condition Classifier')

current_path = os.getcwd()

def save_uploaded_file(uploaded_file):
    
    try:
        with open(os.path.join(current_path, 'static/images',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1    
    except Exception as e:
        st.write(f"{e}")        
        return 0


def main():
    uploaded_file = st.file_uploader("Upload Image")
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=True)
        tfile.write(uploaded_file.read())
        st.sidebar.write('Please wait for the magic to happen! This may take up to a minute.')
        
        display_image = Image.open(tfile)
        
        st.image(display_image)

        prediction, prob = predictor(tfile)

        st.text(f'Model predicts {prediction} with a {prob}% confidence!')


    # # text over upload button "Upload Image"

    # st.write("## Task")

    # st.write(f"{uploaded_file}")
    # if uploaded_file is not None:

    #     if save_uploaded_file(uploaded_file): 
    #         st.write("## Does it get here")

    #         # display the image

    #         display_image = Image.open(uploaded_file)

    #         st.image(display_image)

    #         prediction, prob = predictor(os.path.join(current_path, 'static/images',uploaded_file.name))

    #         os.remove(current_path+'/static/images/'+uploaded_file.name)

    #         # deleting uploaded saved picture after prediction

    #         # drawing graphs

    #         st.text(f'Model predicts {prediction} with a {prob}% confidence!')

    #         # fig, ax = plt.subplots()

    #         # ax  = sns.barplot(y = 'name',x='values', data = prediction,order = prediction.sort_values('values',ascending=False).name)

    #         # ax.set(xlabel='Confidence %', ylabel='Breed')

    #         # st.pyplot(fig)

main()