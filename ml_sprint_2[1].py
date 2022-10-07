import streamlit as st
from PIL import Image
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pickle import load
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import random

lr_classifier = load(open(r'C:\Users\91836\Desktop\project_handwritten\models\log_model_1.pkl', 'rb'))


def load_image(image_file):
    inp_img = Image.open(image_file)
    resized_img = inp_img.resize((28, 28))
    x = np.asarray(resized_img)
    y=x[:,:,1]
    z=y.flatten()
    arr = np.array(z)
    arr = arr.reshape(1,-1)
    answer = lr_classifier.predict(arr)
    answer_1 = ''.join(answer)
    return answer_1


    

st.markdown('Select an image and view to confirm')
uploadFile = st.file_uploader("choose it")


inp_img = Image.open(uploadFile)
resized_img = inp_img.resize((28, 28))
x_1 = np.asarray(resized_img)
    

if uploadFile is not None :
    st.image(uploadFile,width=120, use_column_width=120)
    
show_res_1 = st.button('show resized image')
if show_res_1 == True :
    st.image(x_1,width=120, use_column_width=120)

abcd = st.button('PREDICT')
if abcd == True :
    st.button(load_image(uploadFile))
    
