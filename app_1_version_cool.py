
import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
import matplotlib as plt

import streamlit as st

# Custom imports 
# st.title("Streamlit exemple")
df_test = pd.read_csv("test.csv")
MODEL_DIR = os.path.join("C:/Users/simplon/Desktop/Notebooks/Dossier_ALISON/Outil_Visualisation_Réseau_Neuronal"  , "model_weight.h5")
model = load_model("C:/Users/simplon/Desktop/Notebooks/Dossier_ALISON/Outil_Visualisation_Réseau_Neuronal/model_weight.h5")

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
  df_test = pd.read_csv(uploaded_file)
  st.write(df_test.head())

# Preprocess
if df_test is not None:
    df_test = df_test/255 
    df_test = np.array(df_test)
    df_test = df_test.reshape(df_test.shape[0], 28, 28, 1)
    prediction = model.predict(df_test)
    prediction = np.argmax(prediction, axis=1)

# Prédiction image
def test_prediction(index):
    # Barre
    number = st.slider("Pick a number", 0, 9)
    st.write('Number :', number)
    print('Predicted category :', prediction[index])
    img = df_test[index].reshape(28,28)
    st.image(img, width=140)
    plt.imshow(img, cmap='gray')
page = st.selectbox("Choose your page", ["Version 1", "Version 2"])


if page == "Version 1":
  print("tetst")
if page == "Version 2":
  print("whtf")
index = np.random.choice(df_test.shape[0])
test_prediction(index)
