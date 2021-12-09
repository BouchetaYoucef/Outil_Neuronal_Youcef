import streamlit as st

from numpy.core.fromnumeric import size
import streamlit as st
import pandas as pd
import numpy as np


import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2


from PIL import Image



import os
from tensorflow.keras.models import load_model
import matplotlib as plt



#from multiapp import MultiApp
#
#app = MultiApp()
#
#st.markdown("this is a multy app test")
#
#app.add_app("Version 1" , drawing_second_version)
#app.add_app("Version 2" , drawing_copy)
#app.run()

if st.selectbox("Choose your page", ["Version 1", "Version 2"]) == "Version 1" :    
    MODEL_DIR = os.path.join(os.path.dirname('C:/Users/edenl/Desktop/ia_coding/deep_learning/RNN , CNN/CNN'), 'model.h5')
    # Specify canvas parameters in application
    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
    stroke_color = st.sidebar.color_picker("Stroke color hex: ")
    bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
    bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
    drawing_mode = st.sidebar.selectbox(
        "Drawing tool:", ("freedraw", "line", "rect", "circle", "transform")
    )
    realtime_update = st.sidebar.checkbox("Update in realtime", True)
    model = load_model('model.h5')
    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        background_image=Image.open(bg_image) if bg_image else None,
        update_streamlit=realtime_update,
        height=150,
        drawing_mode=drawing_mode,
        key="canvas",)

        # Do something interesting with the image data and paths
    if canvas_result.image_data is not None:
        st.image(canvas_result.image_data)
        img = cv2.resize(canvas_result.image_data.astype("uint8"), (28,28))
        rescaled = cv2.resize(img, (150, 150), interpolation=cv2.INTER_NEAREST)
        st.write('Model Input')
        st.image(rescaled)
    #if canvas_result.json_data is not None:
    #    objects = pd.json_normalize(canvas_result.json_data["objects"]) # need to convert obj to str because PyArrow
    #    for col in objects.select_dtypes(include=['object']).columns:
    #        objects[col] = objects[col].astype("str")
    #    st.dataframe(objects)
        if st.button('Predict'):
            test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            val = model.predict(test_x.reshape(1, 28, 28,1))
            st.write(f'result: {np.argmax(val[0])}')
            st.bar_chart(val[0])           
else : 

    df_test = pd.read_csv("test.csv")
    MODEL_DIR = os.path.join("C:/Users/edenl/Desktop/ia_coding/deep_learning/RNN , CNN/CNN"  , "model.h5")
    model = load_model("C:/Users/edenl/Desktop/ia_coding/deep_learning/RNN , CNN/CNN/model.h5")

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
        #plt.imshow(img, cmap='gray')