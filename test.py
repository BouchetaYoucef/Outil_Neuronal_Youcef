import pandas as pd
import tensorflow
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import load_model 
import cv2
import numpy as np
import os

# from svgpathtools import parse_path
# import SessionState
MODEL_DIR = os.path.join(os.path.dirname('C:/Users/simplonDesktop/Notebooks/Dossier_ALISON/Outil_Visualisation_Réseau_Neuronal'), 'model_weight.h5')

# 1)
# # Specify canvas parameters in application
# stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
# stroke_color = st.sidebar.color_picker("Stroke color hex: ")
# bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
# bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
# drawing_mode = st.sidebar.selectbox(
#     "Drawing tool:", ("freedraw", "line", "rect", "circle", "transform")
# )
# realtime_update = st.sidebar.checkbox("Update in realtime", True)
# model = load_model('my_model2.h5')
# # Create a canvas component
# canvas_result = st_canvas(
#     fill_color="rgba(255, 255, 255, 0.3)",  # Fixed fill color with some opacity
#     stroke_width=stroke_width,
#     stroke_color=stroke_color,
#     background_color=bg_color,
#     background_image=Image.open(bg_image) if bg_image else None,
#     update_streamlit=realtime_update,
#     height=150,
#     width=150,
#     drawing_mode=drawing_mode,
#     key="canvas",
# )

# # Do something interesting with the image data and paths
# if canvas_result.image_data is not None:
#     img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
#     rescaled = cv2.resize(img, (150, 150), interpolation=cv2.INTER_NEAREST)
# if canvas_result.json_data is not None:
#     objects = pd.json_normalize(canvas_result.json_data["objects"]) # need to convert obj to str because PyArrow
#     for col in objects.select_dtypes(include=['object']).columns:
#         objects[col] = objects[col].astype("str")
#     st.dataframe(objects)
# if st.button('Predict'):
#     test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     val = model.predict(test_x.reshape(1, 28, 28,1))
#     st.write(f'result: {np.argmax(val[0])}')
#     # st.bar_chart(val[0])

# 2
#Specify canvas parameters in application


stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
stroke_color = st.sidebar.color_picker("Stroke color hex: ")
bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("freedraw", "line", "rect", "circle", "transform")
)
realtime_update = st.sidebar.checkbox("Update in realtime", True)
model = load_model('model_weight.h5')
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
    key="canvas",
)

# Do something interesting with the image data and paths
if canvas_result.image_data is not None:
    st.image(canvas_result.image_data)
    img = cv2.resize(canvas_result.image_data.astype("uint8"), (28,28))

    rescaled = cv2.resize(img, (150, 150), interpolation=cv2.INTER_NEAREST)
    st.write('Model Input')
    st.image(rescaled)
if canvas_result.json_data is not None:
    objects = pd.json_normalize(canvas_result.json_data["objects"]) # need to convert obj to str because PyArrow
    for col in objects.select_dtypes(include=['object']).columns:
        objects[col] = objects[col].astype("str")
    st.dataframe(objects)
if st.button('Predict'):
    test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    val = model.predict(test_x.reshape(1, 28, 28,1))
    st.write(f'result: {np.argmax(val[0])}')
    # st.bar_chart(val[0])


# from pathlib import Path
# import pandas as pd
# import streamlit as st

# @st.cache
# def load_data():
#    bikes_data_path = Path() / 'data/bike_sharing_demand_train.csv'
#    data = pd.read_csv(bikes_data_path)
#    return data

# df = load_data()
# st.write(df)


# @st.cache
# def load_metadata():
#     DATA_URL = "https://streamlit-self-driving.s3-us-west-2.amazonaws.com/labels.csv.gz"
#     return pd.read_csv(DATA_URL, nrows=1000)

# @st.cache
# def create_summary(metadata, summary_type):
#     one_hot_encoded = pd.get_dummies(metadata[["frame", "label"]], columns=["label"])
#     return getattr(one_hot_encoded.groupby(["frame"]), summary_type)()

# # Piping one st.cache function into another forms a computation DAG.
# summary_type = st.selectbox("Type of summary:", ["sum", "any"])
# metadata = load_metadata()
# summary = create_summary(metadata, summary_type)
# st.write('## Metadata', metadata, '## Summary', summary)