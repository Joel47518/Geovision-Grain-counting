import streamlit as st
from streamlit_option_menu import option_menu
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import io
import numpy as np
import tensorflow_hub as hub
from skimage.util import img_as_ubyte
import pyclesperanto_prototype as cle


def load_model():
    model = tf.keras.models.load_model('EfficientNetB0.hdf5', custom_objects={'KerasLayer':hub.KerasLayer})
    return model


def prep_image(filename, img_shape=224):
    """
  Reads an image from filename, turns it into a tensor and reshape it
  to (img_shape, img_shape, colour_channels)
  """

    # # Read in the image
    # img = tf.io.read_file(filename)
    # Decode the read file into a tensor
    img = tf.convert_to_tensor(filename, dtype=tf.float32)
    # img = tf.image.decode_image(filename)
    # Resize the image
    img = tf.image.resize(img, size=[img_shape, img_shape])
    # Rescale the image (get all values between o and 1)
    img = img / 255.
    return img


def prediction(model, img, class_names):
    """
  Imports an image located at filename, makes a prediction with model.
  """

    # Make a prediction
    pred = model.predict(tf.expand_dims(img, axis=0))

    # Add in logic for multi-class
    if len(pred[0]) > 1:
        pred_class = class_names[tf.argmax(pred[0])]
    else:
        pred_class = class_names[int(tf.round(pred[0]))]

    return pred_class


st.title('GEO-VISION')

selected = option_menu(
    menu_title=None,
    options=["Rock Detection", "Grain Counting"],
    orientation="horizontal",
)

if selected == "Rock Detection":
    uploaded_file = st.sidebar.file_uploader('Upload Image')

    st.sidebar.divider()

    pred_button = st.sidebar.button("Predict", type='primary')

    class_names = ['Amphibolite', 'Andesite', 'Basalt', 'Breccia', 'Coal', 'Conglomerate', 'Gabbro', 'Gneiss',
                   'Limestone', 'Quartz_diorite', 'Quartzite', 'Sandstone', 'Shale', 'schist']

    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(bytes_data))
        img_array = np.array(image)

        # Display the uploaded image
        st.image(uploaded_file, caption='Uploaded Image')

        # Preprocess image
        img_preprocessed = prep_image(img_array)

        # Preprocess and make predictions
        if pred_button:
            predicted_class = prediction(model=load_model(), img=img_preprocessed, class_names=class_names)

            # Display the prediction
            # display_pred = st.markdown(f"<p style='font-size:24px'><strong>Prediction:</strong> {predicted_class}</p>", unsafe_allow_html=True)
            string = "Prediction: " + predicted_class
            st.success(string)
    else:
        st.warning("Please upload an image using the sidebar.")

if selected == "Grain Counting":
    uploaded_file = st.sidebar.file_uploader('Upload Image')
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(bytes_data))
        img_array = np.array(image)
        st.image(image=image)
    count_grain_button = st.sidebar.button("count grains", type='primary')
    if count_grain_button:
        from skimage import io
        input_image_original = img_array
        input_image = np.invert(input_image_original)
        binary = cle.binary_not(cle.threshold_otsu(input_image))
        labels = cle.voronoi_labeling(binary)
        num_objects = cle.maximum_of_all_pixels(labels)
        st.write("Total objects detected are:", num_objects) # make results big
    else:
        st.warning("Please upload an image using the sidebar.")