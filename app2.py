import streamlit as st
import os
# from segmentation.instance.instance_segmentation import do_instance
from semantic_segmentation import do_semantic
import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt

PAGE_TITLE = "Image Segmentation with Streamlit"


def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)


def file_selector_ui():
    # Select a file
    if st.checkbox('Select a file in current directory'):
        folder_path = '.'
        if st.checkbox('Change directory'):
            folder_path = st.text_input('Enter folder path', '.')
        filename = file_selector(folder_path=folder_path)
        st.write('You selected `%s`' % filename)
    else:
        return "."

    return filename


def main():
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")
    # gpu_memory_init()
    st.title(PAGE_TITLE)
    st.subheader("Get the file path")
    image_path = file_selector_ui()
    image_path = os.path.abspath(image_path)

    if os.path.isfile(image_path) is True:
        file_name = os.path.basename(image_path)
        _, file_extension = os.path.splitext(image_path)
        if file_extension == ".png":
            st.button('Instance Segmentation')
            start_time_2 = time.time()
            semantic_image, img = do_semantic(image_path)
            # st.write(np.unique(semantic_image))#.shape)
            # st.write(type(semantic_image))
            end_time_2 = time.time() - start_time_2
            st.write("**Semantic Segmentation**")
            plt.imshow(img)
            plt.imshow(semantic_image, alpha=0.5)
            plt.show()
            st.pyplot(plt)
            st.write(end_time_2)


@st.cache
def gpu_memory_init():
    # tf.debugging.set_log_device_placement(True)
    # Below code is for "failed to create cublas handle: CUBLAS_STATUS_ALLOC_FAILED"
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


if __name__ == "__main__":
    main()
