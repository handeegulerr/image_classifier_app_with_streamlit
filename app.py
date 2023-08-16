import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import PIL
import shutil
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from pathlib import Path
import time
import warnings
warnings.filterwarnings("ignore")


# Veri yüklemeyi hazırlama
datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # Resimleri 0-1 aralığına ölçekleme
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    )


# Derin öğrenme modelini yükleyin
model = tf.keras.models.load_model('my_model_augmentation_mobileNetV2_4.h5')

# Streamlit uygulama başlığı
st.title(':blue[_Image Classifier App_]')
st.caption('**:red[This app classifies images into 4 categories: cups, nature,sewing, working_place]**')

# Resim yükleme bileşeni
src_dir = st.text_input("**source directory**", key="src_dir")
if os.path.isdir(src_dir):
    st.success("Folder selected!✔️")

dst_dir = st.text_input("**destination directory**",key="dst_dir")
if os.path.isdir(dst_dir):
    st.success("Folder selected!✔️")


# Resmi sınıflandırın
    if os.path.isdir(src_dir) and os.path.isdir(dst_dir):
        test_generator = datagen.flow_from_directory(
            src_dir,
            target_size=(224, 224),
            batch_size=len(os.listdir(src_dir + "/test")),
            class_mode='categorical',
            shuffle=False)

        classes = ["cups", "nature","sewing", "working_place"]
        file_names = test_generator.filepaths
        sample_batch = next(iter(test_generator))
        sample_images, sample_labels = sample_batch
        preds = model.predict(sample_images)

        for i in range(len(file_names)):       
            pred = np.argmax(preds[i])
            src_path = file_names[i]
            base_name = os.path.basename(src_path)
            dst_path = os.path.join(dst_dir, classes[pred], base_name)
            shutil.copy(src_path, dst_path)

        st.write("Classified images are copied to the destination directory")               
        




