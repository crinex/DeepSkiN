import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from skimage.transform import resize
from skimage.io import imread
import os
import tensorflow as tf
from tensorflow.keras import backend as K

st.set_page_config(layout="wide")

smooth = 1.
def dice_coef(y_true, y_pred):
    y_true_f = tf.keras.layers.Flatten()(y_true)
    y_pred_f = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

class FixedDropout(tf.keras.layers.Dropout):
    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape
        symbolic_shape = K.shape(inputs)
        noise_shape = [symbolic_shape[axis] if shape is None else shape
        for axis, shape in enumerate(self.noise_shape)]
        return tuple(noise_shape)


# def enhance(img):
#     # reshape(1, 256, 256, 1)
#     #sub = (model.predict(img.reshape(1,256,256,3))).flatten()
#     img = img.reshape((1, 224, 224, 3)).astype(np.float32) / 255.
#     sub = (model.predict(img)).flatten()
#
#     for i in range(len(sub)):
#         if sub[i] > 0.5:
#             sub[i] = 1
#         else:
#             sub[i] = 0
#     return sub
#
# def applyMask(img):
#     sub = img.reshape((1, 224, 224, 3)).astype(np.float32) / 255.
#     #sub = np.array(img.reshape(256, 256), dtype=np.uint8)
#     mask = np.array(enhance(sub).reshape(224, 224), dtype=np.uint8)
#     sub2 = img.reshape(224, 224, 3)
#     #sub2 = np.array(img.reshape(256, 256, 3), dtype=np.uint8)
#     res = cv2.bitwise_and(sub2, sub2, mask = mask)
#
#     return res

def enhance(img):
    input_img = img.reshape((1, 224, 224, 3)).astype(np.float32) / 255.
    seg_img = (model.predict(input_img)).flatten()

    for i in range(len(seg_img)):
        if seg_img[i] > 0.5:
            seg_img[i] = 1
        else:
            seg_img[i] = 0
    return seg_img.reshape(224, 224)

def applyMask(x):
    img = x
    mask = np.array(enhance(img), dtype=np.uint8)
    res = cv2.bitwise_and(img, img, mask=mask)

    return res


def hair_remove(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(1, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _,threshold = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    final_image = cv2.inpaint(img, threshold, 1, cv2.INPAINT_TELEA)
    final_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
    return final_image

classes = [
    'Actinic Keratoses',
    'Basal Cell Carcinoma',
    'Benign keratosis-like lesions',
    'Dermatofibroma',
    'Melanocytic nevi',
    'Melanoma',
    'Squamous cell carcinoma',
    'Vascular lesions',
    'UNK'
]


def predict_img(x, model):
    x_data = (np.expand_dims(x, 0))
    pred = model.predict(x_data)

    return pred

##############
# Model Load #
##############
@st.cache
def load():
    return load_model('ResUnetpp_re.h5', custom_objects={'dice_coef':dice_coef, 'dice_loss':dice_loss})

def clf_load():
    return load_model('clf.h5', custom_objects={'FixedDropout':FixedDropout})
model = load()
clf = clf_load()
#st.set_page_config(layout="wide")


##############
# Side Bar   #
##############
with st.sidebar.header('Upload your Skin Image'):
    upload_file = st.sidebar.file_uploader('Choose your Skin Image', type=['jpg', 'jpeg', 'png'])


##############
# Page Title #
##############
st.write('# 🧠Skin Lesion Segmentation🧠')
st.write('This Website was created by Crinex. The code for the Website and Segmentation is in the Github. If you want to use this Code, please Fork and use it.🤩🤩')
st.write('📕 Github:https://github.com/crinex/Skin-Lesion-Segmentation-Streamlit 📕')


###############
# Main Screen #
###############
col1, col2, col3, col4, col5 = st.beta_columns(5)

if upload_file is not None:
    with col1:
        st.write('### Original Image')
        img = imread(upload_file)
        img = cv2.resize(img, (224, 224))
        img = hair_remove(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(img)
else:
    col1.write('### Original Image')
    img = imread('skin_img1.jpg')
    img = cv2.resize(img, (224, 224))
    img = hair_remove(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    col1.image(img)


col2.write('### Button')
clicked = col2.button('Segment!!')
clicked2 = col2.button('Predict Mask')

if clicked:
    col3.write('### Segmentation Image')
    mask_img = applyMask(img)
    col3.image(mask_img)

if clicked2:
    enhance_img = enhance(img).reshape(224, 224)
    col3.write('### Prediction Image')
    col3.image(enhance_img)

col4.write('### Button')
clicked3 = col4.button('Classify')

if clicked3:
    col5.write('### Classify Image!!')
    mask_img2 = applyMask(img)
    pred_img = predict_img(mask_img2, clf)
    pred_idx = np.argmax(pred_img)
    pred_num = 100*pred_img[0][pred_idx]
    pred_cls = classes[pred_idx]
    col5.write('## {:.0f}%: {:}'.format(pred_num, pred_cls))
