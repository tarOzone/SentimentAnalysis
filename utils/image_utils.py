import os
import cv2
import numpy as np


def impath_to_np(image_path, rgb=True):
    # Assertion step
    assert type(image_path) == str, "img_path must be string (str)."
    assert os.path.exists(image_path), "img_path must exists."

    img = cv2.imread(image_path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if rgb else img


def bytes_to_np(img_bytes, rgb=True):
    # Assertion step
    assert type(img_bytes) == bytes, "img_bytes must be bytes."

    bytes_buffer = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(bytes_buffer, -1)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if rgb else img


def normalize_img(img):
    assert type(img) == np.ndarray, "img_bytes must be numpy array."
    return img / 255.0


def preprocess_img(model, img_path, resize_shape):
    img = impath_to_np(img_path, rgb=True)
    img = cv2.resize(normalize_img(img), resize_shape)
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)[0]
    return pred