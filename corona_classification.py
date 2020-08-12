import numpy as np
import scipy.io
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

log_dir = "logs/"
BATCH_SIZE = 64
BUFFER_SIZE = 1000
images_shape = [28, 28, 1]
embedding_size = 32
class_count = 10
EPOCHS = 5
margin = 0.5
abnormal_path = "/home/alireza/projects/python/triplet loss/dataset/ab.mat"
normal_path = "/home/alireza/projects/python/triplet loss/dataset/norm.mat"


def load_data_set():
    abnormal_data = scipy.io.loadmat(abnormal_path)['arr']
    normal_data = scipy.io.loadmat(normal_path)['arr']
    y = np.concatenate([np.zeros(normal_data.shape[0]), np.ones(abnormal_data.shape[0])])
    x = np.concatenate([normal_data, abnormal_data])

    x = np.expand_dims(x, axis=-1)
    x = x.astype("float32")
    x += x.min()
    x /= x.max()
    y_categorical = to_categorical(y, 2)

    print(x)
    print(x.shape)
    print(y)
    print(y.shape)
    print(y_categorical)
    print(y_categorical.shape)

    x_train, x_test, y_train_categorical, y_test_categorical, y_train, y_test = train_test_split(x, y_categorical, y,
                                                                                                 test_size=0.2)

    return tf.data.Dataset.from_tensor_slices((x_train, y_train_categorical, y_train)).shuffle(BUFFER_SIZE).batch(
        BATCH_SIZE), tf.data.Dataset.from_tensor_slices(
        (x_test, y_test_categorical, y_test)).batch(BATCH_SIZE)
