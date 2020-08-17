import os
import time

import numpy as np
import scipy.io
import tensorflow as tf
import tensorflow_addons as tfa
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

log_dir = "logs/"
BATCH_SIZE = 64
BUFFER_SIZE = 1000
images_shape = [256, 256, 1]
images_size = (256, 256)
embedding_size = 64
class_count = 2
EPOCHS = 50
margin = 0.5
abnormal_path = "/home/alireza/projects/python/triplet loss/dataset/ab.mat"
normal_path = "/home/alireza/projects/python/triplet loss/dataset/norm.mat"


def load_data_set():
    abnormal_data = np.array(scipy.io.loadmat(abnormal_path)['arr'], np.uint16)
    normal_data = np.array(scipy.io.loadmat(normal_path)['arr'], np.uint16)

    y = np.concatenate([np.zeros(normal_data.shape[0]), np.ones(abnormal_data.shape[0])])
    x = np.concatenate([normal_data, abnormal_data])
    x = x.astype("float32") / x.max()
    new_x = []
    for image in x:
        trim_image = image[~np.all(image == 0, axis=1)]
        resized_image = resize(trim_image, images_size, anti_aliasing=True)
        new_x.append(resized_image)

    x = np.array(new_x)
    x = np.expand_dims(x, axis=-1)
    y_categorical = to_categorical(y, class_count)

    x_train, x_test, y_train_categorical, y_test_categorical, y_train, y_test = train_test_split(x, y_categorical, y,
                                                                                                 test_size=0.2)

    return tf.data.Dataset.from_tensor_slices((x_train, y_train_categorical, y_train)).shuffle(BUFFER_SIZE).batch(
        BATCH_SIZE), tf.data.Dataset.from_tensor_slices(
        (x_test, y_test_categorical, y_test)).batch(BATCH_SIZE)


def make_model():
    input_layer = layers.Input(shape=images_shape)
    output_layer = layers.Conv2D(64, (2, 2), padding='same')(input_layer)
    output_layer = layers.BatchNormalization()(output_layer)
    output_layer = layers.ReLU()(output_layer)
    output_layer = layers.MaxPooling2D()(output_layer)
    output_layer = layers.Dropout(0.3)(output_layer)
    output_layer = layers.Conv2D(32, (2, 2), padding='same')(output_layer)
    output_layer = layers.MaxPooling2D()(output_layer)
    output_layer = layers.ReLU()(output_layer)
    output_layer = layers.Dropout(0.3)(output_layer)
    output_layer = layers.Flatten()(output_layer)
    output_layer = layers.ReLU()(output_layer)
    output_layer = layers.Dropout(0.3)(output_layer)
    output_layer = layers.Dense(1024)(output_layer)
    output_layer = layers.Dropout(0.3)(output_layer)
    output_layer = layers.ReLU()(output_layer)
    output_layer = layers.Dense(100)(output_layer)
    output_layer = layers.Dropout(0.3)(output_layer)
    output_layer = layers.ReLU()(output_layer)
    embedding_layer = layers.Dense(embedding_size)(output_layer)
    output_layer = layers.Dense(class_count)(embedding_layer)
    output_layer = layers.Softmax()(output_layer)
    return tf.keras.Model(inputs=input_layer, outputs=[embedding_layer, output_layer])


def print_models():
    model = make_model()
    tf.keras.utils.plot_model(
        model, to_file='model.png', show_shapes=True, show_layer_names=True,
        rankdir='TB', expand_nested=True, dpi=96
    )


def triplet_loss(y, model_output):
    return triplet_loss_object(y, model_output)


def classification_loss(y, output):
    return loss_object(y, output)


train_data_gen, test_data_gen = load_data_set()
model_optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.CategoricalCrossentropy()
triplet_loss_object = tfa.losses.TripletSemiHardLoss()
model = make_model()
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=model_optimizer, model=model)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')


@tf.function
def test_step(x, y, z):
    feature_output_anchor, model_output_anchor = model(x, training=False)

    loss = [classification_loss(y, model_output_anchor),
            triplet_loss(z, feature_output_anchor)]
    test_loss(loss)
    test_accuracy(y, model_output_anchor)


@tf.function
def train_step(x, y, z):
    with tf.GradientTape() as tape:
        feature_output_anchor, model_output_anchor = model(x, training=True)

        loss = [triplet_loss(z, feature_output_anchor),
                classification_loss(y, model_output_anchor)]

    gradients_of_model = tape.gradient(loss, model.trainable_variables)

    model_optimizer.apply_gradients(zip(gradients_of_model, model.trainable_variables))
    train_loss(loss)
    train_accuracy(y, model_output_anchor)


def train(train_data_set, test_data_set, epochs=EPOCHS):
    for epoch in range(epochs):
        start = time.time()

        for x, y, z in train_data_set:
            train_step(x, y, z)

        for x, y, z in test_data_set:
            test_step(x, y, z)

        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}, time: {}'
        print(template.format(epoch + 1, train_loss.result(),
                              train_accuracy.result() * 100,
                              test_loss.result(),
                              test_accuracy.result() * 100,
                              time.time() - start))


train(train_data_gen, test_data_gen, EPOCHS)
