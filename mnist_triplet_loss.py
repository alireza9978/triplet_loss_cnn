import os
import time

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

log_dir = "logs/"
BATCH_SIZE = 64
BUFFER_SIZE = 1000
images_shape = [28, 28, 1]
embedding_size = 32
class_count = 10
EPOCHS = 5
margin = 0.5


def load_data_set():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = np.concatenate([x_train, x_test])
    y_train = np.concatenate([y_train, y_test])

    x_train = np.expand_dims(x_train, axis=-1)
    x_train = x_train.astype("float32") / 255.0

    classes = []
    for i in range(class_count):
        classes.append([])
    for i in range(len(x_train)):
        x = x_train[i]
        y = y_train[i]
        classes[y].append(x)

    classes = np.array(classes)
    for i in range(class_count):
        temp_class = classes[i]
        idx = np.random.choice(np.arange(len(temp_class)), 50, replace=False)
        idx = np.array(idx)
        temp_class = np.array(temp_class)[idx, :]
        classes[i] = temp_class

    x_train = []
    y_train = []
    sample_count = 10
    for my_class_number, my_class in enumerate(classes):
        my_class = np.array(my_class)
        for anchor in my_class:
            for other_class_number, other_class in enumerate(classes):
                if other_class_number == my_class_number:
                    continue
                other_class = np.array(other_class)
                idx = np.random.choice(np.arange(len(other_class)), sample_count, replace=False)
                sample_from_other_class = other_class[idx]
                idx = np.random.choice(np.arange(len(my_class)), sample_count, replace=False)
                sample_from_my_class = my_class[idx]
                for i in range(sample_count):
                    x_train.append([anchor, sample_from_my_class[i], sample_from_other_class[i]])
                    y_train.append(my_class_number)

    y_train = to_categorical(y_train, 10)
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2)

    return tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(BUFFER_SIZE).batch(
        BATCH_SIZE), tf.data.Dataset.from_tensor_slices(
        (x_test, y_test)).batch(BATCH_SIZE)


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


def triplet_loss(model_output_anchor, model_output_positive, model_output_negative):
    d_pos = tf.reduce_sum(tf.square(model_output_anchor - model_output_positive), 1)
    d_neg = tf.reduce_sum(tf.square(model_output_anchor - model_output_negative), 1)

    loss = tf.maximum(0., margin + d_pos - d_neg)
    loss = tf.reduce_mean(loss)
    return loss


def classification_loss(y, output):
    return loss_object(y, output)


train_data_gen, test_data_gen = load_data_set()
model_optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.CategoricalCrossentropy()
model = make_model()
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=model_optimizer, model=model)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')


@tf.function
def test_step(x, y):
    x = tf.reshape(x, (3, -1, 28, 28, 1))
    input_image_anchor = x[0]
    input_image_positive = x[1]
    input_image_negative = x[2]
    feature_output_anchor, model_output_anchor = model(input_image_anchor, training=False)
    feature_output_positive, model_output_positive = model(input_image_positive, training=False)
    feature_output_negative, model_output_negative = model(input_image_negative, training=False)

    loss = [classification_loss(y, model_output_anchor),
            classification_loss(y, model_output_positive),
            triplet_loss(feature_output_anchor, feature_output_positive, feature_output_negative)]
    test_loss(loss)
    test_accuracy(y, model_output_anchor)


@tf.function
def train_step(x, y):
    x = tf.reshape(x, (3, -1, 28, 28, 1))
    input_image_anchor = x[0]
    input_image_positive = x[1]
    input_image_negative = x[2]
    with tf.GradientTape() as tape:
        feature_output_anchor, model_output_anchor = model(input_image_anchor, training=True)
        feature_output_positive, model_output_positive = model(input_image_positive, training=True)
        feature_output_negative, model_output_negative = model(input_image_negative, training=True)
        loss = [triplet_loss(feature_output_anchor, feature_output_positive, feature_output_negative),
                classification_loss(y, model_output_anchor),
                classification_loss(y, model_output_positive)]

    gradients_of_model = tape.gradient(loss, model.trainable_variables)

    model_optimizer.apply_gradients(zip(gradients_of_model, model.trainable_variables))
    train_loss(loss)
    train_accuracy(y, model_output_anchor)


def train(train_data_set, test_data_set, epochs=EPOCHS):
    for epoch in range(epochs):
        start = time.time()

        for x, y in train_data_set:
            train_step(x, y)

        for x, y in test_data_set:
            test_step(x, y)

        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}, time: {}'
        print(template.format(epoch + 1, train_loss.result(),
                              train_accuracy.result() * 100,
                              test_loss.result(),
                              test_accuracy.result() * 100,
                              time.time() - start))


train(train_data_gen, test_data_gen, EPOCHS)
