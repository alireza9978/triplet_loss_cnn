import os
import random
import time

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers

log_dir = "logs/"
BATCH_SIZE = 32
BUFFER_SIZE = 1000
images_shape = [28, 28, 1]
embedding_size = 64
class_count = 10
EPOCHS = 20
margin = 2


def load_data_set():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = np.concatenate([x_train, x_test])
    y_train = np.concatenate([y_train, y_test])
    x_train = x_train / 255.0
    x_train = x_train[..., tf.newaxis]

    classes = []
    for i in range(class_count):
        classes.append([])
    for i in range(len(x_train)):
        x = x_train[i]
        y = y_train[i]
        classes[y].append(x)

    for i in range(class_count):
        classes[i] = classes[i][0:500]

    x_train = []
    y_train = []
    for i in range(class_count):
        class_len = len(classes[i])
        k = 0
        for j in range(class_len):
            k = k + 1
            if k == class_count:
                k = 0
            if k == i:
                k = k + 1
                if k == class_count:
                    k = 0
            temp = random.sample(classes[i], 2)
            x_train.append([temp[0], temp[1], random.choice(classes[k])])
            y_train.append(i)

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2)

    return tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(BUFFER_SIZE).batch(
        BATCH_SIZE), tf.data.Dataset.from_tensor_slices(
        (x_test, y_test)).batch(BATCH_SIZE)


def make_model():
    input_layer = layers.Input(shape=images_shape)
    output_layer = layers.Conv2D(32, (3, 3), padding='same')(input_layer)
    output_layer = layers.BatchNormalization()(output_layer)
    output_layer = layers.ReLU()(output_layer)
    output_layer = layers.MaxPooling2D()(output_layer)
    output_layer = layers.Flatten()(output_layer)
    output_layer = layers.Dropout(0.3)(output_layer)
    output_layer = layers.Dense(256)(output_layer)
    output_layer = layers.Dropout(0.3)(output_layer)
    output_layer = layers.ReLU()(output_layer)
    output_layer = layers.Dense(100)(output_layer)
    output_layer = layers.Dropout(0.3)(output_layer)
    output_layer = layers.ReLU()(output_layer)
    output_layer = layers.Dense(embedding_size)(output_layer)
    classified_layer = layers.ReLU()(output_layer)
    classified_layer = layers.Dense(class_count)(classified_layer)
    classified_layer = layers.Softmax()(classified_layer)
    return tf.keras.Model(inputs=input_layer, outputs=[output_layer, classified_layer])


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
model_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
model = make_model()
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=model_optimizer, model=model)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


@tf.function
def test_step(x, y):
    x = tf.reshape(x, (3, -1, 28, 28, 1))
    y = tf.reshape(y, (-1, 1))
    input_image_anchor = x[0]
    input_image_positive = x[1]
    input_image_negative = x[2]

    model_output_anchor, model_output_anchor_class = model(input_image_anchor, training=False)
    model_output_positive, model_output_positive_class = model(input_image_positive, training=False)
    model_output_negative, model_output_negative_class = model(input_image_negative, training=False)

    losses = [triplet_loss(model_output_anchor, model_output_positive, model_output_negative),
              classification_loss(y, model_output_anchor_class),
              classification_loss(y, model_output_positive_class)]
    test_loss(losses)
    test_accuracy(y, model_output_anchor_class)
    test_accuracy(y, model_output_positive_class)


@tf.function
def train_step(x, y):
    x = tf.reshape(x, (3, -1, 28, 28, 1))
    y = tf.reshape(y, (-1, 1))
    input_image_anchor = x[0]
    input_image_positive = x[1]
    input_image_negative = x[2]
    with tf.GradientTape() as tape:
        model_output_anchor, model_output_anchor_class = model(input_image_anchor, training=True)
        model_output_positive, model_output_positive_class = model(input_image_positive, training=True)
        model_output_negative, model_output_negative_class = model(input_image_negative, training=True)

        losses = [triplet_loss(model_output_anchor, model_output_positive, model_output_negative),
                  classification_loss(y, model_output_anchor_class),
                  classification_loss(y, model_output_positive_class)]

    gradients_of_model = tape.gradient(losses, model.trainable_variables)

    model_optimizer.apply_gradients(zip(gradients_of_model, model.trainable_variables))
    train_loss(losses)
    train_accuracy(y, model_output_anchor_class)
    train_accuracy(y, model_output_positive)


def train(train_data_set, test_data_set, epochs=EPOCHS):
    for epoch in range(epochs):
        start = time.time()

        for x, y in train_data_set:
            train_step(x, y)
        #
        # for x, y in test_data_set:
        #     test_step(x, y)

        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}, time: {}'
        print(template.format(epoch + 1, train_loss.result(),
                              train_accuracy.result() * 100,
                              test_loss.result(),
                              test_accuracy.result() * 100,
                              time.time() - start))


train(train_data_gen, test_data_gen, EPOCHS)
