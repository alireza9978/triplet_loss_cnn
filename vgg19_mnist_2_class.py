import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow_addons as tfa
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

log_dir = "logs/"
BATCH_SIZE = 16
BUFFER_SIZE = 1000
images_shape = [32, 32, 3]
images_size = (32, 32)
input_size = (32, 32, 3)
embedding_size = 64
class_count = 2
EPOCHS = 50

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x = np.concatenate([x_train, x_test])
y = np.concatenate([y_train, y_test])

ones = x[np.where(y == 1)]
sevens = x[np.where(y == 7)]
ones = ones[np.random.choice(np.arange(ones.shape[0]), 100)]
sevens = sevens[np.random.choice(np.arange(sevens.shape[0]), 150)]

x = np.concatenate([ones, sevens])
y = np.concatenate([np.zeros(ones.shape[0]), np.ones(sevens.shape[0])])

new_x = []
for image in x:
    resized_image = resize(image, images_size, anti_aliasing=True)
    new_x.append(resized_image)

x = np.array(new_x)
x = np.divide(x, x.max())
x = np.multiply(x, 255)
print(x.max())
print(x.min())
x = np.repeat(x[..., np.newaxis], 3, -1)
y_categorical = to_categorical(y, class_count)

x_train, x_test, y_train_categorical, y_test_categorical, y_train, y_test = train_test_split(x, y_categorical, y,
                                                                                             test_size=0.2)

train_data_gen, test_data_gen = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(BUFFER_SIZE).batch(
    BATCH_SIZE), tf.data.Dataset.from_tensor_slices(
    (x_test, y_test)).batch(BATCH_SIZE)

base_model = tf.keras.applications.VGG19(input_shape=images_shape,
                                         include_top=False,
                                         weights='imagenet')

base_model.trainable = False

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
preprocess_input = tf.keras.applications.vgg19.preprocess_input

inputs = tf.keras.Input(shape=input_size)
model = preprocess_input(inputs)
model = base_model(model)
model = global_average_layer(model)
model = layers.Dense(256, activation='relu')(model)
model = layers.Dense(64, activation='relu', name='feature')(model)
outputs = layers.Dense(1, name='classification')(model)
model = tf.keras.Model(inputs, [outputs, model])

base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss={'classification': tf.keras.losses.BinaryCrossentropy(from_logits=True),
                    'feature': tfa.losses.TripletSemiHardLoss()},
              metrics=['accuracy'])

history = model.fit(train_data_gen,
                    epochs=10,
                    validation_data=test_data_gen)

acc = history.history['classification_accuracy']
val_acc = history.history['val_classification_accuracy']

loss = history.history['classification_loss']
val_loss = history.history['val_classification_loss']
triplet_loss = history.history['feature_loss']
triplet_val_loss = history.history['val_feature_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()), 1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.plot(triplet_loss, label='Training Triplet Loss')
plt.plot(triplet_val_loss, label='Validation Triplet Loss')
plt.legend(loc='lower left')
plt.ylabel('Loss')
plt.ylim([0, 1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()
