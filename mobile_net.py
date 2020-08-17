import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import tensorflow as tf
import tensorflow_addons as tfa
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

log_dir = "logs/"
BATCH_SIZE = 16
BUFFER_SIZE = 1000
images_shape = [224, 224, 3]
images_size = (224, 224)
embedding_size = 64
class_count = 2
EPOCHS = 50

abnormal_path = "/home/alireza/projects/python/triplet loss/dataset/ab.mat"
normal_path = "/home/alireza/projects/python/triplet loss/dataset/norm.mat"

abnormal_data = np.array(scipy.io.loadmat(abnormal_path)['arr'], np.uint16)
normal_data = np.array(scipy.io.loadmat(normal_path)['arr'], np.uint16)

y = np.concatenate([np.zeros(normal_data.shape[0]), np.ones(abnormal_data.shape[0])])
x = np.concatenate([normal_data, abnormal_data])
x = x.astype("float32")

new_x = []
for image in x:
    trim_image = image[~np.all(image == 0, axis=1)]
    resized_image = resize(trim_image, images_size, anti_aliasing=True)
    new_x.append(resized_image)

x = np.array(new_x)
x = np.divide(x, x.max() / 2)
x = np.subtract(x, 1)
print(x.max())
print(x.min())
x = np.repeat(x[..., np.newaxis], 3, -1)
y_categorical = to_categorical(y, class_count)

x_train, x_test, y_train_categorical, y_test_categorical, y_train, y_test = train_test_split(x, y_categorical, y,
                                                                                             test_size=0.2)

train_data_gen, test_data_gen = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(BUFFER_SIZE).batch(
    BATCH_SIZE), tf.data.Dataset.from_tensor_slices(
    (x_test, y_test)).batch(BATCH_SIZE)

base_model = tf.keras.applications.MobileNetV2(input_shape=images_shape,
                                               include_top=False,
                                               weights='imagenet')

base_model.trainable = False

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(1)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

inputs = tf.keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(512, activation='relu', use_bias=True)(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(256, activation='relu', use_bias=True)(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(128, activation='relu', use_bias=True)(x)
x = tf.keras.layers.Dropout(0.2)(x)
f_x = tf.keras.layers.Dense(64, activation='relu', use_bias=True, name="feature_dense")(x)
outputs = tf.keras.layers.Dense(1, name="classification_dense")(f_x)
model = tf.keras.Model(inputs, [outputs, f_x])

base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss={'classification_dense': tf.keras.losses.BinaryCrossentropy(from_logits=True),
                    'feature_dense': tfa.losses.TripletSemiHardLoss()},
              metrics=['accuracy'])

model.summary()
len(model.trainable_variables)

history = model.fit(train_data_gen,
                    epochs=10,
                    validation_data=test_data_gen)

acc = history.history['classification_dense_accuracy']
val_acc = history.history['val_classification_dense_accuracy']

loss = history.history['classification_dense_loss']
val_loss = history.history['val_classification_dense_loss']
triplet_loss = history.history['feature_dense_loss']
triplet_val_loss = history.history['val_feature_dense_loss']

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

base_model.trainable = True
# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(base_model.layers))

# Fine-tune from this layer onwards
fine_tune_at = 100

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate / 10),
              loss={'classification_dense': tf.keras.losses.BinaryCrossentropy(from_logits=True),
                    'feature_dense': tfa.losses.TripletSemiHardLoss()},
              metrics=['accuracy'])
len(model.trainable_variables)

history_fine = model.fit(train_data_gen,
                         epochs=20,
                         initial_epoch=history.epoch[-1],
                         validation_data=test_data_gen)

acc += history_fine.history['classification_dense_accuracy']
val_acc += history_fine.history['val_classification_dense_accuracy']

loss += history_fine.history['classification_dense_loss']
val_loss += history_fine.history['val_classification_dense_loss']
triplet_loss += history_fine.history['feature_dense_loss']
triplet_val_loss += history_fine.history['val_feature_dense_loss']

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