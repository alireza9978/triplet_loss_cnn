import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from skimage.transform import resize

images_size = (224, 224)
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
x = np.divide(x, x.max())
# x = np.subtract(x, 1)

print(x.max())
print(x.min())

index = np.random.choice(np.arange(len(x)), 9, False)
x = x[index]
y = y[index]

fig = plt.figure()
i = 1
for img in x:
    ax1 = fig.add_subplot(3, 3, i)
    ax1.imshow(img, cmap='gray')
    label = y[i - 1]
    if label == 1:
        ax1.set_title("abnormal")
    else:
        ax1.set_title("normal")
    i += 1

plt.show()
