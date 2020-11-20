import numpy as np
import matplotlib.pyplot as plt
import glob

from skimage.transform import resize
from tensorflow.keras import datasets, layers, models

# Importing all the datasets from the cifar10 dataset in the variables which contains 50000 training images and 10000 test images
(image_train, label_train), (image_test, label_test) = datasets.cifar10.load_data()

# All the labels for every image in the dataset
label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog',
               'frog', 'horse', 'ship', 'truck']

# Scaling the data down between 0 and 1
image_test = image_test / 255
image_train = image_train / 255

# Reduce the amount of data to 20000 from 50000 for the training images/labels and 4000 from 10000 for the testing images/labels
image_train = image_train[:20000]
label_train = label_train[:20000]
image_test = image_test[:4000]
label_test = label_test[:4000]


# Loading the model
model = models.load_model('image_specification.model')

# Reading the image resize it and converting every pixel into a binary
image = plt.imread('./assets/cat.jpg')
resized = resize(image, (32,32,3))
plt.imshow(resized, cmap=plt.cm.binary)

# Passing the images in an array 
path = glob.glob('./assets/*.jpg')
images = []
for i in path:
    n = plt.imread(i)
    images.append(n)

# Printing all the images   
for i in range(10):
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(images[i], cmap=plt.cm.binary)
plt.show()

# Predict the label for the imported image, converting it to array and find the bigget value of it (the predicted label)
prediction = model.predict(np.array([resized]))
index = np.argmax(prediction)

#A classic sorting method to find the closest label for the selected image
indexList = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
for i in range(10):
    for j in range(10):
        if prediction[0][indexList[i]] > prediction[0][indexList[j]]:
            temp = indexList[i]
            indexList[i] = indexList[j]
            indexList[j] = temp


#Printing 10 images sorted from the closest image to least closest with the predicted label and the percentage of it
for i in range(10):
    print(label_names[indexList[i]],':', round(prediction[0][indexList[i]] * 100, 2))