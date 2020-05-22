import glob, os
import re

# TensorFlow and Keras for Machine Learning
import tensorflow as tf
from tensorflow import keras

# Data Libraries
import numpy as np
import matplotlib.pyplot as plt

# Image Processing
import PIL
from PIL import Image


# Use Pillow to convert an input jpeg to a 8 bit grey scale image array for processing.
def jpeg_to_8_bit_greyscale(path, maxsize):
	img = Image.open(path).convert('L')   # convert image to 8-bit grayscale
	# Make aspect ratio as 1:1, by applying image crop.
	WIDTH, HEIGHT = img.size
	if WIDTH != HEIGHT:
		m_min_d = min(WIDTH, HEIGHT)
		img = img.crop((0, 0, m_min_d, m_min_d))
	# Scale the image to the requested maxsize by Anti-alias sampling.
	img.thumbnail(maxsize, PIL.Image.ANTIALIAS)
	return np.asarray(img)

class_names = ['Chihuahua', 'Muffin'] # Class types used to identify images

# Identifying our Training Set
def load_image_dataset(path_dir, maxsize):
	images = []
	labels = []
	os.chdir(path_dir)
	for file in glob.glob("*.jpg"):
		img = jpeg_to_8_bit_greyscale(file, maxsize)
		if re.match('chihuahua.*', file):
			images.append(img)
			labels.append(0)
		elif re.match('muffin.*', file):
			images.append(img)
			labels.append(1)
	return (np.asarray(images), np.asarray(labels))

# Handling our Testing Set
def load_test_set(path_dir, maxsize):
	test_images = []
	os.chdir(path_dir)
	for file in glob.glob("*.jpg"):
		img = jpeg_to_8_bit_greyscale(file, maxsize)
		test_images.append(img)
	return (np.asarray(test_images))

maxsize = 100, 100 # Uniform Aspect Ratio for all images

# Splitting data into a training and testing set
(train_images, train_labels) = load_image_dataset(
	path_dir='/home/jorge/Desktop/code/python/Machine Learning/Final Project/chihuahua-muffin',
	maxsize=maxsize)

(test_images, test_labels) = load_image_dataset(
	path_dir='/home/jorge/Desktop/code/python/Machine Learning/Final Project/chihuahua-muffin/test_set',
	maxsize=maxsize)

# Seeing the shape and characteristics of our data
print(train_images.shape)

print(len(train_labels))

print(train_labels)

print(test_images.shape)
print(test_labels)

# Scaling the images to values between 0 and 1
train_images = train_images / 255.0

test_images = test_images / 255.0


# Using matplotlib display images.
def display_images(images, labels):
	plt.figure(figsize=(10,10))
	grid_size = min(25, len(images))
	for i in range(grid_size):
		plt.subplot(5, 5, i+1)
		plt.xticks([])
		plt.yticks([])
		plt.grid(False)
		plt.imshow(images[i], cmap=plt.cm.binary)
		plt.xlabel(class_names[labels[i]])


display_images(train_images, train_labels)
plt.show()


# Building Model

# Setting up the layers
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(100, 100)),
        keras.layers.Dense(128, activation=tf.nn.sigmoid),
        keras.layers.Dense(16, activation=tf.nn.sigmoid),
    keras.layers.Dense(2, activation=tf.nn.softmax)
])

sgd = keras.optimizers.SGD(lr=0.01, decay=1e-5, momentum=0.7, nesterov=True)

model.compile(optimizer=sgd,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# Training
model.fit(train_images, train_labels, epochs=100)

# Evaluation
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

# Predictions
predictions = model.predict(test_images)
print(predictions)

# Display Results
display_images(test_images, np.argmax(predictions, axis = 1))
plt.show()
