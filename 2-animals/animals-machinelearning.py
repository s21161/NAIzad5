import tensorflow as tf
from keras import layers, models
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
"""
Animal recognition neural network
Authors: Kamil Rominski, Artur Jankowski
Required components:
- Tensorflow
- matplotlib
- sklearn
- keras
- seaborn

Cifar10 dataset is required - can be manually downloaded from web, but int it is handled by keras
"""

"""
Load Data
"""
cifar10 = tf.keras.datasets.cifar10

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

"""
Data scaling
"""
train_images = train_images / 255.0
test_images = test_images / 255.0

"""
Model preparation
"""
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
"""
Fitting model
"""
model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))
"""
Accuracy
"""
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('Accuracy:', test_acc)

"""
Predictions
"""
y_pred = model.predict(test_images)
"""
Visualisation
"""
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(test_labels, axis=1)
confusion_mtx = tf.math.confusion_matrix(y_true, y_pred_classes)

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
plt.figure(figsize=(12, 9))
c = sns.heatmap(confusion_mtx, annot=True, fmt='g')
c.set(xticklabels=classes, yticklabels=classes)
plt.show()
