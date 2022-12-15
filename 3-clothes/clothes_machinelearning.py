import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

"""
Clothes recognition neural network
Authors: Kamil Rominski, Artur Jankowski
Required components:
- Tensorflow
- matplotlib
- sklearn
- keras

Clothes dataset is required - can be manually downloaded from web, but int it is handled by keras
"""

"""
Load Data
"""

clothes = tf.keras.datasets.fashion_mnist

(train_images, train_labes), (test_images, test_labels) = clothes.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
"""
Data scaling
"""
train_images = train_images / 255.0
test_images = test_images / 255.0

"""
Model preparation
"""
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(10),
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
"""
Fitting model
"""
model.fit(train_images, train_labes, epochs=10)

"""
Accuracy
"""
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print("\nAccuracy: ", test_acc)

"""
Predictions
"""

prob = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = prob.predict(test_images)


"""
Visualisation
"""


y_probs = model.predict(test_images)
y_preds = y_probs.argmax(axis=1)
cm=confusion_matrix(y_preds,test_labels)
disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=class_names)
fig, ax = plt.subplots(figsize=(10,10))
disp.plot(ax=ax)
plt.show()
