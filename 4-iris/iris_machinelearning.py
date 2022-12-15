import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

"""
Iris recognition neural network
Authors: Kamil Rominski, Artur Jankowski
Required components:
- Tensorflow
- Pandas
- Sklearn
Iris dataset contains septal and petal width and length
Class options:
0 - Iris-setosa
1 - Iris-versicolor
2 - Iris-virginica

Dataset used can be found here: https://archive.ics.uci.edu/ml/datasets/Iris
"""


"""
Loading data
"""
iris = pd.read_csv('iris.csv', delimiter=',')
print(iris.head(5))

"""
Data scaling and converting to data frames
"""
Scaler = StandardScaler()
Scaler.fit(iris.drop('class', axis=1))
Scaled_features = Scaler.fit_transform(iris.drop('class', axis=1))
df_features = pd.DataFrame(Scaled_features, columns=iris.columns[:-1])
x = df_features
y = iris['class']

"""
Splitting data frames into training and testing datasets
"""
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

"""
Model preparation
"""
sep_l = tf.feature_column.numeric_column("sep_l")
sep_w = tf.feature_column.numeric_column("sep_w")
pet_l = tf.feature_column.numeric_column("pet_l")
pet_w = tf.feature_column.numeric_column("pet_w")


feat_cols = [sep_l, sep_w, pet_l, pet_w]
classifier = tf.estimator.DNNClassifier(hidden_units=[10, 20, 10], n_classes=4, feature_columns=feat_cols)
input_function = tf.compat.v1.estimator.inputs.pandas_input_fn(x=x_train, y=y_train, batch_size=20, shuffle=True)
classifier.train(input_fn=input_function, steps=300)

"""
Model eval
"""
prediction_function = tf.compat.v1.estimator.inputs.pandas_input_fn(x=x_test, batch_size=len(x_test), shuffle=True)
note_predictions = list(classifier.predict(input_fn=prediction_function))
final_predictions = []
for prediction in note_predictions:
    final_predictions.append(prediction['class_ids'][0])

print(confusion_matrix(y_test, final_predictions))
print(classification_report(y_test, final_predictions))
