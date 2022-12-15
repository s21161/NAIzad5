import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
"""
Wheat recognition neural network
Authors: Kamil Rominski, Artur Jankowski
Required components:
- Tensorflow
- matplotlib
- sklearn
- keras

Dataset used can be found here: https://archive.ics.uci.edu/ml/datasets/seeds
"""

"""
Loading data
"""
wheat = pd.read_csv('seeds_dataset.csv', delimiter='\t')
print(wheat.head(5))

"""
Data scaling and converting to data frames
"""
Scaler = StandardScaler()
Scaler.fit(wheat.drop('class', axis=1))
Scaled_features = Scaler.fit_transform(wheat.drop('class', axis=1))
df_features = pd.DataFrame(Scaled_features, columns=wheat.columns[:-1])
x = df_features
y = wheat['class']

"""
Splitting data frames into training and testing datasets
"""
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

"""
Model preparation
"""
area = tf.feature_column.numeric_column("area")
perimeter = tf.feature_column.numeric_column("perimeter")
compactness = tf.feature_column.numeric_column("compactness")
length = tf.feature_column.numeric_column("length")
width = tf.feature_column.numeric_column("width")
assymetry = tf.feature_column.numeric_column("assymetry")
lenght_groove = tf.feature_column.numeric_column("lenght_groove")

feat_cols = [area, perimeter, compactness, length, width, assymetry, lenght_groove]
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
