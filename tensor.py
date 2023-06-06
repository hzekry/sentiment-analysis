import joblib
import tensorflow as tf

# Load the scikit-learn model from the .pkl file
best_model = joblib.load('best_model.pkl')

# Convert the scikit-learn model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model to a file
with open('best_model.tflite', 'wb') as f:
    f.write(tflite_model)
