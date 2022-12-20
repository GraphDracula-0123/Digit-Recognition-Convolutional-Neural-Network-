import tensorflow as tf
import pandas as pd
from keras.utils import to_categorical

new_model = tf.keras.models.load_model('saved_model/my_model')

# Check its architecture
new_model.summary()

# Evaluate the restored model
df = pd.read_csv("mnist_test.csv")
X = df.iloc[:, 1:].values.reshape(-1, 28, 28, 1)
test_images = X.astype("float32") / 255.0
test_labels = to_categorical(df.iloc[:, 0])

loss, acc = new_model.evaluate(test_images, test_labels, verbose=0)
print('Restored model accuracy: {:5.2f}%'.format(100 * acc))
print(new_model.predict(test_images).shape) 