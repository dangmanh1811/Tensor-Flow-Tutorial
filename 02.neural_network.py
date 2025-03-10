import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28 * 28).astype("float32") / 255.
x_test = x_test.reshape(-1, 28 * 28).astype("float32") / 255.

# Sequantial API (Very convenient, not very flexible)
model = keras.Sequential(
	[
		keras.Input(shape=(28*28,)),
		layers.Dense(512, activation='relu'),
		layers.Dense(256, activation='relu'),
		layers.Dense(10)	
	]
)

model = keras.Sequential()
model.add(keras.Input(shape=(28*28,)))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(256, activation='relu', name='my_layer'))
model.add(layers.Dense(10))

# model = keras.Model(inputs=model.inputs,
# 					outputs=model.layers[-2].output)

# model = keras.Model(inputs=model.inputs,
# 					outputs=model.get_layer("my_layer").output)

# feature = model.predict(x_train)
# print(feature.shape)

model = keras.Model(inputs=model.inputs,
					outputs=[layer.output for layer in model.layers])

features = model(x_train)

for feature in features:
	print(feature.shape)
import sys
sys.exit()

# Functional API (A bit more flexible)
inputs = keras.Input(shape=(28*28,))
x = layers.Dense(516, activation='relu', name='first_layer')(inputs)
x = layers.Dense(256, activation='relu', name='second_layer')(x)
outputs = layers.Dense(10)(x)
model = keras.Model(inputs=inputs, outputs=outputs)

print(model.summary())

model.compile(
	loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	optimizer=keras.optimizers.Adam(learning_rate=0.001),
	metrics=["accuracy"],	
)

"""
verbose = 0 : Không hiển thị gì cả (chạy im lặng)
		= 1 : Hiển thị thanh tiến trình (progress bar)
		= 2 : Hiển thị thông tin từng epoch theo dạng văn bản, no progress bar
"""
model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=1)
model.evaluate(x_test, y_test, batch_size=32, verbose=1)

