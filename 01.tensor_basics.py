import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
# physical_device = tf.config.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(physical_device[0], True)

# Initialization of Tensors
x = tf.constant(4, shape=(1,1), dtype=tf.float32)
x = tf.constant([[1, 2, 3], [4, 5, 6]])

x = tf.ones(shape=(3, 3))
x = tf.zeros(shape=(2, 3))
x = tf.eye(3) # I for the identity matrix (eye)
x = tf.random.normal(shape=(3,3), mean=0, stddev=1)
x = tf.random.uniform(shape=(1,3), minval=0, maxval=1)
x = tf.range(start=1, limit=10, delta=2)
x = tf.cast(x, dtype=tf.float32)
# tf.float (16, 32, 64) | tf.int (8, 16, 32, 64) | tf.bool

# Mathematical Operations
x = tf.constant([1, 2, 3])
y = tf.constant([9, 8, 7])

z = tf.add(x, y)
z = x + y

z = tf.subtract(x, y)
z = x - y

z = tf.divide(x, y)
z = x / y

z = tf.multiply(x, y)
z = x * y

z = tf.tensordot(x, y, axes=1)
z = tf.reduce_sum(x * y, axis= 0)

z = x ** 5

x = tf.random.normal(shape=(2,3))
y = tf.random.normal(shape=(3,4))

z = tf.matmul(x, y)
z = x @ y

# Indexing 
x = tf.constant([0, 1, 1, 2, 3, 1, 2, 3])
# print(x[:])
# print(x[1:])
# print(x[1:3])
# print(x[::2])
# print(x[::-1])

indices = tf.constant([0, 3])
x_ind = tf.gather(x, indices)


x = tf.constant([[1, 2],
				 [3, 4],
				 [5, 6]])
# print(x[0,:])
# print(x[0:2,:])

# Reshaping
x = tf.range(9)
print(x)

x = tf.reshape(x, shape=(3,3))
print(x)

x = tf.transpose(x, perm=[1, 0])
print(x)