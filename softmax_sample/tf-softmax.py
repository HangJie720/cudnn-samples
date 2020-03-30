import tensorflow as tf
import numpy as np
import keras
from keras import backend

input = np.random.randn(1, 10, 1, 1)
input_rounded = np.around(input, 8)
print(input_rounded.shape)
print(input_rounded)

# execute fcn operator.

input_tensor = tf.placeholder(dtype=tf.float32, shape=(1, 10, 1, 1), name='input_image')
output_tensor = tf.nn.softmax(input_tensor, axis=1)
with tf.Session() as sess:
    print("image:")
    image = sess.run(output_tensor, feed_dict={input_tensor: input_rounded})
    print(image)

a = []
b = []

# save input data.
input_rounded = np.reshape(input_rounded, 1*10*1*1)
for v in input_rounded:
    str_v = str(v)
    v_splited = str(v).split('.')
    if len(v_splited[-1]) < 8:
        for i in range(8 - len(v_splited[-1])):
            str_v += '0'
        # print(str_v)
    a.append(str_v + 'f')
print(len(a))

with open('./input.txt', "w") as f:
    f.write(','.join(a))

# save output data.
image = np.reshape(image, 1 * 10 * 1 * 1)
print(image.shape)
for v in image:
    str_v = str(v)
    v_splited = str(v).split('.')
    if len(v_splited[-1]) < 8:
        for i in range(8 - len(v_splited[-1])):
            str_v += '0'
        # print(str_v)
    b.append(str_v + 'f')
print(len(b))
with open('./output.txt', "w") as f:
    f.write(','.join(b))
