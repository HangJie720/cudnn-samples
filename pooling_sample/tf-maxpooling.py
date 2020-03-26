import tensorflow as tf
import numpy as np
import keras
from keras import backend

input = np.random.randn(1 * 20 * 24 * 24)
input_rounded = np.around(input, 6)
print(input_rounded.shape)
a = []
b = []
for v in input_rounded:
    str_v = str(v)
    v_splited = str(v).split('.')
    if len(v_splited[-1])<6:
        for i in range(6-len(v_splited[-1])):
            str_v += '0'
        print(str_v)
    a.append(str_v + 'f')
print(len(a))
print(a)

with open('./input.txt', "w") as f:
    f.write(','.join(a))

input_tensor = tf.reshape(input_rounded, [1, 20, 24, 24])
pooling_tenosr = keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid', data_format='channels_first')
with backend.get_session() as sess:
    print("image:")
    image = sess.run(pooling_tenosr(input_tensor))
    print(image.shape)
image = np.reshape(image, 1 * 20 * 12 * 12)
print(image.shape)

for v in image:
    str_v = str(v)
    v_splited = str(v).split('.')
    if len(v_splited[-1])<6:
        for i in range(6-len(v_splited[-1])):
            str_v += '0'
        print(str_v)
    b.append(str_v + 'f')
print(len(b))
with open('./output.txt', "w") as f:
    f.write(','.join(b))
