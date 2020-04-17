import tensorflow as tf
import numpy as np

input = np.random.randn(4 * 4 * 50, 1)
weight = np.random.randn(500, 4 * 4 * 50)
input_rounded = np.around(input, 8)
weight_rounded = np.around(weight, 8)
print(input_rounded.shape)
print(weight_rounded.shape)
input_ = np.reshape(input_rounded, 4 * 4 * 50 * 1)
weight_ = np.reshape(weight_rounded, 3 * 4 * 4 * 500)

# execute fcn operator.
input_tensor = tf.placeholder(dtype=tf.float32, shape=(4 * 4 * 5, 1), name='input_image')
weight_tensor = tf.placeholder(dtype=tf.float32, shape=(3, 4 * 4 * 5), name='weight')

output_tensor = tf.matmul(weight_tensor, input_tensor)
with tf.Session() as sess:
    print("image:")
    image = sess.run(output_tensor, feed_dict={input_tensor: input_rounded,
                                               weight_tensor: weight_rounded})
    print(image.shape)

a = []
b = []
w = []
b_ = []

# save input data.
for v in input_:
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

# save weight.
for v in weight_:
    str_v = str(v)
    v_splited = str(v).split('.')
    if len(v_splited[-1]) < 8:
        for i in range(8 - len(v_splited[-1])):
            str_v += '0'
        # print(str_v)
    w.append(str_v + 'f')
print(len(w))

with open('./weight.txt', "w") as f:
    f.write(','.join(w))

# save bias.
# for v in bias_:
#     str_v = str(v)
#     v_splited = str(v).split('.')
#     if len(v_splited[-1]) < 8:
#         for i in range(8 - len(v_splited[-1])):
#             str_v += '0'
#         # print(str_v)
#     b_.append(str_v + 'f')
# print(len(b_))

# with open('./bias.txt', "w") as f:
#     f.write(','.join(b_))

# save output data.
image = np.reshape(image, 3*1)
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
