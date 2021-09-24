
import tensorflow as tf
from keras.layers import Conv2D, AveragePooling2D, BatchNormalization, Activation
import PIL.Image as Image
import matplotlib.pyplot as plt
import time
import cv2
import numpy as np


# ===== Config ===== #
image1 = 'bubbly_0038.jpg'
# Texture54.png
layer_S = 3
Iter = 3000
layer_D = 9
Adam = 0
mean = 1
inner = 10
Gau = 0.000001
save_weights = 0
Scale = 1e12
im_dir = './Image/'
image_path1 = im_dir + image1


I = Image.open(image_path1).resize((256, 256))  # was 256,256


class model1():
    def __init__(self, num_Com_layer):
        self.num_Com_layer = num_Com_layer

        self.Wave_List = list()
        self.Wave_List.append(Conv2D(32, (3, 3), padding='same', name='convpool1'))
        self.Wave_List.append(AveragePooling2D((2, 2), strides=(2, 2), name='wavepool1'))

        self.Layer_List = list()

        self.Layer_List.append(Conv2D(32, (3, 3), padding='same', name='conv_wave_1'))
        # # self.Layer_List.append(Activation("relu",name = 'pool_ReLu1'))
        self.Layer_List.append(AveragePooling2D((2, 2), strides=(2, 2), name='block_wave1_pool1'))

        # self.Layer_List.append(Conv2D(32, (3, 3), padding='same', name='conv_wave_2'))
        # # self.Layer_List.append(Activation("relu",name = 'pool_ReLu2'))
        # self.Layer_List.append(AveragePooling2D((2, 2), strides=(2, 2), name='block_wave1_pool2'))

        self.Layer_List.append(Conv2D(128, (3, 3), padding='same', name='conv1'))
        self.Layer_List.append(BatchNormalization(name='batch_norm1'))
        self.Layer_List.append(Activation("relu", name='relu1'))

        self.Layer_List.append(Conv2D(128, (3, 3), padding='same', name='conv2'))
        self.Layer_List.append(BatchNormalization(name='batch_norm2'))
        self.Layer_List.append(Activation("relu", name='relu2'))

        self.Layer_List.append(AveragePooling2D((2, 2), strides=(2, 2), name='block1_pool'))

        self.Layer_List.append(Conv2D(128, (3, 3), padding='same', name='conv3'))
        self.Layer_List.append(BatchNormalization(name='batch_norm3'))
        self.Layer_List.append(Activation("relu", name='relu3'))

        self.Layer_List.append(Conv2D(128, (3, 3), padding='same', name='conv4'))
        self.Layer_List.append(BatchNormalization(name='batch_norm4'))
        self.Layer_List.append(Activation("relu", name='relu4'))

        self.Layer_List.append(AveragePooling2D((2, 2), strides=(2, 2), name='block2_pool'))

        self.Layer_List.append(Conv2D(128, (3, 3), padding='same', name='conv5'))
        self.Layer_List.append(BatchNormalization(name='batch_norm5'))
        self.Layer_List.append(Activation("relu", name='relu5'))

        self.Layer_List.append(Conv2D(128, (3, 3), padding='same', name='conv6'))
        self.Layer_List.append(BatchNormalization(name='batch_norm6'))
        self.Layer_List.append(Activation("relu", name='relu6'))

        self.Layer_List.append(AveragePooling2D((2, 2), strides=(2, 2), name='block3_pool'))

        self.High_res_com_list = list()
        self.High_res_com_list.append(Conv2D(32, (3, 3), padding='same', dilation_rate=(1, 1), name='high_res_conv1'))
        self.High_res_com_list.append(BatchNormalization(name='batch_high_res_comm_1'))
        self.High_res_com_list.append(Conv2D(32, (3, 3), padding='same', dilation_rate=(2, 2), name='high_res_conv2'))
        self.High_res_com_list.append(BatchNormalization(name='batch_high_res_comm_2'))

        self.Composite_List = list()
        self.scale = np.array([15, 35, 55, 75, 135])
        n = 32

        for i in range(num_Com_layer):
            self.Composite_List.append(
                Conv2D(n, 5, padding='valid', dilation_rate=int(self.scale[i] / 15), name='conv1' + str(i)))
            self.Composite_List.append(BatchNormalization(name='batch_Com' + str(i)))

    def run(self, x):
        # Return the var list
        self.hid_dict = dict()

        # for i in range(int(len(self.High_res_com_list)/2)):
        #     tmp = self.High_res_com_list[i * 2](x)
        #     tmp = self.High_res_com_list[i * 2  + 1](tmp)
        #     self.hid_dict['high_res_composite' + str(i+1)] = tf.clip_by_value(tmp, 0, 1)

        # for i in range(len(self.Wave_List)):
        #     x = self.Wave_List[i](x)
        #     self.hid_dict[self.Wave_List[i].name] = x

        for i in range(self.num_Com_layer):
            tmp = self.Composite_List[i * 2](x)
            tmp = self.Composite_List[i * 2 + 1](tmp)
            self.hid_dict['composite' + str(i + 1)] = tf.clip_by_value(tmp, 0, 1)

        for i in range(len(self.Layer_List)):
            x = self.Layer_List[i](x)
            if 'relu' in self.Layer_List[i].name:
                x = tf.clip_by_value(x, 0, 1)
            self.hid_dict[self.Layer_List[i].name] = x
        self.out = x

        self.var_list = list()

        # for i in range(len(self.Wave_List)):
        #     tmp = self.Wave_List[i].trainable_weights
        #     for j in range(len(tmp)):
        #         self.var_list.append(tmp[j])

        # for i in range(len(self.High_res_com_list)):
        #     tmp = self.High_res_com_list[i].trainable_weights
        #     for j in range(len(tmp)):
        #         self.var_list.append(tmp[j])

        for i in range(len(self.Layer_List)):
            tmp = self.Layer_List[i].trainable_weights
            for j in range(len(tmp)):
                self.var_list.append(tmp[j])

        for i in range(len(self.Composite_List)):
            tmp = self.Composite_List[i].trainable_weights
            for j in range(len(tmp)):
                self.var_list.append(tmp[j])

        return self.out, self.hid_dict, self.var_list, self.num_Com_layer

# ==== Functions ===== #

def gram_matrix(feature_maps):
    """Computes the Gram matrix for a set of feature maps."""
    batch_size, height, width, channels = tf.unstack(tf.shape(feature_maps))
    denominator = tf.cast(height * width, dtype=tf.float32)
    feature_maps = tf.reshape(feature_maps, tf.stack([batch_size, height * width, channels]))
    matrix = tf.matmul(feature_maps, feature_maps, adjoint_a=True)
    return matrix / denominator


def gram_loss(feature1, reference):
    F1 = gram_matrix(feature1)
    F2 = gram_matrix(reference)
    loss = tf.reduce_mean((F1 - F2) ** 2)
    return loss


def mean_loss(feature1, reference):
    m1 = tf.reduce_mean(feature1, axis=(1, 2))
    # print(m1.shape)
    m2 = tf.reduce_mean(reference, axis=(1, 2))
    loss = tf.reduce_mean(tf.square(m1 - m2))
    return loss

loaded_image_array1 = np.array(I, dtype=np.float) / 255

img_nrows, img_ncols, _ = loaded_image_array1.shape

AVE = np.average(loaded_image_array1, axis=(0, 1))
image_processed = loaded_image_array1 - AVE

ref = tf.expand_dims(tf.constant(image_processed, dtype=tf.float32), axis=0)

x = tf.Variable(np.random.randn(3, img_nrows, img_ncols, 3), dtype=tf.float32)


model = model1(layer_S)

# %%

out, h_dict, var_list, _ = model.run(x)
out_ref, h_dict_ref, _, _ = model.run(ref)

# %%

# layer_name = ['wavepool1']
layer_name = []
layer_name += ['block_wave1_pool1']
# layer_name += ['wavepool1']
layer_name_ = ['relu1', 'relu2', 'block1_pool', 'relu3', 'relu4', 'block2_pool', 'relu5', 'relu6', 'block3_pool',
               'relu7', 'relu8', 'block4_pool', 'relu9', 'relu10', 'block5_pool']
layer_name += layer_name_[0:layer_D]
# layer_name += ['high_res_composite1','high_res_composite2']
layer_c = ['composite1', 'composite2', 'composite3', 'composite4', 'composite5']
layer_name += layer_c[0: layer_S]


LOSS_LIST = []
for layer in layer_name:
    if mean == 0:
        tmp_loss = gram_loss(h_dict_ref[layer], h_dict[layer])
    else:
        tmp_loss = mean_loss(h_dict_ref[layer], h_dict[layer])
    LOSS_LIST.append(tmp_loss)
Loss = tf.add_n(LOSS_LIST)
Loss += tf.reduce_sum(x ** 2) * Gau
Loss *= Scale

# %%

if Adam == 0:
    op_I = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(Loss, var_list=x)
    op_w = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(-Loss, var_list=var_list)
else:
    op_I = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5, beta2=0.5).minimize(Loss, var_list=x)
    op_w = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5, beta2=0.5).minimize(-Loss, var_list=var_list)

# %%

sess = tf.Session()
sess.run(tf.global_variables_initializer())

G_vars = tf.global_variables()
G_g_vars = [var for var in G_vars if 'RMSProp' not in var.name and 'Adam' not in var.name]

saver = tf.train.Saver(var_list=G_g_vars, max_to_keep=5)

sample = x.get_shape()[0].value

# %%

start_time = time.time()

for i in range(Iter + 1):

    for _ in range(inner):
        sess.run(op_I)

    sess.run(op_w)

    out = sess.run([x, Loss] + LOSS_LIST)

    current_time = time.time()

    print(i, out[1], out[2:], 'already used %ds' % (current_time - start_time))

    if i % 500 == 0:
        if save_weights == 1:
            s = saver.save(sess, 'SAVE_WEIGHT/' + image1 + '/', global_step=i, write_meta_graph=False)

        for j in range(sample):
            tmp = (np.clip(out[0][j] + AVE, 0, 1) * 255.).astype(np.uint8)

            plt.imsave('Produce/' + 'TEST_02_' + image1 + '_' + 'inner_' + str(inner) + '_' + '_layer_S_' + str(
                layer_S) + '_layer_D_' + str(layer_D) + '_IsMean_' + str(mean) + '_Adam_' + str(Adam) + '_' + str(
                j) + '_' + str(i) + '_' + '.jpg', tmp)
