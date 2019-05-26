import tensorflow as tf
from keras.layers import Conv2D, BatchNormalization, add, UpSampling2D, Flatten, Dense
from keras.models import Model


rs_blocks = 16


def leaky_relu(features):
    return tf.nn.leaky_relu(features=features, alpha=0.2)


class Discriminator:

    def __init__(self):
        print('Initializing Discriminator network...')
        pass

    def build_discriminator(self, img, is_train, reuse):
        W = tf.random_normal_initializer(stddev=0.02)
        B = tf.random_normal_initializer(stddev=0)
        gamma = tf.random_normal_initializer(1, 0.02)
        momentum = 0.5

        with tf.variable_scope('Discriminator', reuse=reuse) as vs:
            # print('Initializing Discriminator...')
            input0 = img

            # layer 0
            n_0 = Conv2D(64, kernel_size=[3, 3], strides=[1, 1], kernel_initializer=W, bias_initializer=B,
                         padding='SAME', activation=leaky_relu, name='N64S1/C0')(input0)

            n = Conv2D(64, kernel_size=[3, 3], strides=[2, 2], kernel_initializer=W, bias_initializer=B,
                       padding='SAME', activation=leaky_relu, name='N64S1/C1')(n_0)
            n = BatchNormalization(momentum=momentum, gamma_initializer=gamma, name='N64S1/B0')(n)

            n = Conv2D(128, kernel_size=[3, 3], strides=[1, 1], kernel_initializer=W, bias_initializer=B,
                       padding='SAME', activation=leaky_relu, name='N64S1/C2')(n)
            n = BatchNormalization(momentum=momentum, gamma_initializer=gamma, name='N64S1/B1')(n)

            n = Conv2D(128, kernel_size=[3, 3], strides=[2, 2], kernel_initializer=W, bias_initializer=B,
                       padding='SAME', activation=leaky_relu, name='N64S1/C3')(n)
            n = BatchNormalization(momentum=momentum, gamma_initializer=gamma, name='N64S1/B2')(n)

            n = Conv2D(256, kernel_size=[3, 3], strides=[1, 1], kernel_initializer=W, bias_initializer=B,
                       padding='SAME', activation=leaky_relu, name='N64S1/C4')(n)
            n = BatchNormalization(momentum=momentum, gamma_initializer=gamma, name='N64S1/B3')(n)

            n = Conv2D(256, kernel_size=[3, 3], strides=[2, 2], kernel_initializer=W, bias_initializer=B,
                       padding='SAME', activation=leaky_relu, name='N64S1/C5')(n)
            n = BatchNormalization(momentum=momentum, gamma_initializer=gamma, name='N64S1/B4')(n)

            n = Conv2D(512, kernel_size=[3, 3], strides=[1, 1], kernel_initializer=W, bias_initializer=B,
                       padding='SAME', activation=leaky_relu, name='N64S1/C6')(n)
            n = BatchNormalization(momentum=momentum, gamma_initializer=gamma, name='N64S1/B5')(n)

            n = Conv2D(512, kernel_size=[3, 3], strides=[2, 2], kernel_initializer=W, bias_initializer=B,
                       padding='SAME', activation=leaky_relu, name='N64S1/C7')(n)
            n = BatchNormalization(momentum=momentum, gamma_initializer=gamma, name='N64S1/B6')(n)

            n = Flatten()(n)
            n = Dense(1024, activation=leaky_relu)(n)
            n = Dense(1, activation=tf.nn.sigmoid)(n)

            model = Model(inputs=input0, outputs=n)

            return model


class Generator:

    def __init__(self):
        print('Initializing Generator network...')
        pass

    def residual_block(self, n, W, B, m):
        for i in range(rs_blocks):
            rs = Conv2D(64, kernel_size=[3, 3], strides=[1, 1], kernel_initializer=W, bias_initializer=B,
                        padding='SAME', activation=tf.nn.relu, name='N64S1/RC0')(n)
            rs = BatchNormalization(momentum=m)(rs)
            rs = Conv2D(64, kernel_size=[3, 3], strides=[1, 1], kernel_initializer=W, bias_initializer=B,
                        padding='SAME', activation=tf.nn.relu, name='N64S1/RC1')(rs)
            rs = BatchNormalization(momentum=m)(rs)

            rs = add([n, rs])
        return rs

    def build_generator(self, img_shape, is_train, reuse):

        W = tf.random_normal_initializer(stddev=0.02)
        B = tf.random_normal_initializer(stddev=0)
        gamma = tf.random_normal_initializer(1, 0.02)
        momentum = 0.5

        with tf.variable_scope('Generator', reuse=reuse) as vs:
            # print('Initializing Generator...')
            input0 = img_shape

            # first layer - 0;  name='n64s1/c'
            n = Conv2D(64, kernel_size=[3, 3], strides=[1, 1], kernel_initializer=W, bias_initializer=B,
                       padding='SAME', activation=tf.nn.relu, name='N64S1/C0')(input0)

            n_0 = n

            # Begin Residual Layer
            n = self.residual_block(n, W, B, momentum)

            #
            n = Conv2D(64, kernel_size=[3, 3], strides=[1, 1], kernel_initializer=W, bias_initializer=B,
                       padding='SAME', activation=None, name='N64S1/C1')(n)
            n = BatchNormalization(momentum=momentum, gamma_initializer=gamma, name='N64S1/B0')(n)

            n = add([n_0, n])
            # Ending residual block

            print('Residual block finished...\nUp-Sampling...')

            # Up-Sampling part 1.
            n = Conv2D(256, kernel_size=[3, 3], strides=[1, 1], kernel_initializer=W, bias_initializer=B,
                       padding='SAME', activation=tf.nn.relu, name='N256S1/C2')(n)
            n = UpSampling2D(interpolation='bilinear', name='SubPixel1')(n)

            # Up-Sampling part 2.
            n = Conv2D(256, kernel_size=[3, 3], strides=[1, 1], kernel_initializer=W, bias_initializer=B,
                       padding='SAME', activation=tf.nn.relu, name='N256S1/C3')(n)
            n = UpSampling2D(interpolation='bilinear', name='SubPixel2')(n)

            # Convolve again
            n = Conv2D(filters=3, kernel_size=[1, 1], strides=[1, 1], kernel_initializer=W, bias_initializer=B,
                       padding='SAME', activation=tf.nn.tanh, name='N256S1/C4')(n)

            model = Model(inputs=input0, outputs=n)

        return model



