import os
import datetime
from tqdm import tqdm
import tensorflow as tf
from model import Discriminator, Generator
from keras.models import Model
import numpy as np
from Utils import vgg_loss, load_training_data
from keras.layers import Input
from keras.optimizers import Adam

# ################ CONFIGURATION #################

INPUT_DIR = './Train_Images/'
SAVE_DIR = './Samples/'
MODEL_DIR = './Model/'
# in_shape = [None, None, 3]
hr_shape = [448, 448, 3]
d_factor = 4  # downscaling factor
prob = 0.85     # Real to fake probability

# ################ HYPER-PARAMETERS #################

# For Generator
g_epochs = 100

# Fot training
batch_size = 16
adam_lr = 1e-4  # Initial learning rate for Adam Optimizer
adam_b1 = 0.9
adam_b2 = 0.99
adam_epsilon = 1e-08
adam_decay = 0.1
n_epochs = 2000


class CTGAN:

    def __init__(self):
        self.adam_optimizer = Adam(
            lr=adam_lr, decay=adam_decay,
            beta_1=adam_b1, beta_2=adam_b2,
            epsilon=adam_epsilon)

        self.load_data = load_training_data(INPUT_DIR, '.png', 200, 0.8)
        self.lr_shape = [hr_shape[0]//d_factor, hr_shape[1]//d_factor, 3]

        self.img_lr = Input(shape=self.lr_shape)
        self.img_hr = Input(shape=hr_shape)

        # self.vgg = vgg_model(img_hr=hr_shape, optimizer=self.adam_optimizer)

        pass

    def GAN(self, lr_img, discriminator, generator, optimizer, loss):
        # cnt_loss = loss[2]

        network_input = lr_img

        g = generator(network_input)

        network_output = discriminator(g)

        model = Model(inputs=network_input, outputs=[g, network_output])
        model.compile(loss=[loss, "binary_crossentropy"], loss_weights=[1., 1e-3],
                      optimizer=optimizer)

        return model

    def train(self):

        start = datetime.datetime.now()
        train_lr, train_hr, test_lr, test_hr = self.load_data

        n_batches = (train_hr.shape[0] // batch_size)

        # Generator and Discriminator
        generator = Generator().build_generator(img_shape=self.img_lr, is_train=True, reuse=False)
        generator_output = self.img_lr # generator.outputs                    # Generator output from lr image

        discriminator = Discriminator().build_discriminator(img=self.img_hr, is_train=True, reuse=False)
        # discriminator_output = self.img_hr # discriminator.outputs            # Discriminator output on real hr image
        discriminator.trainable = False

        vgg_loss(g_out=generator_output, img_hr=self.img_hr)

        content_loss = vgg_loss

        generator.compile(loss=content_loss, optimizer=self.adam_optimizer)
        discriminator.compile(loss='binary_crossentropy', optimizer=self.adam_optimizer)

        gan = self.GAN(lr_img=self.img_lr,
                       discriminator=discriminator, generator=generator,
                       optimizer=self.adam_optimizer, loss=content_loss)

        save_loss = open(MODEL_DIR + 'losses.txt', 'w+')
        save_loss.close()

        print('Starting Epochs...\n\n')

        for epoch in range(1, n_epochs+1):
            print('\t'*4, 'Epoch %d'%epoch, '\t'*4)

            # tqdm to provide progress bar
            for t in tqdm(range(n_batches)):
                batch_hr, batch_lr = self.generate_batches(train_hr, train_lr)

                generator_output = generator.predict(batch_lr)

                # Manually training for certain probability
                real_Y, fake_Y = self.generate_Y()

                discriminator.trainable = True

                discriminator_loss = discriminator.train_on_batch(batch_hr, real_Y)*0.5
                discriminator_loss += discriminator.train_on_batch(generator_output, fake_Y)*0.5

                # Learning
                batch_hr, batch_lr = self.generate_batches(train_hr, train_lr)

                discriminator.trainable = False

                gan_Y, _ = self.generate_Y()
                gan_loss = gan.train_on_batch(batch_lr, [batch_hr, gan_Y])

            loss_file = open(MODEL_DIR + 'losses.txt', 'a')
            loss_file.write('epoch%d : gan_loss = %s ; discriminator_loss = %f\n' % (epoch, gan_loss, discriminator_loss))
            loss_file.close()

            if epoch % 500 == 0:
                generator.save(MODEL_DIR + 'gen_model%d.h5' % epoch)
                discriminator.save(MODEL_DIR + 'dis_model%d.h5' % epoch)
        print('Training time: ', start-datetime.datetime.now())
        pass

    def generate_Y(self):
        real = np.random.uniform(low=prob, high=1, size=(batch_size,))
        fake = np.random.uniform(low=1e-2, high=1-prob, size=(batch_size,))
        return real, fake

    def generate_batches(self, train_hr, train_lr):
        batch_index_hr = np.random.randint(0, len(train_hr), size=batch_size)

        batch_index_lr = np.random.randint(0, len(train_lr), size=batch_size)

        # Batches of Images
        batch_hr = train_hr[batch_index_hr]
        batch_lr = train_lr[batch_index_lr]

        return batch_hr, batch_lr


def main():
    print('INITIALIZING GAN...')
    gan = CTGAN()
    gan.train()
    pass


if __name__ == '__main__':
    main()

