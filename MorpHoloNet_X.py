import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from positional_encoding import positional_encoding
import numpy as np
from time import time
from PIL import Image
import math
import os
import shutil

file_name = '.\\holograms\\hologram_noisy.tif'
image_array = Image.open(file_name)
width, height = image_array.size
image_array = np.array(image_array) # / 255.0
image_array = image_array.reshape(width, height)

file_name_mask = '.\\holograms\\segmentation.tif'
image_array_mask = Image.open(file_name_mask)
width, height = image_array_mask.size
image_array_mask = np.array(image_array_mask) / 255.0
image_array_mask = image_array_mask.reshape(width, height)

image_array_norm = image_array

image_tensor = tf.convert_to_tensor(image_array_norm)
image_array_mask = tf.convert_to_tensor(image_array_mask)

U_z0 = tf.cast(image_tensor, tf.float32)
U_incident_avg_real = tf.sqrt(1.0)
image_array_mask = tf.cast(image_array_mask, tf.float32)

width = 128
height = 128

minPX = 1
maxPX = width
minPY = 1
maxPY = height
segment_size = width

z_obj_plane = 100
dz = 20.0*0.001 # micron
z_min = z_obj_plane - dz*40
z_max = z_obj_plane + dz*40
num_samples = 10

z_min_obj = z_obj_plane - dz*5
z_max_obj = z_obj_plane + dz*5

x = tf.range(1, width + 1, dtype=tf.float32) / width
y = tf.range(1, height + 1, dtype=tf.float32) / height
xx, yy = tf.meshgrid(x, y)

def create_z_dataset(z_min, z_max, dz, batch_size):
    z_values = tf.range(z_min*100, (z_max + dz)*100, dz*100, dtype=tf.float32)
    z_values = z_values/100
    tf.random.shuffle(z_values)
    num_batches = int(np.ceil(len(z_values) / batch_size))
    batches = []

    for i in range(num_batches):
        start_index = i * batch_size
        end_index = start_index + batch_size
        batch_values = z_values[start_index:end_index]
        batches.append(batch_values)
    return np.array(batches)

def angular_spectrum_propagator(image, depth):
    H = tf.signal.fft2d(tf.cast(image, tf.complex128))
    M2 = segment_size
    N2 = segment_size
    physicalLength = 10.0*0.001 # micron
    waveLength = 0.1327*0.001 # micron
    k = 2 * math.pi / waveLength
    u = tf.linspace(0, M2 - 1, M2)
    v = tf.linspace(0, N2 - 1, N2)
    u = tf.where(u > M2 / 2, u - M2, u)
    v = tf.where(v > N2 / 2, v - N2, v)

    V, U = tf.meshgrid(v, u)
    U = tf.cast(waveLength * U / (M2 * physicalLength), tf.float64)
    V = tf.cast(waveLength * V / (N2 * physicalLength), tf.float64)
    U2 = tf.cast(U**2, tf.complex128)
    V2 = tf.cast(V**2, tf.complex128)
    F = 1j * k * tf.sqrt(1 - U2 - V2)

    if depth == 0:
        simulated_image = tf.cast(image, dtype=tf.complex64)
    else:
        depth = tf.cast(depth, tf.complex128)
        G = tf.exp(depth * F)
        recons = tf.signal.ifft2d(H * G)
        simulated_image = tf.cast(recons[0:M2, 0:N2], dtype=tf.complex64)

    return simulated_image

class MorpHoloNet_X(Model):
    def __init__(self):
        super(MorpHoloNet_X, self).__init__()

        initializer_hidden = tf.keras.initializers.LecunNormal
        initializer_output = tf.keras.initializers.GlorotUniform
        activator = tf.keras.activations.swish

        self.h1 = positional_encoding(gaussian_projection=64, gaussian_scale=15.0)
        self.h2 = Dense(128, activation=activator, kernel_initializer=initializer_hidden)
        self.h3 = Dense(128, activation=activator, kernel_initializer=initializer_hidden)
        self.h4 = Dense(128, activation=activator, kernel_initializer=initializer_hidden)
        self.u = Dense(1, activation='sigmoid', kernel_initializer=initializer_output)
        self.v = Dense(1, activation='sigmoid', kernel_initializer=initializer_output)

    def call(self, pos):
        x = self.h1(pos)
        x = self.h2(x)
        x = self.h3(x)
        x = self.h4(x)
        out_real = self.u(x)
        out_abso = self.v(x)

        return out_real, out_abso

class twin_image_removal(object):
    def __init__(self):
            self.lr_free = 0.0001
            self.opt_free = Adam(self.lr_free)

            self.lr_prior = 0.001
            self.opt_prior = Adam(self.lr_prior)

            self.phase_factor_real = tf.Variable(0.3, dtype=tf.float32, name='phase_factor_real')
            self.phase_factor_abso = tf.Variable(0.03, dtype=tf.float32, name='phase_factor_abso')
            self.incident_light = tf.Variable(U_incident_avg_real, dtype=tf.float32, name='incident_light')

            self.Complex_field = MorpHoloNet_X()
            self.Complex_field.build(input_shape=(None, 3))

            self.train_loss_history = []
            self.iter_count = 0
            self.instant_loss = 0

    @tf.function
    def compute_loss_and_grads(self, z_batch):
        with tf.GradientTape() as tape:
            #tape.watch([self.phase_factor_real])
            #tape.watch([self.phase_factor_abso])
            #tape.watch([self.incident_light])
            loss = tf.constant(0.0, dtype=tf.float32)
            U_z_following_prop = tf.complex(tf.zeros_like(U_z0), tf.zeros_like(U_z0))

            z_values = tf.range(z_min*100, (z_max + dz)*100, dz*100, dtype=tf.float32)/100
            z_values = z_values[::-1]  # [z, z-dz, z-2*dz, ..., z_min]

            for z in z_values:
                if z == z_max:
                    z_following = tf.fill([width, height], (z - z_min) / (z_max - z_min))
                    z_preceding = tf.fill([width, height], (z - dz - z_min) / (z_max - z_min))
                    tensor_array_following = tf.stack([xx, yy, z_following], axis=-1)
                    tensor_array_preceding = tf.stack([xx, yy, z_preceding], axis=-1)
                    tensor_array_following = tf.reshape(tensor_array_following, (-1, 3))
                    tensor_array_preceding = tf.reshape(tensor_array_preceding, (-1, 3))

                    classification_following_real, classification_following_abso  = self.Complex_field(tensor_array_following)
                    classification_preceding_real, classification_preceding_abso = self.Complex_field(tensor_array_preceding)
                    classification_following_real = tf.reshape(classification_following_real, [width, height])
                    classification_following_abso = tf.reshape(classification_following_abso, [width, height])
                    classification_preceding_real = tf.reshape(classification_preceding_real, [width, height])
                    classification_preceding_abso = tf.reshape(classification_preceding_abso, [width, height])

                    real_following_ref = tf.multiply(tf.ones([width, height], dtype=tf.float32), self.incident_light)
                    imag_following_ref = tf.zeros([width, height], dtype=tf.float32)
                    U_z_following_ref = tf.complex(real_following_ref, imag_following_ref)
                    phase_factor_complex_real = tf.complex(self.phase_factor_real, 0.0)
                    classification_preceding_complex_real = tf.complex(classification_preceding_real, tf.zeros_like(classification_preceding_real))
                    phase_shift = tf.exp(tf.complex(0.0, 1.0) * -phase_factor_complex_real * classification_preceding_complex_real)
                    U_z_following_ref *= phase_shift

                    phase_factor_complex_abso = tf.complex(0.0, -self.phase_factor_abso)
                    classification_preceding_complex_abso = tf.complex(classification_preceding_abso, tf.zeros_like(classification_preceding_abso))
                    phase_abso = tf.exp(tf.complex(0.0, 1.0) * -phase_factor_complex_abso * classification_preceding_complex_abso)
                    U_z_following_ref *= phase_abso

                    U_z_following_prop = angular_spectrum_propagator(U_z_following_ref, dz)

                    loss += tf.reduce_mean(tf.square(classification_following_real))
                    loss += tf.reduce_mean(tf.square(classification_following_abso))

                elif z == z_min:
                    z_preceding = tf.fill([width, height], (z - dz - z_min) / (z_max - z_min))
                    tensor_array_preceding = tf.stack([xx, yy, z_preceding], axis=-1)
                    tensor_array_preceding = tf.reshape(tensor_array_preceding, (-1, 3))

                    classification_preceding_real, classification_preceding_abso = self.Complex_field(tensor_array_preceding)
                    classification_preceding_real = tf.reshape(classification_preceding_real, [width, height])
                    classification_preceding_abso = tf.reshape(classification_preceding_abso, [width, height])

                    phase_factor_complex_real = tf.complex(self.phase_factor_real, 0.0)
                    classification_preceding_complex_real = tf.complex(classification_preceding_real, tf.zeros_like(classification_preceding_real))
                    phase_shift = tf.exp(tf.complex(0.0, 1.0) * -phase_factor_complex_real * classification_preceding_complex_real)
                    U_z_following_prop *= phase_shift

                    phase_factor_complex_abso = tf.complex(0.0, -self.phase_factor_abso)
                    classification_preceding_complex_abso = tf.complex(classification_preceding_abso, tf.zeros_like(classification_preceding_abso))
                    phase_abso = tf.exp(tf.complex(0.0, 1.0) * -phase_factor_complex_abso * classification_preceding_complex_abso)
                    U_z_following_prop *= phase_abso

                    U_z_following_prop = angular_spectrum_propagator(U_z_following_prop, z_min)
                    U_z_following_prop_amp = tf.square(tf.abs(U_z_following_prop))

                    loss += tf.reduce_mean(tf.square(U_z0 - U_z_following_prop_amp))
                    loss += tf.reduce_mean(tf.square(classification_preceding_real))
                    loss += tf.reduce_mean(tf.square(classification_preceding_abso))

                else:
                    z_preceding = tf.fill([width, height], (z - dz - z_min) / (z_max - z_min))
                    tensor_array_preceding = tf.stack([xx, yy, z_preceding], axis=-1)
                    tensor_array_preceding = tf.reshape(tensor_array_preceding, (-1, 3))

                    classification_preceding_real, classification_preceding_abso = self.Complex_field(tensor_array_preceding)
                    classification_preceding_real = tf.reshape(classification_preceding_real, [width, height])
                    classification_preceding_abso = tf.reshape(classification_preceding_abso, [width, height])

                    phase_factor_complex_real = tf.complex(self.phase_factor_real, 0.0)
                    classification_preceding_complex_real = tf.complex(classification_preceding_real, tf.zeros_like(classification_preceding_real))
                    phase_shift = tf.exp(tf.complex(0.0, 1.0) * -phase_factor_complex_real * classification_preceding_complex_real)
                    U_z_following_prop *= phase_shift

                    phase_factor_complex_abso = tf.complex(0.0, -self.phase_factor_abso)
                    classification_preceding_complex_abso = tf.complex(classification_preceding_abso, tf.zeros_like(classification_preceding_abso))
                    phase_abso = tf.exp(tf.complex(0.0, 1.0) * -phase_factor_complex_abso * classification_preceding_complex_abso)
                    U_z_following_prop *= phase_abso

                    U_z_following_prop = angular_spectrum_propagator(U_z_following_prop, dz)

        grads = tape.gradient(loss, self.Complex_field.trainable_variables) # + [self.phase_factor_real] + [self.phase_factor_abso] + [self.incident_light])

        del tape

        return loss, grads

    def compute_loss_and_grads_BC(self, z_batch):
        with tf.GradientTape() as tape:
            loss = tf.constant(0.0, dtype=tf.float32)

            x_BC = tf.range(1, width + 1, dtype=tf.float32) / width
            y_BC = tf.range(1, height + 1, dtype=tf.float32) / height
            z_BC = (tf.range(z_min, z_max + dz, dz, dtype=tf.float32) - z_min) / (z_max - z_min)

            offset = 0.0

            xx_BC, zz_BC = tf.meshgrid(x_BC, z_BC)
            yy_0 = tf.fill(xx_BC.shape, 0.0 + (offset / height))
            yy_1 = tf.fill(xx_BC.shape, 1.0 - (offset / height))

            tensor_array_BC_0 = tf.stack([xx_BC, yy_0, zz_BC], axis=-1)
            tensor_array_BC_1 = tf.stack([xx_BC, yy_1, zz_BC], axis=-1)

            tensor_array_BC_0 = tf.reshape(tensor_array_BC_0, (-1, 3))
            tensor_array_BC_1 = tf.reshape(tensor_array_BC_1, (-1, 3))

            classification_BC_0_real, classification_BC_0_abso = self.Complex_field(tensor_array_BC_0)
            classification_BC_1_real, classification_BC_1_abso = self.Complex_field(tensor_array_BC_1)

            loss += tf.reduce_mean(tf.square(classification_BC_0_real))
            loss += tf.reduce_mean(tf.square(classification_BC_0_abso))
            loss += tf.reduce_mean(tf.square(classification_BC_1_real))
            loss += tf.reduce_mean(tf.square(classification_BC_1_abso))

            yy_BC, zz_BC = tf.meshgrid(y_BC, z_BC)
            xx_0 = tf.fill(yy_BC.shape, 0.0 + (offset / width))
            xx_1 = tf.fill(yy_BC.shape, 1.0 - (offset / width))

            tensor_array_BC_2 = tf.stack([xx_0, yy_BC, zz_BC], axis=-1)
            tensor_array_BC_3 = tf.stack([xx_1, yy_BC, zz_BC], axis=-1)

            tensor_array_BC_2 = tf.reshape(tensor_array_BC_2, (-1, 3))
            tensor_array_BC_3 = tf.reshape(tensor_array_BC_3, (-1, 3))

            classification_BC_2_real, classification_BC_2_abso = self.Complex_field(tensor_array_BC_2)
            classification_BC_3_real, classification_BC_3_abso = self.Complex_field(tensor_array_BC_3)

            loss += tf.reduce_mean(tf.square(classification_BC_2_real))
            loss += tf.reduce_mean(tf.square(classification_BC_2_abso))
            loss += tf.reduce_mean(tf.square(classification_BC_3_real))
            loss += tf.reduce_mean(tf.square(classification_BC_3_abso))

        grads = tape.gradient(loss, self.Complex_field.trainable_variables)

        del tape

        return loss, grads

    def compute_loss_and_grads_prior(self, z_batch):
        with tf.GradientTape() as tape:
            loss = tf.constant(0.0, dtype=tf.float32)
            for z in z_batch:
                z_following = tf.fill([width, height], (z - z_min) / (z_max - z_min))

                tensor_array_following = tf.stack([xx, yy, z_following], axis=-1)
                tensor_array_following = tf.reshape(tensor_array_following, (-1, 3))
                classification_following_real, classification_following_abso = self.Complex_field(tensor_array_following)
                classification_following_real = tf.reshape(classification_following_real, [width, height])
                classification_following_abso = tf.reshape(classification_following_abso, [width, height])
                if z > z_min_obj and z < z_max_obj:
                    loss += tf.reduce_mean(tf.square(classification_following_real - image_array_mask))
                    loss += tf.reduce_mean(tf.square(classification_following_abso - image_array_mask))
                else:
                    loss += tf.reduce_mean(tf.square(classification_following_real - 0.0))
                    loss += tf.reduce_mean(tf.square(classification_following_abso - 0.0))

        grads = tape.gradient(loss, self.Complex_field.trainable_variables)

        del tape

        return loss, grads

    def save_weights(self, path):
        self.Complex_field.save_weights(path + 'Complex_field.h5')
        np.save(path + 'phase_factor_real.npy', self.phase_factor_real.numpy())
        np.save(path + 'phase_factor_abso.npy', self.phase_factor_abso.numpy())
        np.save(path + 'incident_light.npy', self.incident_light.numpy())

    def load_weights(self, path):
        self.Complex_field.load_weights(path + 'Complex_field.h5')
        self.phase_factor_real.assign(np.load(path + 'phase_factor_real.npy'))
        self.phase_factor_abso.assign(np.load(path + 'phase_factor_abso.npy'))
        self.incident_light.assign(np.load(path + 'incident_light.npy'))

    def callback(self, arg=None):
        if self.iter_count % 1 == 0:
            print('iter=', self.iter_count, ', loss=', self.instant_loss, ', phase_factor_real=', self.phase_factor_real.numpy(), ', phase_factor_abso=', self.phase_factor_abso.numpy(), ', incident_light=', self.incident_light.numpy())
            self.train_loss_history.append([self.iter_count, self.instant_loss])
        self.iter_count += 1

    def train_with_adam(self, adam_num, batch_size, save_root):
        def learn(z_batch):
            loss, grads = self.compute_loss_and_grads(z_batch)
            self.opt_free.apply_gradients(
                zip(grads, self.Complex_field.trainable_variables)) # + [self.phase_factor_real] + [self.phase_factor_abso] + [self.incident_light]))
            return loss

        def learn_BC(z_batch):
            loss, grads = self.compute_loss_and_grads_BC(z_batch)
            self.opt_free.apply_gradients(
                zip(grads, self.Complex_field.trainable_variables))
            return loss

        def learn_prior(z_batch):
            loss, grads = self.compute_loss_and_grads_prior(z_batch)
            self.opt_prior.apply_gradients(
                zip(grads, self.Complex_field.trainable_variables))
            return loss

        weights_dir = os.path.join(save_root, "weights")
        os.makedirs(weights_dir, exist_ok=True)

        for epoch in range(adam_num):

            if epoch <= 300: # epoch for pre-training
                loss = tf.constant(0.0, dtype=tf.float32)
                dataset = create_z_dataset(z_min, z_max, dz, batch_size)
                for z_batch in dataset:
                    if len(z_batch) == 1:
                        z_batch = [z_batch]
                    loss += learn_prior(z_batch)
            else:
                z_batch = [z_max]
                loss = learn(z_batch)
                loss += learn_BC(z_batch)

            if (epoch + 1) % 100 == 0:
                out_dir = os.path.join(weights_dir, f"checkpoint_{epoch + 1}")
                os.makedirs(out_dir, exist_ok=True)

                self.Complex_field.save_weights(out_dir + '/Complex_field.h5')
                np.save(out_dir + '/phase_factor_real.npy', self.phase_factor_real.numpy())
                np.save(out_dir + '/phase_factor_abso.npy', self.phase_factor_abso.numpy())
                np.save(out_dir + '/incident_light.npy', self.incident_light.numpy())


            self.instant_loss = loss.numpy()
            self.callback()

    def predict(self, pos):
        output = self.Complex_field(pos)
        return output

    def train(self, adam_num_free_training, batch_size,save_root):
        t0 = time()
        self.train_with_adam(adam_num_free_training, batch_size, save_root)
        print('\nComputation time of adam free training: {} seconds'.format(time() - t0))

    def get_internal_values(self):
        return self.phase_factor_real.numpy(), self.phase_factor_abso.numpy(), self.incident_light.numpy()

def reset_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

def main():
    adam_num_train = 3300
    batch_size = 16

    save_root = "./save_weights"  # 모든 z0 결과를 모아둘 상위 폴더
    os.makedirs(save_root, exist_ok=True)
    reset_directory(save_root)

    agent = twin_image_removal()
    agent.train(adam_num_train, batch_size, save_root)
    agent.save_weights(save_root + '/')
    np.savetxt(os.path.join(save_root, "loss.txt"),
    agent.train_loss_history, fmt="%.8f")
    phase_factor_real, phase_factor_abso, incident_light = agent.get_internal_values()
    print(f"phase_factor_real={phase_factor_real:.6f}, " f"phase_factor_abso={phase_factor_abso:.6f}, "f"incident_light={incident_light:.6f}\n")

if __name__ == "__main__":
    main()