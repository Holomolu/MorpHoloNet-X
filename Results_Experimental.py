from MorpHoloNet_X import *
from PIL import Image

obj_ph_dir = '.\\Results\\obj_ph'
obj_abso_dir = '.\\Results\\obj_abso'
U_intensity_dir = '.\\Results\\U_intensity'

def reset_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

reset_directory(obj_ph_dir)
reset_directory(obj_abso_dir)
reset_directory(U_intensity_dir)

agent = twin_image_removal()
load_root = "./trained_model/experimental/main_training/1/"
agent.load_weights(load_root)

width, height = 200, 200
segment_size = width

z0 = 8100
dz_local = 80.0 * 0.001
z_min      = z0 - dz_local * 40
z_max      = z0 + dz_local * 40
z_min_obj  = z0 - dz_local * 10
z_max_obj  = z0 + dz_local * 10

x = tf.range(1, width + 1, dtype=tf.float32) / width
y = tf.range(1, height + 1, dtype=tf.float32) / height
xx, yy = tf.meshgrid(x, y)

phase_factor_real, phase_factor_abso, incident_light = agent.get_internal_values()
z_values = tf.range(z_min*100, (z_max + dz_local)*100, dz_local*100, dtype=tf.float32)/100

print('Saving obj_ph at depth of', z0)

for z in z_values:
    z_norm = (z - z_min) / (z_max - z_min)
    z_filled = tf.fill(xx.shape, float(z_norm))
    tensor_array = tf.stack([xx, yy, z_filled], axis=-1)
    tensor_array = tf.reshape(tensor_array, (-1, 3))
    obj, _ = agent.predict(tensor_array)
    obj = np.array(obj)
    obj = obj.reshape(width, height)
    obj = Image.fromarray(obj) #.astype(np.uint8))
    file_directory = os.path.join(obj_ph_dir, f"obj{z.numpy():.2f}.tif")
    obj.save(file_directory)

print('Saving obj_abso at depth of', z0)

for z in z_values:
    z_norm = (z - z_min) / (z_max - z_min)
    z_filled = tf.fill(xx.shape, float(z_norm))
    tensor_array = tf.stack([xx, yy, z_filled], axis=-1)
    tensor_array = tf.reshape(tensor_array, (-1, 3))
    _, abso = agent.predict(tensor_array)
    abso = np.array(abso)
    abso = abso.reshape(width, height)
    abso = Image.fromarray(abso) #.astype(np.uint8))
    file_directory = os.path.join(obj_abso_dir, f"obj{z.numpy():.2f}.tif")
    abso.save(file_directory)

print('Saving U_intensity at depth of', z0)

z_values_reverse = z_values[::-1]
real_following_ref = tf.multiply(tf.ones([width, height], dtype=tf.float32), U_incident_avg_real)
imag_following_ref = tf.zeros([width, height], dtype=tf.float32)
U_z_following_ref = tf.complex(real_following_ref, imag_following_ref)

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

for z in z_values_reverse:
    if z == z_max:
        U_z_following_ref_amp = tf.square(tf.abs(U_z_following_ref)).numpy()
        U_z_following_ref_amp = Image.fromarray(U_z_following_ref_amp)
        file_directory = os.path.join(U_intensity_dir, f"amp{z.numpy():.2f}.tif")
        U_z_following_ref_amp.save(file_directory)
    elif z == (z_max - dz_local):
        z_following = tf.fill([width, height], (z - z_min) / (z_max - z_min))
        tensor_array_following = tf.stack([xx, yy, z_following], axis=-1)
        tensor_array_following = tf.reshape(tensor_array_following, (-1, 3))
        classification_following_real, classification_following_abso = agent.Complex_field(tensor_array_following)
        classification_following_real = tf.reshape(classification_following_real, [width, height])
        classification_following_abso = tf.reshape(classification_following_abso, [width, height])
        phase_factor_complex_real = tf.complex(agent.phase_factor_real, 0.0)
        classification_following_complex_real = tf.complex(classification_following_real, tf.zeros_like(classification_following_real))
        phase_shift = tf.exp(tf.complex(0.0, 1.0) * -phase_factor_complex_real * classification_following_complex_real)
        U_z_following_prop = U_z_following_ref * phase_shift
        phase_factor_complex_abso = tf.complex(0.0, - agent.phase_factor_abso)
        classification_following_complex_abso = tf.complex(classification_following_abso, tf.zeros_like(classification_following_abso))
        abso = tf.exp(tf.complex(0.0, 1.0) * -phase_factor_complex_abso * classification_following_complex_abso)
        U_z_following_prop = U_z_following_prop * abso

        U_z_following_prop = angular_spectrum_propagator(U_z_following_prop, dz_local)
        U_z_following_prop_amp = tf.square(tf.abs(U_z_following_prop)).numpy()
        U_z_following_prop_amp = Image.fromarray(U_z_following_prop_amp)
        file_directory = os.path.join(U_intensity_dir, f"amp{z.numpy():.2f}.tif")
        U_z_following_prop_amp.save(file_directory)

    elif z == z_min:
        z_following = tf.fill([width, height], (z - z_min) / (z_max - z_min))
        tensor_array_following = tf.stack([xx, yy, z_following], axis=-1)
        tensor_array_following = tf.reshape(tensor_array_following, (-1, 3))
        classification_following_real, classification_following_abso = agent.Complex_field(tensor_array_following)
        classification_following_real = tf.reshape(classification_following_real, [width, height])
        classification_following_abso = tf.reshape(classification_following_abso, [width, height])
        phase_factor_complex_real = tf.complex(agent.phase_factor_real, 0.0)
        classification_following_complex_real = tf.complex(classification_following_real, tf.zeros_like(classification_following_real))
        phase_shift = tf.exp(tf.complex(0.0, 1.0) * -phase_factor_complex_real * classification_following_complex_real)
        U_z_following_prop = U_z_following_prop * phase_shift
        phase_factor_complex_abso = tf.complex(0.0, - agent.phase_factor_abso)
        classification_following_complex_abso = tf.complex(classification_following_abso, tf.zeros_like(classification_following_abso))
        abso = tf.exp(tf.complex(0.0, 1.0) * -phase_factor_complex_abso * classification_following_complex_abso)
        U_z_following_prop = U_z_following_prop * abso
        U_z_following_prop = angular_spectrum_propagator(U_z_following_prop, z_min)
        U_z_following_prop_amp = tf.square(tf.abs(U_z_following_prop)).numpy()
        U_z_following_prop_amp = Image.fromarray(U_z_following_prop_amp)
        file_directory = os.path.join(U_intensity_dir, f"amp{0:.2f}.tif")
        U_z_following_prop_amp.save(file_directory)
    else:
        z_following = tf.fill([width, height], (z - z_min) / (z_max - z_min))
        tensor_array_following = tf.stack([xx, yy, z_following], axis=-1)
        tensor_array_following = tf.reshape(tensor_array_following, (-1, 3))
        classification_following_real, classification_following_abso = agent.Complex_field(tensor_array_following)
        classification_following_real = tf.reshape(classification_following_real, [width, height])
        classification_following_abso = tf.reshape(classification_following_abso, [width, height])
        phase_factor_complex_real = tf.complex(agent.phase_factor_real, 0.0)
        classification_following_complex_real = tf.complex(classification_following_real, tf.zeros_like(classification_following_real))
        phase_shift = tf.exp(tf.complex(0.0, 1.0) * -phase_factor_complex_real * classification_following_complex_real)
        U_z_following_prop = U_z_following_prop * phase_shift
        phase_factor_complex_abso = tf.complex(0.0, - agent.phase_factor_abso)
        classification_following_complex_abso = tf.complex(classification_following_abso, tf.zeros_like(classification_following_abso))
        abso = tf.exp(tf.complex(0.0, 1.0) * -phase_factor_complex_abso * classification_following_complex_abso)
        U_z_following_prop = U_z_following_prop * abso
        U_z_following_prop = angular_spectrum_propagator(U_z_following_prop, dz_local)
        U_z_following_prop_amp = tf.square(tf.abs(U_z_following_prop)).numpy()
        U_z_following_prop_amp = Image.fromarray(U_z_following_prop_amp)
        file_directory = os.path.join(U_intensity_dir, f"amp{z.numpy():.2f}.tif")
        U_z_following_prop_amp.save(file_directory)
