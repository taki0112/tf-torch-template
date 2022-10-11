import tensorflow as tf
import os
from glob import glob
from tensorflow.keras import layers
class ImageDataset:
    def __init__(self, img_size, dataset_path):
        self.img_size = img_size
        self.dataset_path = dataset_path

    def image_processing(self, filename, label):
        x = tf.io.read_file(filename)
        x_decode = tf.image.decode_jpeg(x, channels=3, dct_method='INTEGER_ACCURATE')
        img = tf.image.resize(x_decode, [self.img_size, self.img_size], antialias=True,
                              method=tf.image.ResizeMethod.BICUBIC)
        img = (img / 255.0) # [0, 255] -> [0, 1]
        img = (img * 2) - 1.0

        return img, label

    def preprocess(self):
        self.train_images = glob(os.path.join(self.dataset_path, '*.png')) + glob(os.path.join(self.dataset_path, '*.jpg'))
        self.train_labels = [] # make label

def automatic_gpu_usage():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def multi_gpu_loss(x, global_batch_size):
    ndim = len(x.shape)
    no_batch_axis = list(range(1, ndim))
    x = tf.reduce_mean(x, axis=no_batch_axis)
    x = tf.reduce_sum(x) / global_batch_size

    return x

def str2bool(x):
    return x.lower() in ('true')

def cross_entroy_loss(logit, label):

    loss = tf.keras.losses.categorical_crossentropy(y_pred=logit, y_true=label, from_logits=True)

    return loss