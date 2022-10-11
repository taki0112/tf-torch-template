from tf_utils import *
from tf_network import *
import time
import numpy as np

from tensorflow.python.data.experimental import AUTOTUNE

automatic_gpu_usage()

class DeepNetwork():
    def __init__(self, args, strategy):
        super(DeepNetwork, self).__init__()
        self.model_name = 'DeepNetwork'
        self.checkpoint_dir = args['checkpoint_dir']
        self.result_dir = args['result_dir']
        self.log_dir = args['log_dir']
        self.sample_dir = args['sample_dir']
        self.dataset_name = args['dataset']

        self.strategy = strategy
        self.NUM_GPUS = strategy.num_replicas_in_sync

        """ Network parameters """
        self.feature_size = args['feature_size']

        """ Training parameters """
        self.lr = args['lr']
        self.iteration = args['iteration']
        self.img_size = args['img_size']
        self.batch_size = args['batch_size']
        self.global_batch_size = self.batch_size * self.NUM_GPUS

        """ Misc """
        self.save_freq = args['save_freq']
        self.log_template = 'step [{}/{}]: elapsed: {:.2f}s, loss: {:.3f}'

        """ Directory """
        self.sample_dir = os.path.join(self.sample_dir, self.model_dir)
        check_folder(self.sample_dir)
        self.checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_dir)
        check_folder(self.checkpoint_dir)
        self.log_dir = os.path.join(self.log_dir, self.model_dir)
        check_folder(self.log_dir)

        """ Dataset """
        dataset_path = './dataset'
        self.dataset_path = os.path.join(dataset_path, self.dataset_name)

        """ Print """
        self.physical_gpus = tf.config.experimental.list_physical_devices('GPU')
        self.logical_gpus = tf.config.experimental.list_logical_devices('GPU')

    ##################################################################################
    # Model
    ##################################################################################
    def build_model(self):
        """ Dataset Iterator """
        dataset = ImageDataset(self.img_size, self.dataset_path)
        dataset.preprocess()

        self.dataset_num = len(dataset.train_images)

        dataset_slice = tf.data.Dataset.from_tensor_slices((dataset.train_images, dataset.train_labels))
        dataset_iter = dataset_slice.shuffle(buffer_size=self.dataset_num, reshuffle_each_iteration=True).repeat()
        dataset_iter = dataset_iter.map(map_func=dataset.image_processing, num_parallel_calls=AUTOTUNE).batch(self.batch_size, drop_remainder=True)
        dataset_iter = dataset_iter.prefetch(buffer_size=AUTOTUNE)
        dataset_iter = self.strategy.experimental_distribute_dataset(dataset_iter)
        self.dataset_iter = iter(dataset_iter)

        """ Network """
        self.network = NetModel(feature_size=self.feature_size)

        """ Finalize model (build) """
        images = np.ones([1, self.img_size, self.img_size, 3])
        _ = self.network(images)

        """ Optimizer """
        self.optim = tf.keras.optimizers.Adam(self.lr)

        """ Checkpoint """
        self.ckpt = tf.train.Checkpoint(network=self.network,
                                        optim=self.optim)
        self.manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_dir, max_to_keep=2)
        self.start_iteration = 0

        if self.manager.latest_checkpoint:
            self.ckpt.restore(self.manager.latest_checkpoint).expect_partial()
            self.start_iteration = int(self.manager.latest_checkpoint.split('-')[-1])

            print('Latest checkpoint restored!!')
            print('start iteration : ', self.start_iteration)
        else:
            print('Not restoring from saved checkpoint')


    def train_step(self, real_images, label):
        with tf.GradientTape() as tape:
            logit = self.network(real_images)
            loss = cross_entroy_loss(logit, label)
            loss = multi_gpu_loss(loss, global_batch_size=self.batch_size)

        gradients = tape.gradient(loss, self.network.trainable_variables)
        self.optim.apply_gradients(zip(gradients, self.network.trainable_variables))

        return loss

    """ Distribute Train """
    @tf.function
    def distribute_train_step(self, real_images, label):
        loss = self.strategy.run(self.train_step, args=[real_images, label])
        loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None)

        return loss

    def train(self):
        start_time = time.time()

        # setup tensorboards
        train_summary_writer = tf.summary.create_file_writer(self.log_dir)

        # start training
        print()
        print(self.dataset_path)
        print(len(self.physical_gpus), "Physical GPUs,", len(self.logical_gpus), "Logical GPUs")
        print("Global batch size : ", self.batch_size)
        print("Target image size : ", self.img_size)
        print("Save frequency : ", self.save_freq)
        print("TF Version :", tf.__version__)
        print('max_steps: {}'.format(self.iteration))
        print()
        losses = {'loss': 0.0}

        for idx in range(self.start_iteration, self.iteration):
            iter_start_time = time.time()

            real_img, label = next(self.dataset_iter)

            if idx == 0:
                n_params = self.network.count_params()
                print("network parameters : ", format(n_params, ','))

            loss = self.distribute_train_step(real_img, label)
            losses['loss'] = np.float64(loss)

            # tensorboard log
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', losses['loss'], step=idx)
                tf.summary.scalar('g_loss', losses['g_loss'], step=idx)


            elapsed = time.time() - iter_start_time
            print(self.log_template.format(idx, self.iteration, elapsed, losses['loss']))

            # save
            if np.mod(idx, self.save_freq) == 0 or idx == self.iteration - 1:
                self.manager.save(checkpoint_number=idx)

        # train finision
        self.manager.save(checkpoint_number=self.iteration)
        print("Total train time: %4.4f" % (time.time() - start_time))

    @property
    def model_dir(self):
        return "{}_{}_{}".format(self.model_name, self.dataset_name, self.img_size)