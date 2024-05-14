import torch.utils.data

from torch_utils import *
import time
from torch_network import *
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
from functools import partial

# print = partial(print, flush=True)

def set_torch_backend(prefer_speed=True, seed=None):
    torch.backends.cudnn.enabled = True

    if prefer_speed:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True
    else: # prefer the performance
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        torch.backends.cudnn.benchmark = False

    if seed is None:
        torch.backends.cudnn.deterministic = False
    else:
        torch.backends.cudnn.deterministic = True
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def get_different_generator_for_each_rank(num_gpu, rank, seed=None):
    if seed is None:
        return None
    else:
        g = torch.Generator()
        rank_seed = seed * num_gpu + rank
        g.manual_seed(rank_seed)
        return g
           
def run_fn(rank, args, world_size):
    device = torch.device('cuda', rank)
    set_torch_backend(prefer_speed=True, seed=args['seed'])

    model = DeepNetwork(args, world_size)
    model.build_model(rank, device)
    model.train_model(rank, device)

class DeepNetwork():
    def __init__(self, args, NUM_GPUS):
        super(DeepNetwork, self).__init__()
        self.model_name = 'DeepNetwork'
        self.checkpoint_dir = args['checkpoint_dir']
        self.log_dir = args['log_dir']
        self.sample_dir = args['sample_dir']
        self.dataset_name = args['dataset']
        self.seed = args['seed']

        self.NUM_GPUS = NUM_GPUS

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

        """ Dataset """
        dataset_path = './dataset'
        self.dataset_path = os.path.join(dataset_path, self.dataset_name)

    ##################################################################################
    # Model
    ##################################################################################
    def build_model(self, rank, device):
        """ Init process """
        build_init_procss(rank, world_size=self.NUM_GPUS, device=device)

        if rank == 0:
            """ Directory """
            self.sample_dir = os.path.join(self.sample_dir, self.model_dir)
            check_folder(self.sample_dir)
            self.checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_dir)
            check_folder(self.checkpoint_dir)
            self.log_dir = os.path.join(self.log_dir, self.model_dir)
            check_folder(self.log_dir)

        """ Dataset Load """
        dataset = ImageDataset(dataset_path=self.dataset_path, img_size=self.img_size)
        self.dataset_num = dataset.__len__()
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, num_workers=4,
                                             sampler=distributed_sampler(dataset, rank=rank, num_replicas=self.NUM_GPUS, shuffle=True),
                                             drop_last=True, pin_memory=True,
                                             generator=get_different_generator_for_each_rank(self.NUM_GPUS, rank, seed=self.seed))
        self.dataset_iter = infinite_iterator(loader)


        """ Network """
        self.network = NetModel(input_shape=self.img_size, feature_size=self.feature_size).to(device)

        """ Optimizer """
        self.optim = torch.optim.Adam(self.network.parameters(), lr=self.lr)

        """ Distributed Learning """
        self.network = dataparallel_and_sync(self.network, rank)

        """ Checkpoint """
        latest_ckpt_name, start_iter = find_latest_ckpt(self.checkpoint_dir)

        if latest_ckpt_name is not None:
            if rank == 0:
                # "rank == 0" means a first gpu.
                print('Latest checkpoint restored!! ', latest_ckpt_name)
                print('start iteration : ', start_iter)
            self.start_iteration = start_iter

            latest_ckpt = os.path.join(self.checkpoint_dir, latest_ckpt_name)
            ckpt = torch.load(latest_ckpt, map_location=device)

            self.network.load_state_dict(ckpt["network"])
            self.optim.load_state_dict(ckpt["optim"])

        else:
            if rank == 0:
                print('Not restoring from saved checkpoint')
            self.start_iteration = 0

    def train_step(self, real_images, label, device=torch.device('cuda')):
        # gradient check
        requires_grad(self.network, True)

        # forward pass
        logit = self.network(real_images)

        # loss
        loss = cross_entroy_loss(logit, label)

        # backword
        apply_gradients(loss, self.optim)

        return loss

    def train_model(self, rank, device):
        start_time = time.time()
        fid_start_time = time.time()

        # setup tensorboards
        train_summary_writer = SummaryWriter(self.log_dir)

        # start training
        if rank == 0:
            print()
            print(self.dataset_path)
            print("Dataset number : ", self.dataset_num)
            print("GPUs : ", self.NUM_GPUS)
            print("Each batch size : ", self.batch_size)
            print("Global batch size : ", self.global_batch_size)
            print("Target image size : ", self.img_size)
            print("Save frequency : ", self.save_freq)
            print("PyTorch Version :", torch.__version__)
            print('max_steps: {}'.format(self.iteration))
            print()


        for idx in range(self.start_iteration, self.iteration):
            iter_start_time = time.time()

            real_img, label = next(self.dataset_iter)
            real_img = real_img.to(device)
            label = label.to(device)

            if idx == 0:
                if rank == 0:
                    print("count params")
                    n_params = count_parameters(self.network)
                    print("network parameters : ", format(n_params, ','))

            loss = self.train_step(real_img, label, device=device)

            loss = reduce_loss(loss)

            if rank == 0:
                for k, v in losses.items():
                    train_summary_writer.add_scalar(k, v, global_step=idx)


                elapsed = time.time() - iter_start_time
                print(self.log_template.format(idx, self.iteration, elapsed, loss.item()))

                if np.mod(idx, self.save_freq) == 0 or idx == self.iteration - 1:
                    if rank == 0:
                        self.torch_save(idx)

            dist.barrier()

        if rank == 0:
            # save model for final step
            self.torch_save(self.iteration)
            print("Total train time: %4.4f" % (time.time() - start_time))

        dist.barrier()

    def torch_save(self, idx):
        torch.save(
            {
                'network': self.network.state_dict(),
                'optim': self.optim.state_dict()
            },
            os.path.join(self.checkpoint_dir, 'iter_{}.pt'.format(idx))
        )

    @property
    def model_dir(self):
        return "{}_{}_{}".format(self.model_name, self.dataset_name, self.img_size)
