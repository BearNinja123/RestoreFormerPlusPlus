import argparse, os, sys, datetime, glob, importlib
from omegaconf import OmegaConf
import numpy as np
from PIL import Image
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
import torch
import torchvision
from torch.utils.data import random_split, DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar, Tqdm
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities.rank_zero import rank_zero_only
<<<<<<< Updated upstream
from copy import deepcopy
import random, wandb
torch.set_float32_matmul_precision('high')
=======
from pytorch_lightning.trainer import Trainer
from pytorch_lightning import seed_everything
from copy import deepcopy
import random, wandb
torch.set_float32_matmul_precision('high')
torch._dynamo.config.cache_size_limit = 256
torch.backends.cudnn.benchmark = True
>>>>>>> Stashed changes

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "--pretrain",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="pretrain with existed weights",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument("-p", "--project", help="name of new or path to existing project")
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "--random-seed",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )

    parser.add_argument(
        "--root-path",
        type=str,
        default="./",
        help="root path for saving checkpoints and logs"
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=1,
        help="number of gpu nodes",
    )
    parser.add_argument(
        "--enable-profiler",
        type=str2bool,
        default=False,
        help="enable PyTorch profiler (default: False)",
    )

    return parser

def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    #parser = cli.add_arguments_to_parser(parser)
    #parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    if 'basicsr.data' in config["target"] or \
        'FFHQDegradationDataset' in config["target"] or \
        'FFHQAugDataset' in config["target"] or \
        'FFHQUnevenDegradationDataset' in config["target"]:
        return get_obj_from_str(config["target"])(config.get("params", dict()))
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""
    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None,
                 wrap=False, num_workers=None):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size*2
        if train is not None:
            self.dataset_configs["train"] = train
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = self._val_dataloader
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = self._test_dataloader
        self.wrap = wrap

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])
<<<<<<< Updated upstream

    def _train_dataloader(self):
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True, persistent_workers=True, pin_memory=True)
=======
    
    def train_dataloader(self):
        if hasattr(self, '_train_dataloader'):
            return self._train_dataloader
        self._train_dataloader = DataLoader(self.datasets["train"], batch_size=self.batch_size,
                                            num_workers=self.num_workers, shuffle=True, persistent_workers=True, pin_memory=True)
        return self._train_dataloader
>>>>>>> Stashed changes

    def _val_dataloader(self):
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def _test_dataloader(self):
        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers)


class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_pretrain_routine_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # import pdb
            # pdb.set_trace()
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            print("Project config")
            print(self.config.pretty())
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            print("Lightning config")
            print(self.lightning_config.pretty())
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))


class ImageLogger(Callback):
    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True):
        super().__init__()
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.logger_log_images = {
            pl.loggers.WandbLogger: self._wandb,
        }
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp

    @rank_zero_only
    def _wandb(self, pl_module, images, batch_idx, split):
        grids = dict()
        for k in images:
<<<<<<< Updated upstream
            grid = torchvision.utils.make_grid(images[k])
=======
            grid = torchvision.utils.make_grid(images[k], nrow=4)
>>>>>>> Stashed changes
            grids[f"{split}/{k}"] = wandb.Image(grid, )
        pl_module.logger.experiment.log(grids)

    @rank_zero_only
    def log_local(self, save_dir, split, images,
                  global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)

            grid = (grid+1.0)/2.0 # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0,1).transpose(1,2).squeeze(-1)
            grid = grid.float().numpy()
            grid = (grid*255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                k,
                global_step,
                current_epoch,
                batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        if (self.check_frequency(batch_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, pl_module.global_step, split)

            if is_train:
                pl_module.train()

    def check_frequency(self, batch_idx):
        if (batch_idx % self.batch_freq) == 0 or (batch_idx in self.log_steps):
            try:
                self.log_steps.pop(0)
            except IndexError:
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.log_img(pl_module, batch, batch_idx, split="val")

<<<<<<< Updated upstream
=======
class ProgressBar(TQDMProgressBar):
    def init_train_tqdm(self):
        """Override this to customize the tqdm bar for training."""
        return Tqdm(
            desc=self.train_description,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            smoothing=1,
            bar_format=self.BAR_FORMAT,
        )

>>>>>>> Stashed changes
if __name__ == "__main__":
    # custom parser to specify config files, train, test and debug mode,
    # postfix, resume.
    # `--key value` arguments are interpreted as arguments to the trainer.
    # `nested.key=value` arguments are interpreted as config parameters.
    # configs are merged from left-to-right followed by command line parameters.

    # model:
    #   base_learning_rate: float
    #   target: path to lightning module
    #   params:
    #       key: value
    # data:
    #   target: main.DataModuleFromConfig
    #   params:
    #      batch_size: int
    #      wrap: bool
    #      train:
    #          target: path to train dataset
    #          params:
    #              key: value
    #      validation:
    #          target: path to validation dataset
    #          params:
    #              key: value
    #      test:
    #          target: path to test dataset
    #          params:
    #              key: value
    # lightning: (optional, has sane defaults and can be specified on cmdline)
    #   trainer:
    #       additional arguments to trainer
    #   logger:
    #       logger to instantiate
    #   modelcheckpoint:
    #       modelcheckpoint to instantiate
    #   callbacks:
    #       callback1:
    #           target: importpath
    #           params:
    #               key: value
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`
    # (in particular `main.DataModuleFromConfig`)
    sys.path.append(os.getcwd())

    parser = get_parser()

    opt, unknown = parser.parse_known_args()
    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            idx = len(paths)-paths[::-1].index("logs")+1
            logdir = "/".join(paths[:idx])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs+opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[_tmp.index("logs")+1]+opt.postfix
        logdir = os.path.join(opt.root_path, "logs", nowname)
    else:
        if opt.name:
            name = "_"+opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_"+cfg_name
        else:
            name = ""
        nowname = now+name+opt.postfix
        logdir = os.path.join(opt.root_path, "logs", nowname)

    if opt.random_seed:
        opt.seed = random.randint(1,100)
    logdir = logdir + '_seed' + str(opt.seed)
    rank_zero_only(os.makedirs)(logdir)
    
    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")

    seed_everything(opt.seed)

    try:
        # init and save configs
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)
        whole_config = deepcopy(config)
        lightning_config = config.pop("lightning", OmegaConf.create())
        # merge trainer cli with config
        trainer_config = lightning_config.get("trainer", OmegaConf.create())
        
        for k in nondefault_trainer_args(opt):
            trainer_config[k] = getattr(opt, k)
<<<<<<< Updated upstream
        if not "devices" in trainer_config:
            #del trainer_config["distributed_backend"]
            cpu = True
        else:
            gpuinfo = trainer_config["devices"]
=======
        if not "devices" in trainer_config and os.environ['CUDA_VISIBLE_DEVICES'] == '':
            #del trainer_config["distributed_backend"]
            cpu = True
        else:
            gpuinfo = trainer_config["devices"] if 'devices' in trainer_config else os.environ['CUDA_VISIBLE_DEVICES']
>>>>>>> Stashed changes
            print(f"Running on GPUs {gpuinfo}")
            cpu = False
        lightning_config.trainer = trainer_config

        # model
        model = instantiate_from_config(config.model)

        # trainer and callbacks
        trainer_kwargs = dict()
        trainer_kwargs['sync_batchnorm'] = False
        
        # default logger configs
        default_logger_cfgs = {
            "wandb": {
                "target": "pytorch_lightning.loggers.WandbLogger",
                "params": {
                    "name": nowname,
                    "save_dir": logdir,
                    "offline": opt.debug,
                    "id": nowname,
<<<<<<< Updated upstream
                    "config": OmegaConf.to_container(whole_config),
=======
>>>>>>> Stashed changes
                }
            },
        }
        default_logger_cfg = default_logger_cfgs["wandb"]
        try:
            logger_cfg = lightning_config.logger
        except:
            logger_cfg = OmegaConf.create()
        logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
        logger = None if logger_cfg == {} else instantiate_from_config(logger_cfg)

        if isinstance(logger, pl.loggers.WandbLogger):
            if hasattr(logger.experiment.config, 'update'):
                logger.experiment.config.update(OmegaConf.to_container(whole_config))

        trainer_kwargs["logger"] = logger

        # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
        # specify which metric is used to determine best models
        default_modelckpt_cfg = {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch:06}",
                "verbose": True,
                "save_last": True,
<<<<<<< Updated upstream
=======
                "monitor": "train/rec_loss_epoch",
                "save_top_k": 1,
>>>>>>> Stashed changes
                "every_n_epochs": 1
            }
        }
        if hasattr(model, "monitor"):
            print(f"Monitoring {model.monitor} as checkpoint metric.")
            default_modelckpt_cfg["params"]["monitor"] = model.monitor
            default_modelckpt_cfg["params"]["save_top_k"] = 3

        try:
            modelckpt_cfg = lightning_config.modelcheckpoint
        except:
            modelckpt_cfg = OmegaConf.create()
        modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
<<<<<<< Updated upstream
        #trainer_kwargs["checkpoint_callback"] = instantiate_from_config(modelckpt_cfg)
=======
>>>>>>> Stashed changes
        trainer_kwargs["callbacks"] = [instantiate_from_config(modelckpt_cfg)]
        trainer_kwargs["strategy"] = 'ddp_find_unused_parameters_true'
        trainer_kwargs["default_root_dir"] = logdir

<<<<<<< Updated upstream
        try:
            profiler_config = OmegaConf.to_container(lightning_config.get('profiler', OmegaConf.create()))
            if 'params' in profiler_config and 'schedule' in profiler_config['params']:
                profiler_config['params']['schedule'] = instantiate_from_config(profiler_config['params']['schedule'])
            trainer_kwargs["profiler"] = instantiate_from_config(profiler_config)
        except:
            pass
=======
        if opt.enable_profiler:
            assert opt.debug, "Error: Debug mode must be on for profiler to be enabled."
            try:
                profiler_config = lightning_config.get('profiler')
                if isinstance(profiler_config, str):
                    trainer_kwargs["profiler"] = profiler_config
                else:
                    profiler_config = OmegaConf.to_container(lightning_config.get('profiler', OmegaConf.create()))
                    if 'params' in profiler_config and 'schedule' in profiler_config['params']:
                        profiler_config['params']['schedule'] = instantiate_from_config(profiler_config['params']['schedule'])
                    trainer_kwargs["profiler"] = instantiate_from_config(profiler_config)
                print(trainer_kwargs["profiler"])
            except Exception as e:
                print('Exception when creating profiler:', e)
>>>>>>> Stashed changes

        # add callback which sets up log directory
        default_callbacks_cfg = {
            "setup_callback": {
                "target": "main.SetupCallback",
                "params": {
                    "resume": opt.resume,
                    "now": now,
                    "logdir": logdir,
                    "ckptdir": ckptdir,
                    "cfgdir": cfgdir,
                    "config": config,
                    "lightning_config": lightning_config,
                }
            },
            "image_logger": {
                "target": "main.ImageLogger",
                "params": {
                    "batch_frequency": 750,
                    "max_images": config.data.params.batch_size,
                    "clamp": True
                }
            },
            "learning_rate_logger": {
                "target": "main.LearningRateMonitor",
                "params": {
                    "logging_interval": "step",
                    #"log_momentum": True
                }
            },
            "progress_bar_callback": {
                "target": "main.ProgressBar"
            },
        }
        try:
            callbacks_cfg = lightning_config.callbacks
        except:
            callbacks_cfg = OmegaConf.create()
        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
        trainer_kwargs["callbacks"] += [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]
        trainer_kwargs |= trainer_config

        trainer = Trainer(**trainer_kwargs)

        # data
        data = instantiate_from_config(config.data)
        # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
        # calling these ourselves should not be necessary but it is.
        # lightning still takes care of proper multiprocessing though
        data.prepare_data()
        data.setup()

        # configure learning rate
        bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
        if not cpu:
<<<<<<< Updated upstream
            ngpu = len(lightning_config.trainer.devices.strip(",").split(','))
=======
            dev_str = os.environ['CUDA_VISIBLE_DEVICES']
            if hasattr(lightning_config.trainer, 'devices') and isinstance(lightning_config.trainer.devices, str):
                dev_str = lightning_config.trainer.devices
            print('devices:', dev_str)
            ngpu = len(dev_str.strip(",").split(','))
>>>>>>> Stashed changes
        else:
            ngpu = 1

        try:
            accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
        except:
            accumulate_grad_batches = 1
        print(f"accumulate_grad_batches = {accumulate_grad_batches}")
        lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
        model.learning_rate = accumulate_grad_batches * ngpu * bs * trainer.num_nodes * base_lr
        print("Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (num_nodes) * {} (batchsize) * {:.2e} (base_lr)".format(
            model.learning_rate, accumulate_grad_batches, ngpu, trainer.num_nodes, bs, base_lr))

        # allow checkpointing via USR1
        def melk(*args, **kwargs):
            # run all checkpoint hooks
            if trainer.global_rank == 0:
                print("Summoning checkpoint.")
                ckpt_path = os.path.join(ckptdir, "last.ckpt")
                trainer.save_checkpoint(ckpt_path)

        def divein(*args, **kwargs):
            if trainer.global_rank == 0:
                import pudb; pudb.set_trace()

        import signal
        signal.signal(signal.SIGUSR1, melk)
        signal.signal(signal.SIGUSR2, divein)

        # run
        if opt.train:
            try:
<<<<<<< Updated upstream
                #compiled_model = torch.compile(model)
                #trainer.fit(compiled_model, data)
=======
>>>>>>> Stashed changes
                try:
                    resume_ckpt = config.model.params.ckpt_path
                except:
                    resume_ckpt = None
                trainer.fit(model, data, ckpt_path=resume_ckpt)
            except Exception:
                melk()
                raise
        if not opt.no_test and not trainer.interrupted:
            trainer.test(model, data)
    except Exception:
        if opt.debug and trainer.global_rank==0:
            try:
                import pudb as debugger
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()
        raise
    finally:
        # move newly created debug project to debug_runs
        if opt.debug and not opt.resume and trainer.global_rank==0:
            dst, name = os.path.split(logdir)
            dst = os.path.join(dst, "debug_runs", name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            os.rename(logdir, dst)
