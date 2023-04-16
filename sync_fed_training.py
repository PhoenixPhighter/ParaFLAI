USE_CUDA = True
LOCAL_BATCH_SIZE = 32
EXAMPLES_PER_USER = 500
IMAGE_SIZE = 32

EPOCHS = 1

# suppress large outputs
VERBOSE = False
    
import time

from torchvision.datasets.cifar import CIFAR10
from torchvision import transforms
from flsim.data.data_sharder import SequentialSharder
from flsim.utils.example_utils import DataLoader, DataProvider
import torch
from flsim.utils.example_utils import SimpleConvNet
from flsim.utils.example_utils import FLModel
from flsim.interfaces.metrics_reporter import Channel
from flsim.utils.example_utils import MetricsReporter
import flsim.configs
from flsim.utils.config_utils import fl_config_from_json
from omegaconf import OmegaConf
from hydra.utils import instantiate

start = time.time()

# 1. Create training, eval, and test datasets like in non-federated learning.
transform = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465), 
            (0.2023, 0.1994, 0.2010)
        ),
    ]
)
train_dataset = CIFAR10(
    root="./cifar10", train=True, download=True, transform=transform
)
test_dataset = CIFAR10(
    root="./cifar10", train=False, download=True, transform=transform
)



# 2. Create a sharder, which maps samples in the training data to clients.
sharder = SequentialSharder(examples_per_shard=EXAMPLES_PER_USER)

# 3. Shard and batchify training, eval, and test data.
fl_data_loader = DataLoader(
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    test_dataset=test_dataset,
    sharder=sharder,
    batch_size=LOCAL_BATCH_SIZE,
    drop_last=False,
)

# 4. Wrap the data loader with a data provider.
data_provider = DataProvider(fl_data_loader)
print(f"\nClients in total: {data_provider.num_train_users()}")



# 1. Define our model, a simple CNN.
model = SimpleConvNet(in_channels=3, num_classes=10)

# 2. Choose where the model will be allocated.
cuda_enabled = torch.cuda.is_available() and USE_CUDA
device = torch.device(f"cuda:{0}" if cuda_enabled else "cpu")

model, device



# 3. Wrap the model with FLModel.
global_model = FLModel(model, device)
assert(global_model.fl_get_module() == model)

# 4. Move the model to GPU and enable CUDA if desired.
if cuda_enabled:
    global_model.fl_cuda()



# Create a metric reporter.
metrics_reporter = MetricsReporter([Channel.TENSORBOARD, Channel.STDOUT])



json_config = {
    "trainer": {
        "_base_": "base_sync_trainer",
        # there are different types of aggregator
        # fed avg doesn't require lr, while others such as fed_avg_with_lr or fed_adam do
        "_base_": "base_sync_trainer",
        "server": {
            "_base_": "base_sync_server",
            "server_optimizer": {
                "_base_": "base_fed_avg_with_lr",
                "lr": 2.13,
                "momentum": 0.9
            },
            # type of user selection sampling
            "active_user_selector": {"_base_": "base_uniformly_random_active_user_selector"},
        },
        "client": {
            # number of client's local epoch
            "epochs": EPOCHS,
            "optimizer": {
                "_base_": "base_optimizer_sgd",
                # client's local learning rate
                "lr": 0.01,
                # client's local momentum
                "momentum": 0,
            },
        },
        # number of users per round for aggregation
        "users_per_round": 5,
        # total number of global epochs
        # total #rounds = ceil(total_users / users_per_round) * epochs
        "epochs": EPOCHS,
        # frequency of reporting train metrics
        "train_metrics_reported_per_epoch": 100,
        # frequency of evaluation per epoch
        "eval_epoch_frequency": 1,
        "do_eval": True,
        # should we report train metrics after global aggregation
        "report_train_metrics_after_aggregation": True,
    }
}
cfg = fl_config_from_json(json_config)
if VERBOSE: print(OmegaConf.to_yaml(cfg))

# Instantiate the trainer.
trainer = instantiate(cfg.trainer, model=global_model, cuda_enabled=cuda_enabled)   

# Launch FL training.
final_model, eval_score = trainer.train(
    data_provider=data_provider,
    metrics_reporter=metrics_reporter,
    num_total_users=data_provider.num_train_users(),
    distributed_world_size=1
)

trainer.test(
    data_provider=data_provider,
    metrics_reporter=MetricsReporter([Channel.STDOUT]),
)

end = time.time()
print(f"Total time: {end - start}")