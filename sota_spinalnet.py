import sys

sys.path.append(".")

import mindspore
import mindspore.nn as nn
import mindspore.dataset.vision as vision
from mindspore.dataset import transforms
from mindspore import dtype as mstype
from mindspore.train import LossMonitor, TimeMonitor, Model, CheckpointConfig, ModelCheckpoint
from mindcv import create_dataset
from mindcv.models import create_model


cifar100_train = create_dataset("cifar100", "./", "train", False, download=True)
cifar100_test = create_dataset("cifar100", "./", "test", False, download=True)
columns_to_project = ["image", "fine_label"]
cifar100_train = cifar100_train.project(columns_to_project)
cifar100_test = cifar100_test.project(columns_to_project)
data_transforms = {
    'train': transforms.Compose([
        vision.Resize((272, 272)),
        vision.RandomRotation(15,),
        vision.RandomCrop(256),
        vision.RandomHorizontalFlip(prob=0.5),
        vision.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
        vision.HWC2CHW()
    ]),
    'test': transforms.Compose([
        vision.Resize(256),
        vision.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
        vision.HWC2CHW()
    ])
}

target_trans = transforms.TypeCast(mstype.int32)
cifar100_train = cifar100_train.map(operations=data_transforms['train'], input_columns='image', num_parallel_workers=4)
cifar100_train = cifar100_train.map(operations=target_trans, input_columns='fine_label', num_parallel_workers=4)
cifar100_train = cifar100_train.batch(64)


cifar100_test = cifar100_test.map(operations=data_transforms['test'], input_columns='image', num_parallel_workers=4)
cifar100_test = cifar100_test.map(operations=target_trans, input_columns='fine_label', num_parallel_workers=4)
cifar100_test = cifar100_test.project(columns_to_project)
cifar100_test = cifar100_test.batch(64)


resnet101 = create_model('resnet101', 100, True, 3)
num_ftrs = resnet101.classifier.in_channels

half_in_size = round(num_ftrs / 2)
layer_width = 20 #Small for Resnet, large for VGG
num_class=100
class SpinalNet_ResNet(nn.Cell):
    def __init__(self):
        super(SpinalNet_ResNet, self).__init__()
        
        self.fc_spinal_layer1 = nn.SequentialCell(
            nn.Dense(half_in_size, layer_width),
            nn.ReLU(),)
        self.fc_spinal_layer2 = nn.SequentialCell(
            nn.Dense(half_in_size + layer_width, layer_width),
            nn.ReLU(),)
        self.fc_spinal_layer3 = nn.SequentialCell(
            nn.Dense(half_in_size + layer_width, layer_width),
            nn.ReLU(),)
        self.fc_spinal_layer4 = nn.SequentialCell(
            nn.Dense(half_in_size + layer_width, layer_width),
            nn.ReLU(),)
        self.fc_out = nn.SequentialCell(
            nn.Dense(layer_width * 4, num_class),)
        
    def construct(self, x):
        x1 = self.fc_spinal_layer1(x[:, 0:half_in_size])
        x2 = self.fc_spinal_layer2(mindspore.ops.cat([ x[:,half_in_size:2*half_in_size], x1], axis=1))
        x3 = self.fc_spinal_layer3(mindspore.ops.cat([ x[:,0:half_in_size], x2], axis=1))
        x4 = self.fc_spinal_layer4(mindspore.ops.cat([ x[:,half_in_size:2*half_in_size], x3], axis=1))

        x = mindspore.ops.cat([x1, x2], axis=1)
        x = mindspore.ops.cat([x, x3], axis=1)
        x = mindspore.ops.cat([x, x4], axis=1)
        x = self.fc_out(x)
        return x


resnet101.classifier = SpinalNet_ResNet()
loss_fn = nn.CrossEntropyLoss()
optimizer_ft = nn.SGD(params=resnet101.trainable_params(), learning_rate=0.01, momentum=0.9)
model = Model(resnet101, loss_fn, optimizer_ft, {"accuracy"})

loss_monitor = LossMonitor(100)
time_monitor = TimeMonitor(100)
ckpt_cfg = CheckpointConfig(save_checkpoint_steps=100, append_info=['epoch_num'])
ckpt_monitor = ModelCheckpoint(prefix='cifar100_spinalnet', directory='./checkpoint', config=ckpt_cfg)
model.fit(20, cifar100_train, cifar100_test, callbacks=[loss_monitor, time_monitor, ckpt_monitor])
