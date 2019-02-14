import argparse
import torch
from torch.utils.data.sampler import WeightedRandomSampler

from sal.utils.pytorch_fixes import *
from sal.utils.pytorch_trainer import *
from sal.saliency_model import SaliencyModel, SaliencyLoss
from sal.utils.resnet_encoder_isic import resnet50encoder

from sal.datasets.isic_dataset import ISIC_RESNET50_CKPT_PATH
import sal.datasets.isic_dataset as isic_dataset



# ---- config ----
# You can choose your own dataset and a black box classifier as long as they are compatible with the ones below.
# The training code does not need to be changed and the default values should work well for high resolution ~300x300 real-world images.
# By default we train on 224x224 resolution ImageNet images with a resnet50 black box classifier.
dts = isic_dataset
# black_box_fn = get_black_box_fn(model_zoo_model=densenet169)
# ----------------


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def imagenet_normalize(t, mean=None, std=None):
    if mean is None:
        mean = IMAGENET_MEAN
    if std is None:
        std= IMAGENET_STD

    ts = []
    for i in range(3):
        ts.append(torch.unsqueeze((t[:, i] - mean[i]) / std[i], 1))
    return torch.cat(ts, dim=1)


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad_(False)


def black_box_resnet_isic(cuda=True, ckpt_path=None):
    black_box_model = resnet50encoder()

    if ckpt_path is not None:
        black_box_model.load_state_dict(
            (torch.load(ckpt_path, map_location=lambda storage, loc: storage)['state_dict'])
        )
    if cuda:
        black_box_model = torch.nn.DataParallel(black_box_model).cuda()
    black_box_model.train(False)
    freeze_model(black_box_model)

    def black_box_fn(_images):
        return black_box_model((1 + _images) * 0.5)[-1]
    return black_box_fn


train_dts = dts.get_train_dataset()
val_dts = dts.get_val_dataset()

encoder = resnet50encoder(7, pretrained=True)
# encoder.load_state_dict(torch.load(ISIC_RESNET50_CKPT_PATH)['state_dict'])
# for param in encoder.parameters():
#    param.requires_grad = False

# Default saliency model with pretrained resnet50 feature extractor, produces saliency maps which have resolution 4 times lower than the input image.
saliency = SaliencyModel(encoder, 5, 64, 3, 64, fix_encoder=False, use_simple_activation=False, allow_selector=True, num_classes=7)

saliency_p = saliency.cuda()
saliency_loss_calc = SaliencyLoss(black_box_resnet_isic(ckpt_path=ISIC_RESNET50_CKPT_PATH),
                                  smoothness_loss_coef=0.005, num_classes=7) # model based saliency requires very small smoothness loss and therefore can produce very sharp masks
optim_phase1 = torch_optim.Adam(saliency.selector_module.parameters(), 0.001, weight_decay=0.0001)
optim_phase2 = torch_optim.Adam(saliency.get_trainable_parameters(), 0.001, weight_decay=0.0001)


def set_trainable_phase1(is_training):
    saliency.train(is_training)


def set_trainable_phase2(is_training):
    saliency.train(is_training)
    encoder.train(False)


def make_weights_for_balanced_classes(labels, nclasses):
    count = [0] * nclasses
    for label in labels:
        count[label] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(labels)
    for idx, val in enumerate(labels):
        weight[idx] = weight_per_class[val[1]]
    return weight


@TrainStepEvent()
@EveryNthEvent(4000)
def lr_step_phase1(s):
    print()
    print(GREEN_STR % 'Reducing lr by a factor of 10')
    for param_group in optim_phase1.param_groups:
        param_group['lr'] = param_group['lr'] / 10.


@ev_batch_to_images_labels
def ev_phase1(_images, _labels):
    __fakes = Variable(torch.Tensor(_images.size(0)).uniform_(0, 1).cuda()<FAKE_PROB)
    _targets = (_labels + Variable(torch.Tensor(_images.size(0)).uniform_(1, 6).cuda()).long()*__fakes.long()) % 7
    _is_real_label = PT(is_real_label=(_targets == _labels).long())
    _masks, _exists_logits, _ = saliency_p(imagenet_normalize(_images), _targets)
    PT(exists_logits=_exists_logits)
    exists_loss = F.cross_entropy(_exists_logits, _is_real_label)
    loss = PT(loss=exists_loss)


@ev_batch_to_images_labels
def ev_phase2(_images, _labels):
    __fakes = Variable(torch.Tensor(_images.size(0)).uniform_(0, 1).cuda()<FAKE_PROB)
    _targets = PT(targets=(_labels + Variable(torch.Tensor(_images.size(0)).uniform_(1, 6).cuda()).long()*__fakes.long())
                           % 7)
    _is_real_label = PT(is_real_label=(_targets == _labels).long())
    _masks, _exists_logits, _ = saliency_p(imagenet_normalize(_images), _targets)
    PT(exists_logits=_exists_logits)
    saliency_loss = saliency_loss_calc.get_loss(_images, _labels, _masks, _is_real_target=_is_real_label,  pt_store=PT)
    loss = PT(loss=saliency_loss)


FAKE_PROB = -1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=[1, 2], type=int)
    parser.add_argument('save_dir')
    parser.add_argument('--load-model', dest='load_model')
    config = parser.parse_args()
    print('config', config)

    if config.load_model is not None:
        saliency.minimialistic_restore(config.load_model)
    saliency_old = saliency
    weights = make_weights_for_balanced_classes(train_dts.labels, 7)
    weights = torch.DoubleTensor(weights)
    sampler = WeightedRandomSampler(weights, len(weights))

    if config.phase == 1:
        nt_phase1 = NiceTrainer(ev_phase1, dts.get_loader(train_dts, batch_size=96), optim_phase1,
                         val_dts=dts.get_loader(val_dts, batch_size=96, sampler=sampler),
                         modules=[saliency],
                         printable_vars=['loss', 'exists_accuracy'],
                         events=[lr_step_phase1,],
                         computed_variables={'exists_accuracy': accuracy_calc_op('exists_logits', 'is_real_label')},
                         set_trainable=set_trainable_phase1)
        FAKE_PROB = .5
        nt_phase1.train(8500)
    else:
        saliency.fix_encoder = True
        freeze_model(encoder)
        nt_phase2 = NiceTrainer(ev_phase2, dts.get_loader(train_dts, batch_size=48), optim_phase2,
                         val_dts=dts.get_loader(val_dts, batch_size=48, sampler=sampler),
                         modules=[saliency],
                         printable_vars=['loss', 'exists_accuracy'],
                         events=[],
                         computed_variables={'exists_accuracy': accuracy_calc_op('exists_logits', 'is_real_label')},
                         set_trainable=set_trainable_phase2)
        FAKE_PROB = .3
        nt_phase2.train(3000)

    saliency_old.cpu()
    saliency_old.minimalistic_save(config.save_dir)  # later to restore just use saliency.minimalistic_restore methdod.
