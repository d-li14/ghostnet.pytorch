# PyTorch Implementation of GhostNet
Reproduction of GhostNet architecture as described in [GhostNet: More Features from Cheap Operations](https://arxiv.org/abs/1911.11907) by Kai Han, Yunhe Wang, Qi Tian, Jianyuan Guo, Chunjing Xu, Chang Xu on ILSVRC2012 benchmark with [PyTorch](pytorch.org) framework.

# Pretrained Models
| Architecture      | # Parameters | MFLOPs | Top-1 / Top-5 Accuracy (%) |
| ----------------- | ------------ | ------ | -------------------------- |
| [GhostNet 1.0x](https://github.com/d-li14/ghostnet.pytorch/blob/master/pretrained/ghostnet_1x-9c40f966.pth) | 5.181M | 140.77 | 72.318 / 90.670 |

```python
from ghostnet import ghostnet

net = ghostnet()
net.load_state_dict(torch.load('pretrained/ghostnet_1x-9c40f966.pth'))
```

# Training Strategy
We strictly follow the optimization scheme as stated in the original paper for reproduction.
* *batch size* 1024 on 8 GPUs
* *epoch* 240
* *Initial learning rate* 0.4
* *LR annealing* linear
* *weight decay* 0.00004
* **no** weight decay on BN
* **no** warmup or label smoothing

# Citation
```
@inproceedings{Han_2020_CVPR,
  title={GhostNet: More Features from Cheap Operations},
  author={Han, Kai and Wang, Yunhe and Tian, Qi and Guo, Jianyuan and Xu, Chunjing and Xu, Chang},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year={2020}
}
```
