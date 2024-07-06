# G-Swin Transformer

This is an official implementation for "Vision Transformer Based Multi-class Lesion Detection in IVOCT". This repo contains the supported code and configuration files to reproduce object detection results of [Swin Transformer](https://arxiv.org/pdf/2103.14030.pdf). It is based on [mmdetection](https://github.com/open-mmlab/mmdetection).

## Usage

### Installation

0. Python version == 3.8.19
1. CUDA version == 11.3
2. pytorch version == 1.10.1
3. pip install mmcv-full=={1.4.0} -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
4. pip install -r requirements/build.txt
4.5. pip install cython==0.29.33
5. python setup.py develop
5.5. pip install yapf==0.40.1
6. (optional) git clone https://github.com/NVIDIA/apex
6.5 pip install -v --disable-pip-version-check --no-build-isolation --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

If you find other errors, please refer to docs/get_started.md and [get_started.md](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md) for installation and dataset preparation.

If you find any error about geometric.py or photometric.py in mmcv-full library, replace the content using G-Swin-Transformer/geometric.py or G-Swin-Transformer/photometric.py.

### Inference
```
# single-gpu testing
python tools/test.py <CONFIG_FILE> <DET_CHECKPOINT_FILE> --eval bbox segm

# multi-gpu testing
tools/dist_test.sh <CONFIG_FILE> <DET_CHECKPOINT_FILE> <GPU_NUM> --eval bbox segm
```

### Training

To train a detector with pre-trained models, run:
```
# single-gpu training
python tools/train.py <CONFIG_FILE> --cfg-options model.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments]

# multi-gpu training
tools/dist_train.sh <CONFIG_FILE> <GPU_NUM> --cfg-options model.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments] 
```
For example, to train a Cascade Mask R-CNN model with a `Swin-T` backbone and 8 gpus, run:
```
tools/dist_train.sh configs/swin/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py 8 --cfg-options model.pretrained=<PRETRAIN_MODEL> 
```

**Note:** `use_checkpoint` is used to save GPU memory. Please refer to [this page](https://pytorch.org/docs/stable/checkpoint.html) for more details.


### Apex (optional):
We use apex for mixed precision training by default. To install apex, run:
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
If you would like to disable apex, modify the type of runner as `EpochBasedRunner` and comment out the following code block in the [configuration files](configs/swin):
```
# do not use mmdet version fp16
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)
```

## Citing Swin Transformer
```
@article{liu2021Swin,
  title={Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  journal={arXiv preprint arXiv:2103.14030},
  year={2021}
}
```
## Citing G-Swin Transformer
```
@inproceedings{wang2023vision,
  title={Vision Transformer Based Multi-class Lesion Detection in IVOCT},
  author={Wang, Zixuan and Shao, Yifan and Sun, Jingyi and Huang, Zhili and Wang, Su and Li, Qiyong and Li, Jinsong and Yu, Qian},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={327--336},
  year={2023},
  organization={Springer}
}
```