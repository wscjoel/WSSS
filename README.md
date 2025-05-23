
---

# Knowledge Transfer Based Weakly Supervised Image Semantic Segmentation

This repository contains a reimplementation and extension of the [L2G: Local-to-Global Knowledge Transfer Framework](https://arxiv.org/abs/2204.03206) for Weakly Supervised Semantic Segmentation (WSSS), adapted and validated on both the **PASCAL VOC 2012** natural image dataset and the **MICCAI 2017 ACDC** cardiac MRI dataset for medical image segmentation.

This implementation is based on the original [L2G codebase](https://github.com/PengtaoJiang/L2G) and is extended with additional preprocessing, shape transfer strategies, and post-processing pipelines to support weakly supervised segmentation in medical domains.

## Abstract：

Weakly  supervised semantic segmentation (WSSS) amounts to segment pixels of images to their semantic labels based on weak labels (e.g., image classification label, bounding box of objects).Presently, the practice of extracting accurate class activation maps (CAM) to pinpoint objects in each category and using saliency maps for choosing background areas is extensively embracedin computervision. However,onthedomain of WSSS, it isobserved that CAM generated by some methods often suffer from insufficient activation in certain regions or blurry boundaries. In this thesis, we employ an uncomplicated web-based system for transferring knowledge from local to global levels to create superior attention maps. By substituting the original image for localized views, the classification network has the capability to identify areas of objects with greater detail. With this in mind, we first extract attention from local views arbitrary trimming of the input pictureusing a local classification network. Subsequently, utilizing a globalnetwork, we acquire supplementary knowledge about attention from various local maps online. The derived superior attention is directly applicable for pseudo-segmentation label in laternetworks.Experimental results demonstrate that our method achieves outstanding performance with mIoU scores of 71.2% on PASCAL VOC 2012 and DSC scores of 88.11% on MICCAI 2017 ACDC, respectively, securing excellent rankings.

## Highlights

* Achieved **71.2% mIoU on VOC 2012** validation set (with saliency).
* Achieved **88.11% DSC on ACDC** dataset with significantly improved myocardium segmentation.
* Enhanced shape transfer via **Pixel Correlation Module (PCM)** and **saliency supervision**.
* Unified framework for both natural and medical image segmentation.

## Dataset Preparation

### PASCAL VOC 2012

Download VOC2012 from [Baidu Drive 提取码: cl1e](https://pan.baidu.com/s/1CCR840MJ3Rx7jQ-r1jLX9g) and place it under:

```
data/voc12/
├── JPEGImages
├── SegmentationClass
├── Sal (precomputed PoolNet saliency maps)
```

### MICCAI 2017 ACDC

Register and download the ACDC dataset from the [official challenge site](https://acdc.creatis.insa-lyon.fr/description/databases.html).

Place `.nii.gz` files under:

```
data/acdc/
├── imagesTr/
├── labelsTr/
```

Our preprocessing pipeline resamples, crops to 128×128, and standardizes intensity.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Training & Evaluation

### For VOC2012:

```bash
# Train local-to-global classification
bash scripts/train_l2g_sal_voc.sh

# Generate pseudo labels
bash scripts/gen_gt_voc.sh

# Train segmentation
cd deeplab-pytorch
python main.py train --config-path configs/voc12_resnet_dplv2.yaml
```

### For ACDC:

```bash
# Preprocess & patch extraction
python acdc_preprocess.py

# Train classification + attention transfer
bash scripts/train_l2g_acdc.sh

# Generate pseudo masks and shape-enhanced labels
bash scripts/gen_gt_acdc.sh

# Train segmentation using DeepLab-v2
cd deeplab-pytorch
python main.py train --config-path configs/acdc_resnet_dplv2.yaml
```

## Results

| Dataset        | Metric | Score  |
| -------------- | ------ | ------ |
| VOC 2012       | mIoU   | 71.2%  |
| ACDC (overall) | DSC    | 88.11% |
| ACDC - Myo     | DSC    | 84.28% |
| ACDC - LV      | DSC    | 93.14% |

## Visualization

Qualitative comparisons of segmentation results against CAM baseline and expert annotations are provided in `results/visualizations/`.



