# Explainable Models with Consistent Interpretations
Official PyTorch implementation for the AAAI 2021 paper ['Explainable Models with Consistent Interpretations'](https://www.csee.umbc.edu/~hpirsiav/papers/gc_aaai21.pdf)

Given the widespread deployment of black box deep neural networks in computer vision applications, the interpretability aspect of these black box systems has recently gained traction. Various methods have been proposed to explain the results of such deep neural networks. However, some recent works have shown that such explanation methods are biased and do not produce consistent interpretations. Hence, rather than introducing a novel explanation method, we learn models that are encouraged to be interpretable given an explanation method. We use Grad-CAM as the explanation algorithm and encourage the network to learn consistent interpretations along with maximizing the log-likelihood of the correct class. We show that our method outperforms the baseline on the pointing game evaluation on ImageNet and MS-COCO datasets respectively. We also introduce new evaluation metrics that penalize the saliency map if it lies outside the ground truth bounding box or segmentation mask, and show that our method outperforms the baseline on these metrics as well. Moreover, our model trained with interpretation consistency generalizes to other explanation algorithms on all the evaluation metrics.

![To encourage interpretation consistency, we randomly sample three distractor images for a given input image and feed all four to the Image Gridding module which creates a 2 × 2 grid and places the positive image and the three negative images in random cells. We also feed the Grad-CAM interpretation mask for the ground truth category (‘Keyboard’) to the Image Gridding to obtain the ground truth Grad-CAM mask of this composite image for the positive image category. The negative quadrants of this mask are set to zero. We then penalize the network if the Grad-CAM heatmap of the composite image for the positive image category does not match the ground truth Grad-CAM mask.][teaser]

## Pre-requisites
- Pytorch 1.3 - Please install [PyTorch](https://pytorch.org/get-started/locally/) and CUDA if you don't have it installed. 
- [pycocotools](https://pypi.org/project/pycocotools/)

## Training
### ImageNet
Following code can be used to train a ResNet 18 model using our Grad-CAM consistency method - <br/>
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_gcam_grid_consistency.py <path_to_imagenet_dataset> -a resnet18 -b 256 -j 16 --lambda 25 --save_dir <path_to_checkpoint_dir> 

Following code can be used to train a ResNet 18 model with Global Max Pooling instead of Global Average Pooling along with our Grad-CAM consistency method - <br/>
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_gcam_grid_consistency.py <path_to_imagenet_dataset> -a resnet18 -b 256 -j 16 --lambda 25 --maxpool --save_dir <path_to_checkpoint_dir> 

### MS-COCO
Since MS-COCO dataset is a multi-class dataset, we randomly select one of the ground truth categories to compute the Grad-CAM heatmap for the original image and the composite image. Hence, we perform a pre-processing to extract a dictionary containing a list of negative images corresponding to each ground truth category. We used the script _extract_negative_image_list.py_ to create this dictionary and use it in the COCO dataloader to create the composite images.

Following code can be used to train a ResNet 18 model using our Grad-CAM consistency method - <br/>
CUDA_VISIBLE_DEVICES=0,1 python train_gcam_multiclass_grid_consistency.py <path_to_coco_dataset> -a resnet18 --num-gpus 2 --lr 0.01 -b 256 -j 16 --lambda 1 --resume <path_to_imagenet_pretrained_model_checkpoint> --save_dir <path_to_checkpoint_dir>

Following code can be used to train a ResNet 18 model with Global Max Pooling instead of Global Average Pooling along with our Grad-CAM consistency method - <br/>
CUDA_VISIBLE_DEVICES=0,1 python train_gcam_multiclass_grid_consistency.py <path_to_coco_dataset> -a resnet18 --num-gpus 2 --lr 0.01 -b 256 -j 16 --lambda 1 --maxpool --resume <path_to_imagenet_pretrained_model_checkpoint> --save_dir <path_to_checkpoint_dir>

## Evaluation
We use the evaluation code adapted from the [TorchRay](https://github.com/facebookresearch/TorchRay) framework. For the SPG metric introduced in our paper, we use a stochastic version of the pointing game metric to sample 100 points from the 2D map of the normalized Grad-CAM interpretation heatmap and evaluate using the bounding box annotation for ImageNet validation set.

* Change directory to TorchRay and install the library. Please refer to the [TorchRay](https://github.com/facebookresearch/TorchRay) repository for full documentation and instructions.
    * cd TorchRay
    * python setup.py install

* Change directory to TorchRay/torchray/benchmark
    * cd torchray/benchmark

For the ImageNet dataset, this evaluation requires the following structure for ImageNet validation images and bounding box xml annotations
<ul>
  <li>imagenet_root/val/*.JPEG - Flat list of 50000 validation images</li>
  <li>imagenet_root/val/*.xml - Flat list of 50000 annotation xml files</li>
</ul>

Evaluation metrics for Interpretation Consistency:
<ol>
<li>Pointing Game: <br/>
CUDA_VISIBLE_DEVICES=0 python evaluate_imagenet_gradcam_pointinggame.py <path_to_imagenet_validation_root> -j 0 -b 1 --resume <path_to_model_checkpoint> --input_resize 448
</li> 
<li>Stochastic Pointing Game: <br/>
CUDA_VISIBLE_DEVICES=0 python evaluate_imagenet_gradcam_stochastic_pointinggame.py <path_to_imagenet_validation_root> -j 0 -b 1 --resume <path_to_model_checkpoint> --input_resize 448
</li> 
<li>Content Heatmap: <br/>
CUDA_VISIBLE_DEVICES=0 python evaluate_imagenet_gradcam_energy_inside_bbox.py <path_to_imagenet_validation_root> -j 0 -b 1 --resume <path_to_model_checkpoint> --input_resize 448
</li> 
</ol>
   
## Results
### ImageNet

|          Architecture | Model Name    &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  |     Top-1 Acc (%)     |     Pointing Game    |     Stochastic Pointing Game      |     Content Heatmap    | Pre-trained  |
| --- | ------- | :---: | :---: | :---: | :---: | :---: |
|  AlexNet   |   Baseline     |          56.51             |     72.80      |      53.45        |     45.78      | [checkpoint](https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth)  |
|            |   Ours: GC    |          56.16             |     **73.70**      |      **61.15**        |     **48.10**      | [checkpoint](https://drive.google.com/file/d/1s9xAZ2p8wdyiVVqnokOACtJTOEvh_D35/view?usp=sharing)  |
|  ResNet18  |   Baseline     |          69.43             |     79.80      |      60.50        |     54.36      |  [checkpoint](https://drive.google.com/file/d/1jBPTU75iar5dSoqB3jlNMcAnOSY3H17n/view?usp=sharing) |
|            |   Ours: GC    |          67.74             |     **80.00**      |      65.85        |     57.73      |  [checkpoint](https://drive.google.com/file/d/1W52-sYbyPi-VFLRNlIWAwx4MExWiqMS_/view?usp=sharing) |
|            |   GMP          |          69.08             |     79.30      |      66.66        |     62.89      | [checkpoint](https://drive.google.com/file/d/1ivN3kqDyZ6ekY6jVI0RMAYu_QoiDxCvT/view?usp=sharing)  |
|            |   Ours: GMP + GC    |          69.02             |     79.60      |      **68.74**        |     **65.35**      | [checkpoint](https://drive.google.com/file/d/1NcJ5U8HIHKzReUpcDRdyMyrRmNXZcJ8J/view?usp=sharing)  |
|  ResNet50  |   Baseline     |          76.13             |     80.00      |      60.95        |     54.78      |  [checkpoint](https://download.pytorch.org/models/resnet50-19c8e357.pth) |
|            |   Ours: GC    |          74.40             |     **80.30**      |      65.26        |     59.42      | [checkpoint](https://drive.google.com/file/d/1eGQke4BApVdObrq0NdrhMayUzihdcTvx/view?usp=sharing)  |
|            |   GMP    |          74.63             |     79.80      |      66.29        |     54.23      | [checkpoint](https://drive.google.com/file/d/1bjaEHU-NKrUnsf5QTHpIVeknRbrwkD32/view?usp=sharing)  |
|            |   Ours: GMP + GC    |          74.14             |     79.60      |      **69.51**        |     **59.70**      | [checkpoint](https://drive.google.com/file/d/1yuvwGn6Em_lYWwjTXXfmcGAGV23sZ0D0/view?usp=sharing)  |

### MS-COCO
|          Architecture | Model Name    &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  |     F1-PerClass (%)   |     F1-Overall (%)    |     Pointing Game    |     Stochastic Pointing Game      |     Content Heatmap    | Pre-trained  |
| --- | ------- | :---: | :---: | :---: | :---: | :---: | :---: |
|  ResNet18  |   Baseline     |          69.43     |          69.43             |     79.80      |      60.50        |     54.36      |  [checkpoint](https://drive.google.com/file/d/1jBPTU75iar5dSoqB3jlNMcAnOSY3H17n/view?usp=sharing) |
|            |   Ours: GC    |          67.74      |          69.43             |     **80.00**      |      65.85        |     57.73      |  [checkpoint](https://drive.google.com/file/d/1W52-sYbyPi-VFLRNlIWAwx4MExWiqMS_/view?usp=sharing) |
|            |   GMP          |          69.08     |          69.43             |     79.30      |      66.66        |     62.89      | [checkpoint](https://drive.google.com/file/d/1rKP2oi1K83VP9w9jj_wgYw3psU_T5CMY/view?usp=sharing)  |
|            |   Ours: GMP + GC    |          69.02      |          69.43             |     79.60      |      **68.74**        |     **65.35**      | [checkpoint](https://drive.google.com/file/d/1plynvH8rKqLi37YaTl1I3HfLCw95kbZf/view?usp=sharing)  |


## Acknowledgement
We would like to thank Ashley Rothballer and Dennis Fong for helpful disucssions regarding this work.
This material is based upon work partially supported by the United States Air Force under Contract No. FA8750-19-C-0098, funding from NSF grant number 1845216, SAP SE, and Northrop Grumman. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the United States Air Force, DARPA, or other funding agencies.

## License
This project is licensed under the MIT license.

[teaser]: https://github.com/UMBCvision/Explainable-Models-with-Consistent-Interpretations/blob/main/misc/github_teaser.png
