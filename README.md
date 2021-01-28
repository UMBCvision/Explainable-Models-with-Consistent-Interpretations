# Explainable Models with Consistent Interpretations
Official PyTorch implementation for the AAAI 2021 paper 'Explainable Models with Consistent Interpretations'

Given the widespread deployment of black box deep neural networks in computer vision applications, the interpretability aspect of these black box systems has recently gained traction. Various methods have been proposed to explain the results of such deep neural networks. However, some recent works have shown that such explanation methods are biased and do not produce consistent interpretations. Hence, rather than introducing a novel explanation method, we learn models that are encouraged to be interpretable given an explanation method. We use Grad-CAM as the explanation algorithm and encourage the network to learn consistent interpretations along with maximizing the log-likelihood of the correct class. We show that our method outperforms the baseline on the pointing game evaluation on ImageNet and MS-COCO datasets respectively. We also introduce new evaluation metrics that penalize the saliency map if it lies outside the ground truth bounding box or segmentation mask, and show that our method outperforms the baseline on these metrics as well. Moreover, our model trained with interpretation consistency generalizes to other explanation algorithms on all the evaluation metrics.

## Pre-requisites
Pytorch 1.3 - Please install PyTorch (https://pytorch.org/get-started/locally/) and CUDA if you don't have it installed.

## Training
Following code can be used to train a ResNet 18 model on the ImageNet dataset
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_gcam_grid_consistency.py <path_to_imagenet_dataset> -a resnet18 -b 256 -j 16 --lambda 25 --save_dir <path_to_checkpoint_dir> 

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
<li>Pointing Game:
CUDA_VISIBLE_DEVICES=0 python evaluate_imagenet_gradcam_pointinggame.py <path_to_imagenet_validation_root> -j 0 -b 1 --resume <path_to_model_checkpoint> --input_resize 448
</li> 
<li>Stochastic Pointing Game:
CUDA_VISIBLE_DEVICES=0 python evaluate_imagenet_gradcam_stochastic_pointinggame.py <path_to_imagenet_validation_root> -j 0 -b 1 --resume <path_to_model_checkpoint> --input_resize 448
</li> 
<li>Content Heatmap:
CUDA_VISIBLE_DEVICES=0 python evaluate_imagenet_gradcam_energy_inside_bbox.py <path_to_imagenet_validation_root> -j 0 -b 1 --resume <path_to_model_checkpoint> --input_resize 448
</li> 
</ol>



## Acknowledgement
We would like to thank Ashley Rothballer and Dennis Fong for helpful disucssions regarding this work.
This material is based upon work partially supported by the United States Air Force under Contract No. FA8750-19-C-0098, funding from NSF grant number 1845216, SAP SE, and Northrop Grumman. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the United States Air Force, DARPA, or other funding agencies.
