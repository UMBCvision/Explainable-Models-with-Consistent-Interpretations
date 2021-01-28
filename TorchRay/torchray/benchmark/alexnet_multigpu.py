import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
import numpy as np
import pdb

__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        self.relu = nn.ReLU(inplace=True)

    def _forward(self, images, composite_images=None, targets=None, gt_quadrants=None,
                 xe_criterion=None, l1_criterion=None, eval=False, vanilla=False, return_feats=False):
        """
        This function serves as the wrapper which computes both losses and returns the loss values
        to work around the DataParallel limitation for splitting gradients/feats on each GPU and
        accumulating them later.
        :param images:
        :param composite_images:
        :param targets:
        :param gt_quadrants:
        :param xe_criterion:
        :param l1_criterion:
        :param eval: If True, call forward_vanilla
        :param vanilla: If True, call forward_vanilla
        :param return_feats: return forward vanilla while also returning layer4 feats
        :return:
        """
        # we keep both flags to maintain backward compatibility with previous code
        if eval or vanilla:
            return self.forward_vanilla(images)
        if return_feats:
            return self.forward_with_feats(images)

        # First, we need to sort the images by gt_quadrant indices to create batch independent sequential masks
        sorted_indices = torch.argsort(gt_quadrants)
        gt_quadrants = gt_quadrants[sorted_indices]
        images = images[sorted_indices]
        targets = targets[sorted_indices]
        composite_images = composite_images[sorted_indices]

        images_feats = self.features(images)
        x = self.avgpool(images_feats)
        x = torch.flatten(x, 1)
        images_outputs = self.classifier(x)

        composite_images_feats = self.features(composite_images)
        y = self.avgpool(composite_images_feats)
        y = torch.flatten(y, 1)
        composite_images_outputs = self.classifier(y)

        xe_loss = xe_criterion(images_outputs, targets)

        # compute gcam for images
        orig_gradcam_mask = self.compute_gradcam(images_outputs, images_feats, targets)
        orig_gradcam_mask = orig_gradcam_mask.unsqueeze(dim=1)
        orig_gradcam_mask = F.interpolate(orig_gradcam_mask,
                                          (images.shape[-2], images.shape[-1]),
                                          mode='bilinear', align_corners=False)
        orig_gradcam_mask = orig_gradcam_mask.squeeze(dim=1)
        feats_spatial_size = images.shape[-1]

        gt_quadrants_np = gt_quadrants.cpu().numpy()
        quad_index_start_dict = {}
        for quad_index in range(0, 4):
            quad_position = np.where(gt_quadrants_np == quad_index)[0]
            # Not all 4 quadrants might have been part of the random sampling. This is usually the case
            # if the last batch size in the epoch is much smaller
            if len(quad_position) > 0:
                quad_index_start_dict[quad_index] = quad_position[0]
        quad_index_start_dict['end'] = images.shape[0]   # append the batch size as the last index

        # create the GT mask for the gradcam
        gradcam_gt_mask = torch.zeros((images.shape[0], 2*feats_spatial_size, 2*feats_spatial_size)).cuda()

        # we will be indexing into quad_index_start_dict to obtain quadrant start indices
        for j in range(2):
            for k in range(2):
                index = j * 2 + k
                if index not in quad_index_start_dict:
                    continue
                next_index = self.get_next_index(quad_index_start_dict, index)
                gradcam_gt_mask[quad_index_start_dict[index]:quad_index_start_dict[next_index],
                j * feats_spatial_size:(j + 1) * feats_spatial_size,
                k * feats_spatial_size:(k + 1) * feats_spatial_size] = \
                    orig_gradcam_mask[quad_index_start_dict[index]:quad_index_start_dict[next_index], :, :]

        # compute gcam for composite_images
        composite_gradcam_mask = self.compute_gradcam(composite_images_outputs, composite_images_feats, targets)
        composite_gradcam_mask = composite_gradcam_mask.unsqueeze(dim=1)
        composite_gradcam_mask = F.interpolate(composite_gradcam_mask,
                                               (composite_images.shape[-2], composite_images.shape[-1]),
                                               mode='bilinear', align_corners=False)
        composite_gradcam_mask = composite_gradcam_mask.squeeze(dim=1)
        gcam_loss = l1_criterion(composite_gradcam_mask, gradcam_gt_mask)

        return images_outputs, targets, xe_loss, gcam_loss

    def _forward_vanilla(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _forward_with_feats(self, x):
        feats = self.features(x)
        x = self.avgpool(feats)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x, feats
    
    forward = _forward
    forward_vanilla = _forward_vanilla
    forward_with_feats = _forward_with_feats

    def compute_gradcam(self, output, feats, target):
        """
        Compute the gradcam for the top predicted category
        :param output:
        :param feats:
        :return:
        """
        eps = 1e-8

        target = target.cpu().numpy()
        # target = np.argmax(output.cpu().data.numpy(), axis=-1)
        one_hot = np.zeros((output.shape[0], output.shape[-1]), dtype=np.float32)
        indices_range = np.arange(output.shape[0])
        one_hot[indices_range, target[indices_range]] = 1
        one_hot = torch.from_numpy(one_hot)
        one_hot.requires_grad = True

        # Compute the Grad-CAM for the original image
        one_hot_cuda = torch.sum(one_hot.cuda() * output)
        dy_dz1, = torch.autograd.grad(one_hot_cuda, feats, grad_outputs=torch.ones(one_hot_cuda.size()).cuda(),
                                      retain_graph=True, create_graph=True)
        # dy_dz_sum1 = dy_dz1.sum(dim=2).sum(dim=2)
        # gcam512_1 = dy_dz_sum1.unsqueeze(-1).unsqueeze(-1) * feats
        # Changing to dot product of grad and features to preserve grad spatial locations
        gcam512_1 = dy_dz1 * feats
        gradcam = gcam512_1.sum(dim=1)
        gradcam = self.relu(gradcam)
        spatial_sum1 = gradcam.sum(dim=[1, 2]).unsqueeze(-1).unsqueeze(-1)
        gradcam = (gradcam / (spatial_sum1 + eps)) + eps

        return gradcam

    @staticmethod
    def get_next_index(quad_index_start_dict, index):
        for i in range(index + 1, 4):
            if i in quad_index_start_dict:
                return i
        return 'end'

def alexnet(pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model
