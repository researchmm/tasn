import torch
import torch.nn as nn
from torch.autograd import Function
import math
from torch.nn.modules.utils import _pair
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init
import att_grid_generator_cuda

__all__ = ['ResNet','model', 'resnet18', 'resnet50']



def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Atthead(nn.Module):
    expansion = 4

    def __init__(self, att = False):
        super(Atthead, self).__init__()
        self.att = att
        self.in_channels = 512
        self.out_channels = 512
        self.kernel_size = _pair(3)
        self.weight1 = nn.Parameter(torch.Tensor(self.out_channels, self.in_channels, *self.kernel_size))
        self.weight2 = nn.Parameter(torch.Tensor(self.out_channels, self.in_channels, *self.kernel_size))
        self.relu1_1 = nn.ReLU(inplace=True)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.relu2_3 = nn.ReLU(inplace=True)

        self.reset_parameters()



    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight1.data.uniform_(-stdv, stdv)
        self.weight2.data.uniform_(-stdv, stdv)

    def forward(self, x):
        if not self.att:
            return x

        att1_1 = F.conv2d(x, self.weight1, bias=None, stride=1, padding=1, dilation = 1)
        att1_1 = self.relu1_1(att1_1)
        att1_2 = F.conv2d(x, self.weight1, bias=None, stride=1, padding=2, dilation = 2)
        att1_2 = self.relu1_2(att1_2)
        att1 = att1_1 + att1_2
        att2_1 = F.conv2d(att1, self.weight2, bias=None, stride=1, padding=1, dilation = 1)
        att2_1 = self.relu2_1(att2_1)
        att2_2 = F.conv2d(att1, self.weight2, bias=None, stride=1, padding=2, dilation = 2)
        att2_2 = self.relu2_2(att2_2)
        att2_3 = F.conv2d(att1, self.weight2, bias=None, stride=1, padding=3, dilation = 3)
        att2_3 = self.relu2_3(att2_3)
        att2 = att2_1 + att2_2 + att2_3

        return att2





class tri_att(nn.Module):
  def __init__(self):
    super(tri_att, self).__init__()
    self.feature_norm = nn.Softmax(dim=2)
    self.bilinear_norm = nn.Softmax(dim=2)


  def forward(self, x):
    n = x.size(0)
    c = x.size(1)
    h = x.size(2)
    w = x.size(3)
    f = x.reshape(n, c, -1)

    # *7 to obtain an appropriate scale for the input of softmax function.
    f_norm = self.feature_norm(f * 2)

    bilinear = f_norm.bmm(f.transpose(1, 2))
    bilinear = self.bilinear_norm(bilinear)
    trilinear_atts = bilinear.bmm(f).view(n, c, h, w).detach()
    structure_att = torch.sum(trilinear_atts,dim=1, keepdim=True)

    index = torch.randint(c, (n,))
    detail_att = trilinear_atts[torch.arange(n),index] + 0.01




    '''
    structure_att = torch.sum(trilinear_atts, dim=(2, 3))
    structure_att_sorted, _ = torch.sort(structure_att, dim=1)
    structure_att_mask = (structure_att_sorted[:, 1:] !=
                          structure_att_sorted[:, :-1])
    one_vector = torch.ones((batch_size, 1), dtype=torch.uint8).cuda()
    structure_att_mask = torch.cat((one_vector, structure_att_mask), dim=1)
    structure_att_mask = structure_att_mask.unsqueeze(-1).transpose(1, 2).float()
    structure_att_ori = trilinear_atts.reshape(batch_size, inplanes, -1)


    structure_att = torch.matmul(structure_att_mask, structure_att_ori)
    structure_att = structure_att.view(batch_size, 1, h_in, w_in)
    structure_att = F.interpolate(structure_att, (self.ori_size, self.ori_size),
                                  mode='bilinear', align_corners=False).squeeze(1)
    structure_att = structure_att * structure_att
    '''
    return structure_att, detail_att.unsqueeze(1)


def att_sample(data, att, out_size):
    n = data.size(0)
    h = data.size(2)
    att = F.interpolate(att, (h, h), mode='bilinear', align_corners=False).squeeze(1)
    map_sx, _ = torch.max(att, 2)
    map_sx = map_sx.unsqueeze(2)
    map_sy, _ = torch.max(att, 1)
    map_sy = map_sy.unsqueeze(2)
    sum_sx = torch.sum(map_sx, (1, 2), keepdim=True)
    sum_sy = torch.sum(map_sy, (1, 2), keepdim=True)
    map_sx = torch.div(map_sx, sum_sx)
    map_sy = torch.div(map_sy, sum_sy)
    map_xi = torch.zeros_like(map_sx)
    map_yi = torch.zeros_like(map_sy)
  #index_x = data.new_empty((batch_size, out_size, 1))
  #index_y = data.new_empty((batch_size, out_size, 1)).cuda()
    index_x = torch.zeros((n, out_size, 1)).cuda()
    index_y = torch.zeros((n, out_size, 1)).cuda()
    att_grid_generator_cuda.forward(map_sx, map_sy, map_xi, map_yi,
                                    index_x, index_y,
                                    h, out_size, 4,
                                    5, out_size/h)
    one_vector = torch.ones_like(index_x)
    grid_x = torch.matmul(one_vector, index_x.transpose(1, 2)).unsqueeze(-1)
    grid_y = torch.matmul(index_y, one_vector.transpose(1, 2)).unsqueeze(-1)
    grid = torch.cat((grid_x, grid_y), 3)
    data = F.grid_sample(data, grid)
    return data



class ResNet(nn.Module):

    def __init__(self, block, layers, att = False, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.att = att
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1 if att else 2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1 if att else 2,
                                       dilate=replace_stride_with_dilation[2])
        self.head = Atthead(att) if att else None



#        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.head(x) if self.att else x
#        x = self.avgpool(x)
#        x = torch.flatten(x, 1)
#        x = self.fc(x)

        return x


class Tasn(nn.Module):

    def __init__(self, pretrained, progress):
        super(Tasn, self).__init__()
        self.model_att = resnet18(pretrained = pretrained, progress = progress)
        self.model_cls = resnet50(pretrained = pretrained, progress = progress)
        self.trilinear_att = tri_att()
        self.pool_att = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_att = nn.Linear(512, 200)
        self.pool_cls = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_cls = nn.Linear(512 * 4, 200)

        checkpoint_file = '/v-helzhe/model/resnet50.pth'
        checkpoint = torch.load(checkpoint_file)
        self.model_cls.load_state_dict(checkpoint, strict = False)
        checkpoint_file = '/v-helzhe/model/resnet18.pth'
        checkpoint = torch.load(checkpoint_file)
        self.model_att.load_state_dict(checkpoint, strict = False)

    def forward(self, x):
        n = x.size(0)
        c = x.size(1)
        w = x.size(2)

        input_att = F.interpolate(x, (224, 224), mode='bilinear', align_corners=False)
        conv_att = self.model_att(input_att)
        att_structure, att_detail = self.trilinear_att(conv_att)
        input_structure = att_sample(x,att_structure,224)
        input_detail = att_sample(x,att_detail,224)
        input_cls = torch.cat((input_structure, input_detail),0)
        conv_cls = self.model_cls(input_cls)
        out_att = self.pool_att(conv_att)
        out_att = torch.flatten(out_att, 1)
        out_att = self.fc_att(out_att)
        out_cls = self.pool_cls(conv_cls)
        out_cls = torch.flatten(out_cls, 1)
        out_cls = self.fc_cls(out_cls)
        out_structure, out_detail = torch.chunk(out_cls, 2)

        return out_att, out_structure, out_detail

def model(pretrained=False, progress=True, **kwargs):

    return Tasn(pretrained = pretrained, progress = progress)
#    return resnet18(pretrained = pretrained, progress = progress)#Tasn(pretrained = pretrained, progress = progress)

def _resnet(arch, block, layers, att, pretrained, progress, **kwargs):
    model = ResNet(block, layers, att, **kwargs)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], True, pretrained, progress,
                   **kwargs)

def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], False, pretrained, progress,
                   **kwargs)


