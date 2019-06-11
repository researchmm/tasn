"""
The residual unit is adapted from https://github.com/apache/incubator-mxnet/blob/master/example/image-classification/symbols/resnet.py 
by Heliang Zheng
03/28/2019

"""

import random
import os
import mxnet as mx
import numpy as np


def residual_unit(data, num_filter, stride, dim_match, name, bottle_neck=True, bn_mom=0.9, workspace=256, memonger=False):
    """Return ResNet Unit symbol for building ResNet
    Parameters
    ----------
    data : str
        Input data
    num_filter : int
        Number of output channels
    bnf : int
        Bottle neck channels factor with regard to num_filter
    stride : tuple
        Stride used in convolution
    dim_match : Boolean
        True means channel number between input and output is the same, otherwise means differ
    name : str
        Base name of the operators
    workspace : int
        Workspace used in convolution operator
    """
    if bottle_neck:
        # the same as https://github.com/facebook/fb.resnet.torch#notes, a bit difference with origin paper
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = mx.sym.Convolution(data=act1, num_filter=int(num_filter*0.25), kernel=(1,1), stride=(1,1), pad=(0,0),
                                   no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        conv2 = mx.sym.Convolution(data=act2, num_filter=int(num_filter*0.25), kernel=(3,3), stride=stride, pad=(1,1),
                                   no_bias=True, workspace=workspace, name=name + '_conv2')
        bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')
        act3 = mx.sym.Activation(data=bn3, act_type='relu', name=name + '_relu3')
        conv3 = mx.sym.Convolution(data=act3, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0), no_bias=True,
                                   workspace=workspace, name=name + '_conv3')
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return conv3 + shortcut
    else:
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(3,3), stride=stride, pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        conv2 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv2')
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return conv2 + shortcut

class Detail_Att(mx.operator.CustomOp):
    def forward(self, is_train, req, in_data, out_data, aux):
        f = in_data[0]
        w = f.shape[2]
        n = f.shape[0]
        if is_train:
            a = mx.nd.sum(mx.nd.reshape(f,(0,0,-1)),axis=-1)
            x = a.asnumpy()
            y = np.zeros((n,1,1,1))
            for i in range(n):
                a = x[i]
                idx = np.argsort(a)
                a_sorted = a[idx]
                unq_first = np.concatenate(([True], a_sorted[1:] != a_sorted[:-1]))
                unq_items = idx[unq_first]
                np.random.shuffle(unq_items)
                y[i] = unq_items[0]
            b = mx.nd.array(y)
            b = mx.nd.tile(b,(1,1,w,w))
            c = mx.nd.pick(f, b, axis = 1)
        else:
            c = mx.nd.sum(f, axis=1)
        c = mx.nd.expand_dims(c,axis=1)
        self.assign(out_data[0], req[0], c)
    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        y = np.zeros(in_grad[0].shape)
        self.assign(in_grad[0], req[0], mx.nd.array(y))
@mx.operator.register("detail_att")
class Detail_AttProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(Detail_AttProp, self).__init__(need_top_grad=False)
    def list_arguments(self):
        return ['data']
    def list_outputs(self):
        return ['output']
    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        output_shape = (in_shape[0][0],1,in_shape[0][2],in_shape[0][3])
        return [data_shape], [output_shape], []
    def infer_type(self, in_type):
        dtype = in_type[0]
        return [dtype], [dtype], []
    def create_operator(self, ctx, shapes, dtypes):
        return Detail_Att()

class Structure_Att(mx.operator.CustomOp):
    def forward(self, is_train, req, in_data, out_data, aux):
        f = in_data[0]
        w = f.shape[2]
        n = f.shape[0]
        c = f.shape[1]
        a = mx.nd.sum(mx.nd.reshape(f,(0,0,-1)),axis=-1)
        x = a.asnumpy()
        y = np.zeros((n,c))
        for i in range(n):
            a = x[i]
            idx = np.argsort(a)
            a_sorted = a[idx]
            y[i] = np.concatenate(([1], a_sorted[1:] != a_sorted[:-1]))
        b = mx.nd.array(y)
        o = mx.nd.batch_dot(b.reshape((n,c,1)), f.reshape((n,c,w*w)), True, False).reshape((n,1,w,w))
        self.assign(out_data[0], req[0], o)
    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        y = np.zeros(in_grad[0].shape)
        self.assign(in_grad[0], req[0], mx.nd.array(y))
@mx.operator.register("structure_att")
class Structure_AttProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(Structure_AttProp, self).__init__(need_top_grad=False)
    def list_arguments(self):
        return ['data']
    def list_outputs(self):
        return ['output']
    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        output_shape = (in_shape[0][0],1,in_shape[0][2],in_shape[0][3])
        return [data_shape], [output_shape], []
    def infer_type(self, in_type):
        dtype = in_type[0]
        return [dtype], [dtype], []
    def create_operator(self, ctx, shapes, dtypes):
        return Structure_Att()

def attention_net(data, num_class):
    units = [2, 2, 2, 2] # for resnet-18
    filter_list = [64, 64, 128, 256, 512]
    num_stage = 4
    bottle_neck=False
    bn_mom=0.9
    workspace=512
    memonger=False

    num_unit = len(units)
    #weights and bias for multi-dilation convs 
    att_conv1_weight = mx.sym.Variable(name = 'att_conv1_weight',init=mx.init.MSRAPrelu())
    att_conv1_bias = mx.sym.Variable(name = 'att_conv1_bias')
    att_conv2_weight = mx.sym.Variable(name = 'att_conv2_weight',init=mx.init.MSRAPrelu())
    att_conv2_bias = mx.sym.Variable(name = 'att_conv2_bias')
    #resnet-18 for attention generation
    data = mx.sym.contrib.BilinearResize2D(data=data, height=224, width=224) 
    bn = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='att_bn_data')
    body = mx.sym.Convolution(data=bn, num_filter=filter_list[0], kernel=(7, 7), stride=(2,2), pad=(3, 3),
                                  no_bias=True, name="att_conv0", workspace=workspace)
    body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='att_bn0')
    body = mx.sym.Activation(data=body, act_type='relu', name='att_relu0')
    body = mx.sym.Pooling(data=body, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')

    for i in range(4):
        body = residual_unit(body, filter_list[i+1], (1 if (i!=2) else 2, 1 if (i!=2) else 2), False,
                             name='att_stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, workspace=workspace,
                             memonger=memonger)
        for j in range(units[i]-1):
            body = residual_unit(body, filter_list[i+1], (1,1), True, name='att_stage%d_unit%d' % (i + 1, j + 2),
                                 bottle_neck=bottle_neck, workspace=workspace, memonger=memonger)
    #multi-dilation convs module    
    att_conv1_1 = mx.sym.Convolution(data = body, name = 'att_conv1_1', weight = att_conv1_weight, bias = att_conv1_bias, num_filter = 512, pad = (1,1), kernel = (3,3), cudnn_off = True, dilate = (1,1))
    att_act1_1 = mx.sym.Activation(data = att_conv1_1, act_type='relu', name='att_act1_1')
    att_conv1_2 = mx.sym.Convolution(data = body, name = 'att_conv1_2', weight = att_conv1_weight, bias = att_conv1_bias, num_filter = 512, pad = (2,2), kernel = (3,3), cudnn_off = True, dilate = (2,2))
    att_act1_2 = mx.sym.Activation(data = att_conv1_2, act_type='relu', name='att_act1_2')
    att_conv1 =  mx.sym.ElementWiseSum(att_act1_1, att_act1_2, name = 'att_conv1')

    att_conv2_1 = mx.sym.Convolution(data = att_conv1, name = 'att_conv2_1', weight = att_conv2_weight, bias = att_conv2_bias, num_filter = 512, pad = (1,1), kernel = (3,3), cudnn_off = True, dilate = (1,1))
    att_act2_1 = mx.sym.Activation(data = att_conv2_1, act_type='relu', name='att_act2_1')
    att_conv2_2 = mx.sym.Convolution(data = att_conv1, name = 'att_conv2_2', weight = att_conv2_weight, bias = att_conv2_bias, num_filter = 512, pad = (2,2), kernel = (3,3), cudnn_off = True, dilate = (2,2))
    att_act2_2 = mx.sym.Activation(data = att_conv2_2, act_type='relu', name='att_act2_2')
    att_conv2_3 = mx.sym.Convolution(data = att_conv1, name = 'att_conv2_3', weight = att_conv2_weight, bias = att_conv2_bias, num_filter = 512, pad = (3,3), kernel = (3,3), cudnn_off = True, dilate = (3,3))
    att_act2_3 = mx.sym.Activation(data = att_conv2_3, act_type='relu', name='att_act2_3')
    att_conv2 =  mx.sym.ElementWiseSum(att_act2_1, att_act2_2, att_act2_3, name = 'att_conv2')
    
    bn1 = mx.sym.BatchNorm(data=att_conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='att_bn1')
    relu1 = mx.sym.Activation(data=bn1, act_type='relu', name='att_relu1')
    # Although kernel is not used here when global_pool=True, we should put one
    pool1 = mx.sym.Pooling(data=relu1, global_pool=True, kernel=(7, 7), pool_type='avg', name='att_pool1')
    flat = mx.sym.Flatten(data=pool1)
    fc1 = mx.sym.FullyConnected(data=flat, num_hidden=num_class, name='att_fc1')
    att_net = mx.sym.SoftmaxOutput(data=fc1, name='att_net')
    return att_net, att_conv2    

def trilinear_att(feature_maps):

    feature_norm = mx.sym.softmax(feature_maps.reshape((0,0,-1))*7) # *7 to obtain an appropriate scale for the input of softmax function.
    feature_ori = feature_maps.reshape((0,0,-1))
    bilinear = mx.sym.batch_dot(feature_norm, feature_ori, False, True)
    bilinear = mx.sym.softmax(bilinear)
    trilinear_atts = mx.sym.batch_dot(bilinear, feature_ori, False, False).reshape((0,512,28,28))
    trilinear_atts = mx.sym.BlockGrad(trilinear_atts)
    structure_att = mx.sym.Custom(data = trilinear_atts, op_type='structure_att').reshape((0,28,28))
    structure_att = mx.sym.expand_dims(structure_att, axis=1) # Expand_dims to fit into BilinearResize2D layer.
    structure_att = mx.sym.contrib.BilinearResize2D(data = structure_att, height=512, width=512)# Resized the attention map to match the original input image.
    structure_att = mx.sym.sum(structure_att, axis=1) # Sum to reduce dims, note that there is only one channel (one attention map).
    structure_att = structure_att * structure_att # It is optinal to use structure_att or structure_att * structure_att, and the later has a larger value for important regoins.

    detail_att = mx.sym.Custom(data = trilinear_atts, op_type='detail_att').reshape((0,28,28))
    detail_att = mx.sym.BlockGrad(detail_att)
    detail_att = mx.sym.expand_dims(detail_att, axis=1)
    detail_att = mx.sym.contrib.BilinearResize2D(data = detail_att, height=512, width=512)
    detail_att = mx.sym.sum(detail_att,axis=1)

    return detail_att, structure_att


def att_sample(data, detail_att, structure_att):

    map_sx = mx.sym.expand_dims(mx.sym.max(structure_att, axis=2), axis=-1)
    map_sy = mx.sym.expand_dims(mx.sym.max(structure_att, axis=1), axis=-1)
    sum_sx = mx.sym.sum(map_sx, axis=(1,2))
    sum_sy = mx.sym.sum(map_sy, axis=(1,2))
    map_sx = mx.sym.broadcast_div(map_sx, mx.sym.reshape(sum_sx,(0,1,1)))
    map_sy = mx.sym.broadcast_div(map_sy, mx.sym.reshape(sum_sy,(0,1,1)))

    structure_data = mx.sym.contrib.AttSampler(data=data, attx=map_sx, atty=map_sy, scale=224.0/512, dense=4)

    map_dx = mx.sym.expand_dims(mx.sym.max(detail_att, axis=2), axis=-1)
    map_dy = mx.sym.expand_dims(mx.sym.max(detail_att, axis=1), axis=-1)
    sum_dx = mx.sym.sum(map_dx, axis=(1,2))
    sum_dy = mx.sym.sum(map_dy, axis=(1,2))
    map_dx = mx.sym.broadcast_div(map_dx, mx.sym.reshape(sum_dx,(0,1,1)))
    map_dy = mx.sym.broadcast_div(map_dy, mx.sym.reshape(sum_dy,(0,1,1)))

    detail_data = mx.sym.contrib.AttSampler(data=data, attx=map_dx, atty=map_dy, scale=224.0/512, dense=4)


    return detail_data, structure_data


def part_master_net(data, batch_size, num_class):

    units = [3, 4, 6, 3] # for resnet-50
    filter_list = [64, 256, 512, 1024, 2048]
    num_stage = 4
    bottle_neck = True
    bn_mom=0.9
    workspace=512
    memonger=False

    num_unit = len(units)
    
    data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data')
    body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(7, 7), stride=(2,2), pad=(3, 3),
                                  no_bias=True, name="conv0", workspace=workspace)
    body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
    body = mx.sym.Activation(data=body, act_type='relu', name='relu0')
    body = mx.symbol.Pooling(data=body, kernel=(3, 3), stride=(2,2), pad = (1,1), pool_type='max')
    for i in range(4):
        body = residual_unit(body, filter_list[i+1], (1 if (i==0) else 2, 1 if (i==0) else 2), False,
                             name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, workspace=workspace,
                             memonger=memonger)
        
        for j in range(units[i]-1):
            body = residual_unit(body, filter_list[i+1], (1,1), True, name='stage%d_unit%d' % (i + 1, j + 2),
                                 bottle_neck=bottle_neck, workspace=workspace, memonger=memonger)

    bn1 = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1')
    relu1 = mx.sym.Activation(data=bn1, act_type='relu')
    pool1 = mx.symbol.Pooling(data=relu1, global_pool=True, kernel=(7, 7), pool_type='avg')

    pool_d = mx.sym.slice_axis(pool1, axis=0, begin=0, end=batch_size)
    pool_s = mx.sym.slice_axis(pool1, axis=0, begin=batch_size, end=batch_size*2)

    #Note that the fc layer of part net and master net can be 1) shared or 2) not shared. Here we use non-shared fc layers for distilling and shared fc layers for the final prediction. Different settings can be tried and the performances are supposed be comparable after tunning parameters. 
    
    flat = mx.symbol.Flatten(data=pool_d)
    fc1 = mx.symbol.FullyConnected(data=flat, num_hidden=num_class)#, attr={"lr_mult": "1"})
    part_net = mx.symbol.SoftmaxOutput(data=fc1, name='part_net', grad_scale=0.5)

    flat = mx.symbol.Flatten(data=pool_s)
    fc2 = mx.symbol.FullyConnected(data=flat, num_hidden=num_class)#, attr={"lr_mult": "1"})
    master_net = mx.symbol.SoftmaxOutput(data=fc2, name='master_net', grad_scale=0.5)

    flat = mx.symbol.Flatten(data=pool1)
    fc = mx.symbol.FullyConnected(data=flat, num_hidden=num_class)#, attr={"lr_mult": "1"})
    fc_1 = mx.sym.slice_axis(fc, axis=0, begin=0, end=batch_size)
    fc_2 = mx.sym.slice_axis(fc, axis=0, begin=batch_size, end=batch_size*2)
    
    part_net_aux = mx.symbol.SoftmaxOutput(data=fc_1, name='part_net_aux', grad_scale=0.5)
    master_net_aux = mx.symbol.SoftmaxOutput(data=fc_2, name='master_net_aux', grad_scale=0.5)

    Loss = mx.gluon.loss.SoftmaxCrossEntropyLoss(weight = 1, sparse_label=False)
    distill_loss = Loss((fc2/10), mx.sym.softmax(fc1/10))
    distill_loss = mx.sym.make_loss(distill_loss)

    return part_net, master_net, part_net_aux, master_net_aux, distill_loss


def tasn(symbol, arg_params, aux_params, num_classes, batch_size):
    """
    symbol: the pre-trained network symbol
    arg_params: the argument parameters of the pre-trained model
    aux_params: the aux parameters of the pre-trained model
    num_classes: the number of classes for the fine-tune datasets
    """

    input_data = mx.sym.Variable(name='data')
    att_net, feature_maps = attention_net(input_data, num_classes)
    detail_att, structure_att = trilinear_att(feature_maps)
    detail_data, structure_data = att_sample(input_data, detail_att, structure_att)
    merge_data = mx.sym.concat(detail_data, structure_data, dim=0)
    part_net, master_net, part_net_aux, master_net_aux, distill_loss = part_master_net(merge_data, batch_size, num_classes)
    nets = mx.sym.Group([att_net, part_net, master_net, part_net_aux, master_net_aux, distill_loss])

    new_args = dict({k:arg_params[k] for k in arg_params if 'fc' not in k })#and 'stage3' not in k and 'stage4' not in k and 'bn1' not in k)})
    new_auxs = dict({k:aux_params[k] for k in aux_params})
    prefix = './model/resnet-18'
    epoch = 0
    _, att_params, att_auxs = mx.model.load_checkpoint(prefix, epoch)

    
    rename_lists = ['bn_data', 'conv0', 'bn0', 'relu0', 'stage1', 'stage2', 'stage3', 'stage4', 'bn1']
    for k in att_params:
        for r in rename_lists:
            if r in k:
                new_args['att_' + k] = att_params[k]
                break

    for k in att_auxs:
        new_auxs['att_' + k] = att_auxs[k]

    
    return (nets, new_args, new_auxs)




