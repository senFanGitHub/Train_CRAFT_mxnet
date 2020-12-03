
from mxnet.context import cpu
from mxnet.initializer import Xavier
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
from mxnet.contrib.ndarray import BilinearResize2D
from mxnet import nd
import mxnet as mx


class VGG(HybridBlock):
    r"""VGG model from the `"Very Deep Convolutional Networks for Large-Scale Image Recognition"
    <https://arxiv.org/abs/1409.1556>`_ paper.

    Parameters
    ----------
    layers : list of int
        Numbers of layers in each feature block.
    filters : list of int
        Numbers of filters in each feature block. List length should match the layers.
    classes : int, default 1000
        Number of classification classes.
    batch_norm : bool, default False
        Use batch normalization.
    """

    def __init__(self, layers, filters, classes=1000, batch_norm=True, **kwargs):
        super(VGG, self).__init__(**kwargs)
        assert len(layers) == len(filters)
        with self.name_scope():
            self.features = self._make_features(layers, filters, batch_norm)


    def _make_features(self, layers, filters, batch_norm):
        featurizer = nn.HybridSequential(prefix='')
        for i, num in enumerate(layers):
            for _ in range(num):
                featurizer.add(nn.Conv2D(filters[i], kernel_size=3, padding=1,
                                         weight_initializer=Xavier(rnd_type='gaussian',
                                                                   factor_type='out',
                                                                   magnitude=2),
                                         bias_initializer='zeros'))
                if batch_norm:
                    featurizer.add(nn.BatchNorm())
                if (featurizer.__len__() > 38):
                    break
                featurizer.add(nn.Activation('relu'))
            if (featurizer.__len__() > 38):
                break
            featurizer.add(nn.MaxPool2D(strides=2))
        return featurizer


# Specification
vgg_spec = {
            16: ([2, 2, 3, 3, 3], [64, 128, 256, 512, 512])
}


# Constructors
def get_vgg(num_layers, **kwargs):
    layers, filters = vgg_spec[num_layers]
    net = VGG(layers, filters, **kwargs)
    return net


class CharDet(HybridBlock):
    def __init__(self, train_init=False,ctx=mx.cpu()):
        super(CharDet, self).__init__()
        self.vgg_backbone = get_vgg(16)
        
        
        with self.name_scope():

            self.feature1, self.feature2, self.feature3, self.feature4 = self._copy_vgg()

            self.custom = nn.HybridSequential(prefix='')
            self.custom.add(nn.MaxPool2D(pool_size=(3, 3), strides=1, padding=1))

            self.custom.add(nn.Conv2D(1024, kernel_size=3, padding=6, dilation=(6, 6),
                                      weight_initializer=Xavier(rnd_type='gaussian',
                                                                factor_type='out',
                                                                magnitude=2)))
            self.custom.add(nn.Conv2D(1024, kernel_size=1,
                                      weight_initializer=Xavier(rnd_type='gaussian',
                                                                factor_type='out',
                                                                magnitude=2)))
            self.upsample1 = self._make_up_conv(512, 256)
            self.upsample2 = self._make_up_conv(256, 128)
            self.upsample3 = self._make_up_conv(128, 64)
            self.upsample4 = self._make_up_conv(64, 32)

            self.classifier = nn.HybridSequential(prefix='')
            self.classifier.add(nn.Conv2D(32, kernel_size=3, padding=1, activation='relu'))
            self.classifier.add(nn.Conv2D(32, kernel_size=3, padding=1, activation='relu'))
            self.classifier.add(nn.Conv2D(16, kernel_size=3, padding=1, activation='relu'))
            self.classifier.add(nn.Conv2D(16, kernel_size=1, activation='relu'))
            self.classifier.add(nn.Conv2D(2, kernel_size=1))
            
        if train_init:
            self.vgg_backbone.load_parameters('/data1/ml/.mxnet/models/vgg16_bn-7f01cf05.params', allow_missing=True, ignore_extra=True)
            self.custom.initialize(mx.init.Xavier(factor_type="in", magnitude=2.34), ctx=ctx)
            self.upsample1.initialize(mx.init.Xavier(factor_type="in", magnitude=2.34), ctx=ctx)
            self.upsample2.initialize(mx.init.Xavier(factor_type="in", magnitude=2.34), ctx=ctx)
            self.upsample3.initialize(mx.init.Xavier(factor_type="in", magnitude=2.34), ctx=ctx)
            self.upsample4.initialize(mx.init.Xavier(factor_type="in", magnitude=2.34), ctx=ctx)
            
            self.classifier.initialize(mx.init.Xavier(factor_type="in", magnitude=2.34), ctx=ctx)
        
            
            
            

    def _copy_vgg(self):
        features_1 = nn.HybridSequential(prefix='')
        features_2 = nn.HybridSequential(prefix='')
        features_3 = nn.HybridSequential(prefix='')
        features_4 = nn.HybridSequential(prefix='')
        for i in range(12):
            features_1.add(self.vgg_backbone.features._children[str(i)])
        for i in range(12, 19):
            features_2.add(self.vgg_backbone.features._children[str(i)])
        for i in range(19, 29):
            features_3.add(self.vgg_backbone.features._children[str(i)])
        for i in range(29, 39):
            features_4.add(self.vgg_backbone.features._children[str(i)])

        return features_1, features_2, features_3, features_4

    def _make_up_conv(self, mid_num, out_num):
        up_sample = nn.HybridSequential(prefix='')
        up_sample.add(nn.Conv2D(mid_num, kernel_size=1))
        up_sample.add(nn.BatchNorm())
        up_sample.add(nn.Activation('relu'))
        up_sample.add(nn.Conv2D(out_num, kernel_size=3, padding=1))
        up_sample.add(nn.BatchNorm())
        up_sample.add(nn.Activation('relu'))

        return up_sample

    def hybrid_forward(self, F, x):

        
        features_1=self.feature1(x)
        features_2=self.feature2(features_1)
        features_3=self.feature3(features_2)
        features_4=self.feature4(features_3)
        features_5=self.custom(features_4)

        y = nd.concat(features_5, features_4, dim=1)

        y = self.upsample1(y)

        y = BilinearResize2D(y, height=features_3.shape[2],width=features_3.shape[3])
        
#         y=nd.transpose(y,(0,2,3,1))
#         y=mx.ndarray.image.resize(y,(features_3.shape[3],features_3.shape[2]))
#         y=nd.transpose(y,(0,3,1,2))

        y = nd.concat(y, features_3, dim=1)
        y = self.upsample2(y)
        y = BilinearResize2D(y, height=features_2.shape[2],width=features_2.shape[3])

        
        
        y = nd.concat(y, features_2, dim=1)
        y = self.upsample3(y)
        y = BilinearResize2D(y, height=features_1.shape[2],width=features_1.shape[3])

        y = nd.concat(y, features_1, dim=1)
        
        y = self.upsample4(y)
       
        y = self.classifier(y)
#         y = nd.transpose(y,(0,2,3,1))
        
        return y






