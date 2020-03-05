import torch
from torch import nn

import warp

# build classification network


class FullCNN(nn.Module):
    def __init__(self, opt):
        super(FullCNN, self).__init__()
        self.inDim = 1

        def conv2Layer(outDim):
            conv = nn.Conv2d(self.inDim, outDim, 3)
            self.inDim = outDim
            return conv

        def linearLayer(outDim):
            fc = nn.Linear(self.inDim, outDim)
            self.inDim = outDim
            return fc

        def maxpoolLayer(): return nn.MaxPool2d(2)
        self.conv2Layers = nn.Sequential(
            conv2Layer(3), nn.ReLU(True),
            conv2Layer(6), nn.ReLU(True), maxpoolLayer(),
            conv2Layer(9), nn.ReLU(True),
            conv2Layer(12), nn.ReLU(True)
        )
        self.inDim *= 8**2
        self.linearLayers = nn.Sequential(
            linearLayer(48), nn.ReLU(True),
            linearLayer(opt.labelN)
        )
        initialize(self, opt.stdC)

    def forward(self, image):
        feat = image
        feat = self.conv2Layers(feat).flatten(1)
        feat = self.linearLayers(feat)
        output = feat
        return output

# build classification network


class CNN(nn.Module):
    def __init__(self, opt):
        super(CNN, self).__init__()
        self.inDim = 1

        def conv2Layer(outDim):
            conv = nn.Conv2d(self.inDim, outDim, 9)
            self.inDim = outDim
            return conv

        def linearLayer(outDim):
            fc = nn.Linear(self.inDim, outDim)
            self.inDim = outDim
            return fc

        def maxpoolLayer(): return nn.MaxPool2d(2)
        self.conv2Layers = nn.Sequential(
            conv2Layer(3), nn.ReLU(True)
        )
        self.inDim *= 20**2
        self.linearLayers = nn.Sequential(
            linearLayer(opt.labelN)
        )
        initialize(self, opt.stdC)

    def forward(self, image):
        feat = image
        feat = self.conv2Layers(feat).flatten(1)
        feat = self.linearLayers(feat)
        output = feat
        return output

# an identity class to skip geometric predictors


class Identity(nn.Module):
    def __init__(self): super(Identity, self).__init__()
    def forward(self, feat): return [feat]

# build Spatial Transformer Network


class STN(nn.Module):
    def __init__(self, opt):
        super(STN, self).__init__()
        self.inDim = 1
        self.warpType = opt.warpType

        def conv2Layer(outDim):
            conv = nn.Conv2d(self.inDim, outDim, 7)
            self.inDim = outDim
            return conv

        def linearLayer(outDim):
            fc = nn.Linear(self.inDim, outDim)
            self.inDim = outDim
            return fc

        def maxpoolLayer(): return nn.MaxPool2d(2)
        self.conv2Layers = nn.Sequential(
            conv2Layer(4), nn.ReLU(True),
            conv2Layer(8), nn.ReLU(True), maxpoolLayer()
        )
        self.inDim *= 8**2
        self.linearLayers = nn.Sequential(
            linearLayer(48), nn.ReLU(True),
            linearLayer(opt.warpDim)
        )
        initialize(self, opt.stdGP, last0=True)

    def forward(self, image):
        imageWarpAll = [image]
        feat = image
        feat = self.conv2Layers(feat).flatten(1)
        feat = self.linearLayers(feat)
        p = feat
        imageWarp = warp.transformImage(image, p)
        imageWarpAll.append(imageWarp)
        return imageWarpAll

# build Inverse Compositional STN


class ICSTN(nn.Module):
    def __init__(self, opt):
        super(ICSTN, self).__init__()
        self.inDim = 1
        self.warpN = opt.warpN
        self.warpType = opt.warpType

        def conv2Layer(outDim):
            conv = nn.Conv2d(self.inDim, outDim, 7)
            self.inDim = outDim
            return conv

        def linearLayer(outDim):
            fc = nn.Linear(self.inDim, outDim)
            self.inDim = outDim
            return fc

        def maxpoolLayer(): return nn.MaxPool2d(2)
        self.conv2Layers = nn.Sequential(
            conv2Layer(4), nn.ReLU(True),
            conv2Layer(8), nn.ReLU(True), maxpoolLayer()
        )
        self.inDim *= 8**2
        self.linearLayers = nn.Sequential(
            linearLayer(48), nn.ReLU(True),
            linearLayer(opt.warpDim)
        )
        initialize(self, opt.stdGP, last0=True)

    def forward(self, image, p):
        imageWarpAll = []
        for _ in range(self.warpN):
            imageWarp = warp.transformImage(image, p)
            imageWarpAll.append(imageWarp)
            feat = imageWarp
            feat = self.conv2Layers(feat).flatten(1)
            feat = self.linearLayers(feat)
            dp = feat
            p = warp.compose(p, dp, self.warpType)
        imageWarp = warp.transformImage(image, p)
        imageWarpAll.append(imageWarp)
        return imageWarpAll

# initialize weights/biases


def initialize(model, stddev, last0=False):
    for m in model.conv2Layers:
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, stddev)
            m.bias.data.normal_(0, stddev)
    for m in model.linearLayers:
        if isinstance(m, nn.Linear):
            if last0 and m is model.linearLayers[-1]:
                m.weight.data.zero_()
                m.bias.data.zero_()
            else:
                m.weight.data.normal_(0, stddev)
                m.bias.data.normal_(0, stddev)
