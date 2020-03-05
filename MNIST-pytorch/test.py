import torch

import options
import warp
import graph
import data

if __name__ == "__main__":
    opt = options.set(training=False)
    geometric = graph.ICSTN(opt)
    classifier = graph.CNN(opt)
    print(geometric)
    print(classifier)
    pInit = data.genPerturbations(opt)
    print(pInit)

    # forward/backprop through network
    image = torch.rand(3, 1, 28, 28)
    imagePert = warp.transformImage(image, pInit)
    imageWarpAll = geometric(
        image, pInit) if opt.netType == "IC-STN" else geometric(imagePert)
    imageWarp = imageWarpAll[-1]
    output = classifier(imageWarp)
    print(output.shape)
