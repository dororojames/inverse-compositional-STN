import numpy as np
import scipy.linalg
import torch
import torch.nn.functional as F
from torch import nn

# fit (affine) warp between two sets of points


def fit(Xsrc, Xdst):
    ptsN = len(Xsrc)
    X, Y, U, V, O, I = Xsrc[:, 0], Xsrc[:, 1], Xdst[:,
                                                    0], Xdst[:, 1], np.zeros([ptsN]), np.ones([ptsN])
    A = np.concatenate((np.stack([X, Y, I, O, O, O], axis=1),
                        np.stack([O, O, O, X, Y, I], axis=1)), axis=0)
    b = np.concatenate((U, V), axis=0)
    p1, p2, p3, p4, p5, p6 = scipy.linalg.lstsq(A, b)[0].squeeze()
    pMtrx = np.array([[p1, p2, p3], [p4, p5, p6], [0, 0, 1]],
                     dtype=torch.float32)
    return pMtrx


def vec2mtrx(p):
    """convert warp parameters to matrix"""
    B = p.size(0)
    O = p.new_zeros(B)
    I = p.new_ones(B)

    if p.size(1) == 2:  # "translation"
        tx, ty = torch.unbind(p, dim=1)
        pMtrx = torch.stack([torch.stack([I, O, tx], dim=-1),
                             torch.stack([O, I, ty], dim=-1),
                             torch.stack([O, O, I], dim=-1)], dim=1)
    elif p.size(1) == 4:  # "similarity"
        pc, ps, tx, ty = torch.unbind(p, dim=1)
        pMtrx = torch.stack([torch.stack([I+pc, -ps, tx], dim=-1),
                             torch.stack([ps, I+pc, ty], dim=-1),
                             torch.stack([O, O, I], dim=-1)], dim=1)
    elif p.size(1) == 6:  # "affine"
        p1, p2, p3, p4, p5, p6 = torch.unbind(p, dim=1)
        pMtrx = torch.stack([torch.stack([I+p1, p2, p3], dim=-1),
                             torch.stack([p4, I+p5, p6], dim=-1),
                             torch.stack([O, O, I], dim=-1)], dim=1)
    elif p.size(1) == 8:  # "homography"
        I = torch.eye(3).to(p.device).repeat(B, 1, 1)
        p = torch.cat([p, torch.zeros_like(p[..., 0:1])], dim=-1)
        pMtrx = p.view_as(I) + I

    return pMtrx


def mtrx2vec(pMtrx, warpType):
    """convert warp matrix to parameters"""
    row0, row1, row2 = torch.unbind(pMtrx, dim=1)
    e00, e01, e02 = torch.unbind(row0, dim=1)
    e10, e11, e12 = torch.unbind(row1, dim=1)
    e20, e21,   _ = torch.unbind(row2, dim=1)
    if warpType == "translation":
        p = torch.stack([e02, e12], dim=1)
    elif warpType == "similarity":
        p = torch.stack([e00-1, e10, e02, e12], dim=1)
    elif warpType == "affine":
        p = torch.stack([e00-1, e01, e02, e10, e11-1, e12], dim=1)
    elif warpType == "homography":
        p = torch.stack([e00-1, e01, e02, e10, e11-1, e12, e20, e21], dim=1)
    return p


def compose(p, dp, warpType):
    """compute composition of warp parameters"""
    pMtrx = vec2mtrx(p)
    dpMtrx = vec2mtrx(dp)
    pMtrxNew = dpMtrx.matmul(pMtrx)
    pMtrxNew = pMtrxNew/pMtrxNew[:, 2:3, 2:3]
    pNew = mtrx2vec(pMtrxNew, warpType)
    return pNew


def inverse(p, warpType):
    """compute inverse of warp parameters"""
    pMtrx = vec2mtrx(p)
    pInvMtrx = pMtrx.inverse()
    pInv = mtrx2vec(pInvMtrx, warpType)
    return pInv


def transformImage(image, p):
    """warp the image"""
    B, _, H, W = image.size()
    pMtrx = vec2mtrx(p)
    refMtrx = torch.eye(3).to(p.device).repeat(B, 1, 1)
    transMtrx = refMtrx.matmul(pMtrx)

    # warp the canonical coordinates
    X, Y = torch.meshgrid(torch.linspace(-1, 1, W),
                          torch.linspace(-1, 1, H))
    X, Y = X.flatten(), Y.flatten()
    XYhom = torch.stack([X, Y, torch.ones_like(X)], dim=1).t()
    XYhom = XYhom.repeat(B, 1, 1).to(image.device)
    XYwarpHom = transMtrx.matmul(XYhom).transpose(1, 2).reshape(B, H, W, 3)

    grid, ZwarpHom = XYwarpHom.split([2, 1], dim=-1)
    grid = grid / ZwarpHom.add(1e-8)

    # sampling with bilinear interpolation
    imageWarp = F.grid_sample(image, grid, mode="bilinear", align_corners=True)
    return imageWarp
