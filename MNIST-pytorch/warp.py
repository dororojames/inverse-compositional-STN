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


def vec2mat(p):
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


def mat2vec(pMtrx, warpType):
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
    pMtrx = vec2mat(p)
    dpMtrx = vec2mat(dp)
    pMtrxNew = dpMtrx.bmm(pMtrx)
    pMtrxNew = pMtrxNew/pMtrxNew[:, 2:3, 2:3]
    pNew = mat2vec(pMtrxNew, warpType)
    return pNew


def inverse(p, warpType):
    """compute inverse of warp parameters"""
    pMtrx = vec2mat(p)
    pInvMtrx = pMtrx.inverse()
    pInv = mat2vec(pInvMtrx, warpType)
    return pInv


def meshgrid(size):
    # type: (List[int]) -> Tensor
    """return meshgrid (B, H, W, 2) of input size(width first, range (-1, -1)~(1, 1))"""
    coordh, coordw = torch.meshgrid(torch.linspace(-1, 1, size[-2]),
                                    torch.linspace(-1, 1, size[-1]))
    return torch.stack([coordw, coordh], dim=2).repeat(size[0], 1, 1, 1)


def cat_grid_z(grid, fill_value: int = 1):
    """concat z axis of grid at last dim , return shape (B, H, W, 3)"""
    return torch.cat([grid, torch.full_like(grid[..., 0:1], fill_value)], dim=-1)


def affine_grid(theta, size):
    # type: (Tensor, List[int]) -> Tensor
    grid = meshgrid(size).to(theta.device)
    return cat_grid_z(grid).flatten(1, 2).bmm(theta.transpose(1, 2)).view_as(grid)


def homography_grid(matrix, size):
    # type: (Tensor, List[int]) -> Tensor
    grid = cat_grid_z(meshgrid(size).to(matrix.device))  # B, H, W, 3
    homography = grid.flatten(1, 2).bmm(matrix.transpose(1, 2)).view_as(grid)
    grid, ZwarpHom = homography.split([2, 1], dim=-1)
    return grid / ZwarpHom.add(1e-8)


def transformImage(image, p):
    """warp the image"""
    if p.size(0) != image.size(0):
        p = p.repeat(image.size(0), 1)
    transMtrx = vec2mat(p)

    # warp the canonical coordinates
    grid = homography_grid(transMtrx, image.size())

    # sampling with bilinear interpolation
    imageWarp = F.grid_sample(image, grid, mode="bilinear", align_corners=True)
    return imageWarp
