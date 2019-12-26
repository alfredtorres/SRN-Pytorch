from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import etw_pytorch_utils as pt_utils

from pointnet2.utils import pointnet2_utils

if False:
    # Workaround for type hints without depending on the `typing` module
    from typing import *

class SRNModule(nn.Module):
    def __init__(self, npoints, gu, gv, h, f):
        super(SRNModule, self).__init__()
        self.npoints = npoints
        self.gu = nn.ModuleList()
        self.gv = nn.ModuleList()
        self.h = nn.ModuleList()
        self.f = nn.ModuleList()
        
        self.gu.append(pt_utils.SharedMLP(gu))
        self.gv.append(pt_utils.SharedMLP(gv))
        self.h.append(pt_utils.SharedMLP(h))
        self.f.append(pt_utils.SharedMLP(f))
        
    def forward(self, xyz, features=None):
        '''
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, C, N) tensor of the descriptors of the the features
        '''
        features = features.transpose(1, 2)
#        print(features.shape)
        point_num = xyz.shape[1]
        relation_xyz = []
        relation_feature = []
        for i in range(point_num):
            choice = np.random.choice(point_num, self.npoints, replace=False)
#            choice = torch.from_numpy(choice).expand(xyz.shape[0],xyz.shape[2],self.npoints)
#            choice = choice.transpose(1, 2)
#            print(choice.shape)
            xyz_neighbour = torch.gather(xyz, 1, torch.from_numpy(choice).expand(xyz.shape[0],xyz.shape[2],self.npoints).transpose(1, 2).cuda()) # B,npoint,3
#            print(xyz_neighbour.shape)
            feature_neighbour = torch.gather(features, 1, torch.from_numpy(choice).expand(features.shape[0],features.shape[2],self.npoints).transpose(1, 2).cuda()) # B,npoint,C
#            print(feature_neighbour.shape)
#            catxyz = xyz[:,i,:].expand(self.npoints,xyz.shape[0],xyz.shape[2]).transpose(0, 1)
#            print(catxyz.shape)
            new_xyz = torch.cat((xyz[:,i,:].expand(self.npoints,xyz.shape[0],xyz.shape[2]).transpose(0, 1), xyz_neighbour), dim=-1) # B,npoint,3+3
#            print(new_xyz.shape)
            new_feature = feature_neighbour + features[:,i,:].unsqueeze(1) # B,npoint,C
#            print(new_feature.shape)
            relation_xyz.append(new_xyz)
            relation_feature.append(new_feature)
        relation_xyz =torch.stack(relation_xyz, dim=-1).transpose(1,2) # B,6,npoints,N
#        print(relation_xyz.shape)
        relation_feature =torch.stack(relation_feature, dim=-1).transpose(1,2) # B,C,npoints,N
#        print(relation_feature.shape)
        
        gu_output = self.gu[0](relation_xyz) # B,6,npoints,N
#        print(gu_output.shape)
        gv_output = self.gv[0](relation_feature) # B,2C,npoints,N
#        print(gv_output.shape)
        
        fuse_uv = torch.cat((gu_output, gv_output),dim=1)# B,2C+6,npoints,N
#        print(fuse_uv.shape)
        h_output = self.h[0](fuse_uv) # B,C,npoints,N
#        print(h_output.shape)
        unsqueeze_h = torch.mean(h_output, dim=-2) # B,C,N
#        print(unsqueeze_h.shape)
        output = self.f[0](unsqueeze_h.unsqueeze(2)).squeeze(-2) # B,C,N
#        print(output.shape)
        return xyz, output+features.transpose(1, 2)
    
    
class _PointnetSAModuleBase(nn.Module):
    def __init__(self):
        super(_PointnetSAModuleBase, self).__init__()
        self.npoint = None
        self.groupers = None
        self.mlps = None

    def forward(self, xyz, features=None):
        # type: (_PointnetSAModuleBase, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, N, C) tensor of the descriptors of the the features

        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B,  \sum_k(mlps[k][-1]), npoint) tensor of the new_features descriptors
        """

        new_features_list = []

        xyz_flipped = xyz.transpose(1, 2).contiguous()
        new_xyz = (
            pointnet2_utils.gather_operation(
                xyz_flipped, pointnet2_utils.furthest_point_sample(xyz, self.npoint)
            )
            .transpose(1, 2)
            .contiguous()
            if self.npoint is not None
            else None
        )

        for i in range(len(self.groupers)):
            new_features = self.groupers[i](
                xyz, new_xyz, features
            )  # (B, C, npoint, nsample)

            new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
            new_features = F.max_pool2d(
                new_features, kernel_size=[1, new_features.size(3)]
            )  # (B, mlp[-1], npoint, 1)
            new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)

            new_features_list.append(new_features)

        return new_xyz, torch.cat(new_features_list, dim=1)


class PointnetSAModuleMSG(_PointnetSAModuleBase):
    r"""Pointnet set abstrction layer with multiscale grouping

    Parameters
    ----------
    npoint : int
        Number of features
    radii : list of float32
        list of radii to group with
    nsamples : list of int32
        Number of samples in each ball query
    mlps : list of list of int32
        Spec of the pointnet before the global max_pool for each scale
    bn : bool
        Use batchnorm
    """

    def __init__(self, npoint, radii, nsamples, mlps, bn=True, use_xyz=True):
        # type: (PointnetSAModuleMSG, int, List[float], List[int], List[List[int]], bool, bool) -> None
        super(PointnetSAModuleMSG, self).__init__()

        assert len(radii) == len(nsamples) == len(mlps)

        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz)
                if npoint is not None
                else pointnet2_utils.GroupAll(use_xyz)
            )
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            self.mlps.append(pt_utils.SharedMLP(mlp_spec, bn=bn))


class PointnetSAModule(PointnetSAModuleMSG):
    r"""Pointnet set abstrction layer

    Parameters
    ----------
    npoint : int
        Number of features
    radius : float
        Radius of ball
    nsample : int
        Number of samples in the ball query
    mlp : list
        Spec of the pointnet before the global max_pool
    bn : bool
        Use batchnorm
    """

    def __init__(
        self, mlp, npoint=None, radius=None, nsample=None, bn=True, use_xyz=True
    ):
        # type: (PointnetSAModule, List[int], int, float, int, bool, bool) -> None
        super(PointnetSAModule, self).__init__(
            mlps=[mlp],
            npoint=npoint,
            radii=[radius],
            nsamples=[nsample],
            bn=bn,
            use_xyz=use_xyz,
        )


class PointnetFPModule(nn.Module):
    r"""Propigates the features of one set to another

    Parameters
    ----------
    mlp : list
        Pointnet module parameters
    bn : bool
        Use batchnorm
    """

    def __init__(self, mlp, bn=True):
        # type: (PointnetFPModule, List[int], bool) -> None
        super(PointnetFPModule, self).__init__()
        self.mlp = pt_utils.SharedMLP(mlp, bn=bn)

    def forward(self, unknown, known, unknow_feats, known_feats):
        # type: (PointnetFPModule, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of the xyz positions of the unknown features
        known : torch.Tensor
            (B, m, 3) tensor of the xyz positions of the known features
        unknow_feats : torch.Tensor
            (B, C1, n) tensor of the features to be propigated to
        known_feats : torch.Tensor
            (B, C2, m) tensor of features to be propigated

        Returns
        -------
        new_features : torch.Tensor
            (B, mlp[-1], n) tensor of the features of the unknown features
        """

        if known is not None:
            dist, idx = pointnet2_utils.three_nn(unknown, known)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            interpolated_feats = pointnet2_utils.three_interpolate(
                known_feats, idx, weight
            )
        else:
            interpolated_feats = known_feats.expand(
                *(known_feats.size()[0:2] + [unknown.size(1)])
            )

        if unknow_feats is not None:
            new_features = torch.cat(
                [interpolated_feats, unknow_feats], dim=1
            )  # (B, C2 + C1, n)
        else:
            new_features = interpolated_feats

        new_features = new_features.unsqueeze(-1)
        new_features = self.mlp(new_features)

        return new_features.squeeze(-1)


if __name__ == "__main__":
    from torch.autograd import Variable

    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    xyz = Variable(torch.randn(2, 9, 3).cuda(), requires_grad=True)
    xyz_feats = Variable(torch.randn(2, 9, 6).cuda(), requires_grad=True)

    test_module = PointnetSAModuleMSG(
        npoint=2, radii=[5.0, 10.0], nsamples=[6, 3], mlps=[[9, 3], [9, 6]]
    )
    test_module.cuda()
    print(test_module(xyz, xyz_feats))

    #  test_module = PointnetFPModule(mlp=[6, 6])
    #  test_module.cuda()
    #  from torch.autograd import gradcheck
    #  inputs = (xyz, xyz, None, xyz_feats)
    #  test = gradcheck(test_module, inputs, eps=1e-6, atol=1e-4)
    #  print(test)

    for _ in range(1):
        _, new_features = test_module(xyz, xyz_feats)
        new_features.backward(torch.cuda.FloatTensor(*new_features.size()).fill_(1))
        print(new_features)
        print(xyz.grad)
