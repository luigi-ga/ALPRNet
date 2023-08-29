
from __future__ import absolute_import

import numpy as np
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

def grid_sample(input, grid, canvas = None):
    output = F.grid_sample(input, grid, align_corners=True)
    if canvas is None:
        return output
    else:
        input_mask = torch.ones_like(input)
        output_mask = F.grid_sample(input_mask, grid, align_corners=True)
        padded_output = output * output_mask + canvas * (1 - output_mask)
        return padded_output

def compute_partial_repr(input_points, control_points):
    pairwise_diff = input_points[:, None] - control_points[None, :]
    pairwise_dist = torch.sum(pairwise_diff ** 2, dim=2)
    repr_matrix = 0.5 * pairwise_dist * torch.log(pairwise_dist + 1e-6)
    repr_matrix[torch.isnan(repr_matrix)] = 0
    return repr_matrix

def build_output_control_points(num_control_points, margins):
    margin_x, margin_y = margins
    ctrl_pts_x = np.linspace(margin_x, 1.0 - margin_x, num_control_points // 2)
    ctrl_pts_y_top = np.full(num_control_points // 2, margin_y)
    ctrl_pts_y_bottom = np.full(num_control_points // 2, 1.0 - margin_y)
    ctrl_pts_top = np.column_stack([ctrl_pts_x, ctrl_pts_y_top])
    ctrl_pts_bottom = np.column_stack([ctrl_pts_x, ctrl_pts_y_bottom])
    output_ctrl_pts_arr = np.vstack([ctrl_pts_top, ctrl_pts_bottom])
    output_ctrl_pts = torch.Tensor(output_ctrl_pts_arr)
    return output_ctrl_pts


class TPSSpatialTransformer(nn.Module):

    def __init__(self, output_image_size=None, num_control_points=None, margins=None):
        super(TPSSpatialTransformer, self).__init__()
        self.output_image_size = output_image_size
        self.num_control_points = num_control_points
        self.margins = margins

        self.target_height, self.target_width = output_image_size
        target_control_points = build_output_control_points(num_control_points, margins)
        N = num_control_points

        forward_kernel = torch.zeros(N + 3, N + 3)
        target_control_partial_repr = compute_partial_repr(target_control_points, target_control_points)
        forward_kernel[:N, :N].copy_(target_control_partial_repr)
        forward_kernel[:N, -3].fill_(1)
        forward_kernel[-3, :N].fill_(1)
        forward_kernel[:N, -2:].copy_(target_control_points)
        forward_kernel[-2:, :N].copy_(target_control_points.transpose(0, 1))

        # Compute inverse matrix
        inverse_kernel = torch.inverse(forward_kernel)

        # Create target cordinate matrix
        HW = self.target_height * self.target_width
        Y, X = torch.meshgrid([torch.linspace(0, 1, self.target_height), torch.linspace(0, 1, self.target_width)], indexing='ij')
        target_coordinate = torch.stack((X.flatten(), Y.flatten()), dim=1)
        target_coordinate_partial_repr = compute_partial_repr(target_coordinate, target_control_points)
        target_coordinate_repr = torch.cat([
            target_coordinate_partial_repr, torch.ones(HW, 1), target_coordinate
        ], dim=1)   

        # Register precomputed matrices
        self.register_buffer('inverse_kernel', inverse_kernel)
        self.register_buffer('padding_matrix', torch.zeros(3, 2))
        self.register_buffer('target_coordinate_repr', target_coordinate_repr)
        self.register_buffer('target_control_points', target_control_points)

    def forward(self, input, source_control_points):
        assert source_control_points.ndimension() == 3
        assert source_control_points.size(1) == self.num_control_points
        assert source_control_points.size(2) == 2
        batch_size = source_control_points.size(0)

        Y = torch.cat([source_control_points, self.padding_matrix.expand(batch_size, 3, 2)], 1)
        mapping_matrix = torch.matmul(self.inverse_kernel, Y)
        source_coordinate = torch.matmul(self.target_coordinate_repr, mapping_matrix)

        grid = source_coordinate.view(-1, self.target_height, self.target_width, 2)
        grid = torch.clamp(grid, 0, 1)
        grid = 2.0 * grid - 1.0
        output_maps = grid_sample(input, grid, canvas=None)
        return output_maps, source_coordinate
