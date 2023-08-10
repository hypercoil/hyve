# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for plots that contain multiple overlays
"""
import pytest

import nibabel as nb
import numpy as np

from hyve_examples import (
    get_null400_cifti,
    get_pain_thresh_nifti,
)
from hyve.flows import plotdef
from hyve.transforms import (
    surf_from_archive,
    surf_scalars_from_nifti,
    surf_scalars_from_cifti,
    points_scalars_from_array,
    plot_to_image,
    plot_to_html,
    parcellate_colormap,
    save_snapshots,
    add_surface_overlay,
    add_points_overlay,
    vertex_to_face,
)


def create_sphere(radius=30, inner_radius=25, inmost_radius=20):
    sphere_bounds = np.arange(-radius, radius, 3)
    sphere_coor = np.concatenate([
        c.reshape(1, -1) for c in
        np.meshgrid(sphere_bounds, sphere_bounds, sphere_bounds)
    ]).T
    coor_radius = np.sqrt((sphere_coor ** 2).sum(-1))
    sphere_index = coor_radius < radius
    coor_radius = coor_radius[sphere_index]
    sphere_coor = sphere_coor[sphere_index]
    sphere_inner_index = coor_radius < inner_radius
    sphere_data = 1 + ((coor_radius - inner_radius) / (radius - inner_radius))
    sphere_data[sphere_inner_index] = -(
        1 + ((coor_radius[sphere_inner_index] - inmost_radius) / (
            inner_radius - inmost_radius
        )))
    sphere_inmost_index = coor_radius < inmost_radius
    sphere_data[sphere_inmost_index] = np.random.randn(
        sphere_inmost_index.sum())
    return sphere_coor, sphere_data


@pytest.mark.parametrize('output', ['image', 'html'])
@pytest.mark.parametrize('v2f', [True, False])
def test_parcellation_modal_cmap(output, v2f):
    fname_spec = 'scalars-{surfscalars}_hemisphere-{hemisphere}'
    if v2f:
        v2f_transform_sequence = (
            [vertex_to_face('parcellation')],
            [vertex_to_face('pain')],
        )
        fname_spec += '_mode-face'
    else:
        v2f_transform_sequence = ([], [])
        fname_spec += '_mode-vertex'
    if output == 'image':
        fname_spec += '_view-{view}'
        out_transform_sequence = [
            plot_to_image(),
            save_snapshots(fname_spec=fname_spec),
        ]
    elif output == 'html':
        out_transform_sequence = [
            plot_to_html(fname_spec=fname_spec),
        ]
    sphere_coor, sphere_data = create_sphere()

    plot_f = plotdef(
        surf_from_archive(),
        add_surface_overlay(
            'parcellation',
            surf_scalars_from_cifti('parcellation'),
            parcellate_colormap('modal', 'parcellation'),
            *v2f_transform_sequence[0],
        ),
        add_surface_overlay(
            'pain',
            surf_scalars_from_nifti('pain', template='fsLR'),
            *v2f_transform_sequence[1],
        ),
        add_points_overlay(
            'sphere',
            points_scalars_from_array('sphere', point_size=8),
        ),
        *out_transform_sequence,
    )

    plot_f(
        template="fsLR",
        load_mask=True,
        parcellation_cifti=get_null400_cifti(),
        pain_nifti=nb.load(get_pain_thresh_nifti()),
        pain_cmap='inferno',
        pain_clim='robust',
        pain_alpha=0.5,
        pain_below_color=(0, 0, 0, 0),
        sphere_coor=sphere_coor,
        sphere_values=sphere_data,
        sphere_cmap='Reds',
        sphere_cmap_negative='Blues',
        sphere_clim=(1.0, 2.0),
        sphere_below_color=(0, 0, 0, 0),
        surf_projection=('veryinflated',),
        # surf_scalars_boundary_color='black',
        # surf_scalars_boundary_width=5,
        hemisphere=['left', 'right'],
        output_dir='/tmp',
    )
