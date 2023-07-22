# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for plots that contain multiple overlays
"""
import pytest
from pkg_resources import resource_filename

import nibabel as nb

from conveyant import (
    ichain,
)
from hyve.prim import automap_unified_plotter_p
from hyve.transforms import (
    surf_from_archive,
    resample_to_surface,
    plot_to_image,
    plot_to_html,
    scalars_from_cifti,
    parcellate_colormap,
    save_snapshots,
    add_surface_overlay,
    vertex_to_face,
)


@pytest.mark.parametrize('output', ['image', 'html'])
@pytest.mark.parametrize('v2f', [True, False])
def test_parcellation_modal_cmap(output, v2f):
    fname_spec = 'scalars-{scalars}_hemisphere-{hemisphere}'
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

    plot_f = ichain(
        surf_from_archive(),
        add_surface_overlay(
            scalars_from_cifti('parcellation'),
            parcellate_colormap('modal', 'parcellation'),
            *v2f_transform_sequence[0],
        ),
        add_surface_overlay(
            resample_to_surface('pain', template='fsLR'),
            *v2f_transform_sequence[1],
        ),
        *out_transform_sequence,
    )(automap_unified_plotter_p)

    plot_f(
        template="fsLR",
        load_mask=True,
        parcellation_cifti=resource_filename(
            'hyve',
            'data/examples/nullexample.nii'
        ),
        pain_nifti=nb.load('/Users/rastkociric/Downloads/pain_thresh_cFWE05.nii.gz'),
        pain_cmap='inferno',
        pain_clim='robust',
        pain_alpha=0.5,
        surf_projection=('veryinflated',),
        # surf_scalars_boundary_color='black',
        # surf_scalars_boundary_width=5,
        hemisphere=['left', 'right'],
        output_dir='/tmp',
    )
