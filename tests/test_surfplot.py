# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for elementary brain surface visualisations
"""
import pytest

from pkg_resources import resource_filename as pkgrf

import numpy as np
import templateflow.api as tflow

from hyve.flows import plotdef
from hyve.transforms import (
    surf_from_archive,
    surf_scalars_from_cifti,
    surf_scalars_from_gifti,
    surf_scalars_from_nifti,
    parcellate_colormap,
    parcellate_surf_scalars,
    scatter_into_parcels,
    vertex_to_face,
    plot_to_html,
    plot_to_image,
    save_snapshots,
)


def print_params(**params):
    print(params)
    assert 0


@pytest.mark.ci_unsupported
def test_scalars():
    plot_f = plotdef(
        surf_from_archive(),
        surf_scalars_from_nifti('gmdensity', template='fsaverage', plot=True),
        plot_to_image(),
        save_snapshots(
            fname_spec=(
                'scalars-{surfscalars}_hemisphere-{hemisphere}_view-{view}'
            ),
        ),
    )
    plot_f(
        template='fsaverage',
        load_mask=True,
        gmdensity_nifti=tflow.get(
            template='MNI152NLin2009cAsym',
            suffix='probseg',
            label='GM',
            resolution=2
        ),
        surf_projection=('pial',),
        hemisphere=['left', 'right'],
        output_dir='/tmp',
    )


@pytest.mark.ci_unsupported
def test_parcellation():
    plot_f = plotdef(
        surf_from_archive(),
        surf_scalars_from_cifti('parcellation', plot=True),
        parcellate_colormap('network', 'parcellation'),
        vertex_to_face('parcellation'),
        plot_to_image(),
        save_snapshots(
            fname_spec=(
                'scalars-{surfscalars}_hemisphere-{hemisphere}_view-{view}_cmap-network_mode-face'
            ),
        ),
    )
    plot_f(
        template="fsLR",
        load_mask=True,
        parcellation_cifti=pkgrf(
            'hyve',
            'data/examples/nullexample.nii'
        ),
        surf_projection=('veryinflated',),
        hemisphere=['left', 'right'],
        views=[
            'lateral', 'medial', 'ventral', 'dorsal', 'anterior', 'posterior',
            (-20, 0, 0), ((65, 65, 0), (0, 0, 0), (0, 0, 1))
        ],
        output_dir='/tmp',
    )

    plot_f = plotdef(
        surf_from_archive(),
        surf_scalars_from_cifti('parcellation', plot=True),
        parcellate_colormap('network', 'parcellation'),
        plot_to_image(),
        save_snapshots(
            fname_spec=(
                'scalars-{surfscalars}_hemisphere-{hemisphere}_view-{view}_cmap-network_mode-vertex'
            ),
        ),
    )
    plot_f(
        template="fsLR",
        load_mask=True,
        parcellation_cifti=pkgrf(
            'hyve',
            'data/examples/nullexample.nii'
        ),
        surf_projection=('veryinflated',),
        surf_scalars_boundary_color='black',
        surf_scalars_boundary_width=5,
        hemisphere=['left', 'right'],
        views=[
            'lateral', 'medial', 'ventral', 'dorsal', 'anterior', 'posterior',
            (-20, 0, 0), ((65, 65, 0), (0, 0, 0), (0, 0, 1))
        ],
        output_dir='/tmp',
    )


@pytest.mark.ci_unsupported
def test_parcellation_modal_cmap():
    plot_f = plotdef(
        surf_from_archive(),
        surf_scalars_from_gifti('parcellation', plot=True),
        parcellate_colormap('modal', 'parcellation'),
        vertex_to_face('parcellation'),
        plot_to_image(),
        save_snapshots(
            fname_spec=(
                'scalars-{surfscalars}_hemisphere-{hemisphere}_view-{view}_cmap-modal_mode-face'
            ),
        ),
    )
    plot_f(
        template="fsLR",
        load_mask=True,
        parcellation_gifti_left=pkgrf(
            'hyve',
            'data/examples/nullexample_L.gii'
        ),
        parcellation_gifti_right=pkgrf(
            'hyve',
            'data/examples/nullexample_R.gii'
        ),
        surf_projection=('veryinflated',),
        hemisphere=['left', 'right'],
        views=[
            'lateral', 'medial', 'ventral', 'dorsal', 'anterior', 'posterior',
            (-20, 0, 0), ((65, 65, 0), (0, 0, 0), (0, 0, 1))
        ],
        output_dir='/tmp',
    )

    plot_f = plotdef(
        surf_from_archive(),
        surf_scalars_from_cifti('parcellation', plot=True),
        parcellate_colormap('modal', 'parcellation'),
        plot_to_image(),
        save_snapshots(
            fname_spec=(
                'scalars-{surfscalars}_hemisphere-{hemisphere}_view-{view}_cmap-modal_mode-vertex'
            ),
        ),
    )
    plot_f(
        template="fsLR",
        load_mask=True,
        parcellation_cifti=pkgrf(
            'hyve',
            'data/examples/nullexample.nii'
        ),
        surf_projection=('veryinflated',),
        surf_scalars_boundary_color='black',
        surf_scalars_boundary_width=5,
        hemisphere=['left', 'right'],
        views=[
            'lateral', 'medial', 'ventral', 'dorsal', 'anterior', 'posterior',
            (-20, 0, 0), ((65, 65, 0), (0, 0, 0), (0, 0, 1))
        ],
        output_dir='/tmp',
    )

@pytest.mark.ci_unsupported
def test_parcellation_html():
    plot_f = plotdef(
        surf_from_archive(),
        surf_scalars_from_cifti('parcellation', plot=True),
        parcellate_colormap('network', 'parcellation'),
        vertex_to_face('parcellation'),
        plot_to_html(
            fname_spec=(
                'scalars-{surfscalars}_hemisphere-{hemisphere}_cmap-network'
            ),
        ),
    )
    plot_f(
        template="fsLR",
        load_mask=True,
        parcellation_cifti=pkgrf(
            'hyve',
            'data/examples/nullexample.nii'
        ),
        surf_projection=['veryinflated'],
        # surf_scalars_boundary_color='black',
        # surf_scalars_boundary_width=5,
        hemisphere=['left', 'right'],
        window_size=(800, 800),
        output_dir='/tmp',
    )

@pytest.mark.ci_unsupported
def test_parcellated_scalars():
    plot_f = plotdef(
        surf_from_archive(),
        surf_scalars_from_nifti('gmdensity', template='fsLR', plot=False),
        surf_scalars_from_cifti('parcellation', plot=False),
        parcellate_surf_scalars('gmdensity', 'parcellation'),
        vertex_to_face('gmdensityParcellated', interpolation='mode'),
        plot_to_image(),
        save_snapshots(
            fname_spec=(
                'scalars-{surfscalars}_hemisphere-{hemisphere}_view-{view}_parcellation-null'
            ),
        ),
    )
    plot_f(
        template="fsLR",
        load_mask=True,
        parcellation_cifti=pkgrf(
            'hyve',
            'data/examples/nullexample.nii',
        ),
        gmdensity_nifti=tflow.get(
            template='MNI152NLin2009cAsym',
            suffix='probseg',
            label="GM",
            resolution=2
        ),
        surf_projection=['inflated'],
        surf_scalars_clim=(0.2, 0.9),
        hemisphere=['left', 'right'],
        output_dir='/tmp',
    )

    parcellated = np.random.rand(400)
    plot_f = plotdef(
        surf_from_archive(),
        surf_scalars_from_cifti('parcellation', plot=False),
        scatter_into_parcels('noise', 'parcellation'),
        vertex_to_face('noise', interpolation='mode'),
        plot_to_image(),
        save_snapshots(
            fname_spec=(
                'scalars-{surfscalars}_hemisphere-{hemisphere}_view-{view}_parcellation-null'
            ),
        ),
    )
    plot_f(
        template="fsLR",
        load_mask=True,
        parcellation_cifti=pkgrf(
            'hyve',
            'data/examples/nullexample.nii',
        ),
        noise_parcellated=parcellated,
        surf_projection=['inflated'],
        surf_scalars_clim=(0, 1),
        surf_scalars_cmap='inferno',
        hemisphere=['left', 'right'],
        output_dir='/tmp',
    )
