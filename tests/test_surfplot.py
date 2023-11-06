# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for elementary brain surface visualisations
"""
import pytest

import numpy as np
import templateflow.api as tflow

from hyve_examples import (
    get_null400_cifti,
    get_null400_gifti,
    get_poldrack_freesurfer,
)
from hyve.flows import plotdef
from hyve.transforms import (
    surf_from_archive,
    surf_from_freesurfer,
    surf_scalars_from_cifti,
    surf_scalars_from_freesurfer,
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
        parcellate_colormap('parcellation', 'network'),
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
        parcellation_cifti=get_null400_cifti(),
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
        parcellate_colormap('parcellation', 'network'),
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
        parcellation_cifti=get_null400_cifti(),
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
        parcellate_colormap('parcellation', 'modal'),
        vertex_to_face('parcellation'),
        plot_to_image(),
        save_snapshots(
            fname_spec=(
                'scalars-{surfscalars}_hemisphere-{hemisphere}_view-{view}_cmap-modal_mode-face'
            ),
        ),
    )
    parcellation_gifti = get_null400_gifti()
    plot_f(
        template="fsLR",
        load_mask=True,
        parcellation_gifti_left=parcellation_gifti['left'],
        parcellation_gifti_right=parcellation_gifti['right'],
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
        parcellate_colormap('parcellation', 'modal'),
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
        parcellation_cifti=get_null400_cifti(),
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
        parcellate_colormap('parcellation', 'network'),
        vertex_to_face('parcellation'),
        plot_to_html(),
    )
    plot_f(
        template="fsLR",
        load_mask=True,
        parcellation_cifti=get_null400_cifti(),
        surf_projection=['veryinflated'],
        # surf_scalars_boundary_color='black',
        # surf_scalars_boundary_width=5,
        hemisphere=['left', 'right'],
        window_size=(800, 800),
        output_dir='/tmp',
        fname_spec=(
            'scalars-{surfscalars}_hemisphere-{hemisphere}_cmap-network'
        ),
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
        save_snapshots(),
    )
    plot_f(
        template="fsLR",
        load_mask=True,
        parcellation_cifti=get_null400_cifti(),
        gmdensity_nifti=tflow.get(
            template='MNI152NLin2009cAsym',
            suffix='probseg',
            label="GM",
            resolution=2
        ),
        surf_projection=['inflated'],
        surf_scalars_clim=(0.1, 0.9),
        surf_scalars_below_color=(0, 0, 0, 0),
        hemisphere=['left', 'right'],
        output_dir='/tmp',
        fname_spec=(
            'scalars-{surfscalars}_hemisphere-{hemisphere}_view-{view}_parcellation-null'
        ),
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
        parcellation_cifti=get_null400_cifti(),
        noise_parcellated=parcellated,
        surf_projection=['inflated'],
        surf_scalars_clim=(0, 1),
        surf_scalars_cmap='inferno',
        surf_scalars_below_color=(0, 0, 0, 0),
        hemisphere=['left', 'right'],
        output_dir='/tmp',
    )


def test_freesurfer():
    fs = get_poldrack_freesurfer()
    geom_left, morph_left = fs['left']
    geom_right, morph_right = fs['right']
    plot_f = plotdef(
        surf_from_freesurfer(projection='inflated'),
        surf_scalars_from_freesurfer('curv'),
        plot_to_image(),
        save_snapshots(
            fname_spec=(
                'scalars-{surfscalars}_hemisphere-{hemisphere}_view-{view}'
            ),
        ),
    )
    plot_f(
        inflated_left_surf=geom_left,
        inflated_right_surf=geom_right,
        curv_morph_left=morph_left,
        curv_morph_right=morph_right,
        surf_scalars_cmap='RdYlBu_r',
        surf_scalars_clim=(-0.35, 0.35),
        hemisphere=['left', 'right'],
        output_dir='/tmp',
    )
