# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for flat map visualisations loaded from a GIfTI file
"""
import pytest

from pkg_resources import resource_filename as pkgrf

import numpy as np
import templateflow.api as tflow

from hyve.flows import plotdef
from hyve.transforms import (
    surf_from_gifti,
    surf_scalars_from_cifti,
    surf_scalars_from_nifti,
    parcellate_colormap,
    vertex_to_face,
    plot_to_image,
    plot_to_display,
    plot_to_html,
    save_snapshots,
)


lh_mask = tflow.get(
    template='fsLR',
    hemi='L',
    desc='nomedialwall',
    density='32k',
)
rh_mask = tflow.get(
    template='fsLR',
    hemi='R',
    desc='nomedialwall',
    density='32k',
)


@pytest.mark.ci_unsupported
def test_scalars():
    plot_f = plotdef(
        surf_from_gifti(projection='flat'),
        surf_scalars_from_nifti('GM Density', template='fsLR'),
        # plot_to_display(),
        plot_to_image(),
        save_snapshots(
            fname_spec=(
                'scalars-{surfscalars}_hemisphere-{hemisphere}_projection-flat'
            ),
        ),
        # plot_to_html(
        #     fname_spec=(
        #         'scalars-{surfscalars}_hemisphere-{hemisphere}_projection-flat'
        #     ),
        # ),
    )
    plot_f(
        left_surf='/Users/rastkociric/Downloads/S1200.L.flat.32k_fs_LR.surf.gii',
        right_surf='/Users/rastkociric/Downloads/S1200.R.flat.32k_fs_LR.surf.gii',
        left_mask=lh_mask,
        right_mask=rh_mask,
        gm_density_nifti=tflow.get(
            template='MNI152NLin2009cAsym',
            suffix='probseg',
            label='GM',
            resolution=2,
        ),
        hemisphere=['left', 'right'],
        views=['down'],
        output_dir='/tmp',
    )


@pytest.mark.parametrize('cmap', ['network', 'modal'])
def test_parcellation(cmap):
    plot_f = plotdef(
        surf_from_gifti(projection='flat'),
        surf_scalars_from_cifti('parcellation', plot=True),
        parcellate_colormap(cmap, 'parcellation'),
        vertex_to_face('parcellation'),
        plot_to_image(),
        save_snapshots(
            fname_spec=(
                'scalars-{surfscalars}_hemisphere-{hemisphere}_'
                f'cmap-{cmap}_projection-flat'
            ),
        ),
    )
    plot_f(
        left_surf='/Users/rastkociric/Downloads/S1200.L.flat.32k_fs_LR.surf.gii',
        right_surf='/Users/rastkociric/Downloads/S1200.R.flat.32k_fs_LR.surf.gii',
        left_mask=lh_mask,
        right_mask=rh_mask,
        parcellation_cifti=pkgrf(
            'hyve',
            'data/examples/nullexample.nii'
        ),
        hemisphere=['left', 'right'],
        views=['down'],
        output_dir='/tmp',
    )
