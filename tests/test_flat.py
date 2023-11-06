# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for flat map visualisations loaded from a GIfTI file
"""
import pytest

import templateflow.api as tflow

from hyve_examples import (
    get_null400_cifti,
    get_fsLR_flatmap_gifti,
)
from hyve.flows import plotdef
from hyve.transforms import (
    surf_from_gifti,
    surf_scalars_from_cifti,
    surf_scalars_from_nifti,
    parcellate_colormap,
    vertex_to_face,
    plot_to_image,
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
surf = get_fsLR_flatmap_gifti()


@pytest.mark.ci_unsupported
def test_scalars():
    plot_f = plotdef(
        surf_from_gifti(projection='flat'),
        surf_scalars_from_nifti('GM Density', template='fsLR'),
        plot_to_image(),
        save_snapshots(
            fname_spec=(
                'scalars-{surfscalars}_hemisphere-{hemisphere}_projection-flat'
            ),
        ),
    )
    plot_f(
        flat_left_surf=surf['left'],
        flat_right_surf=surf['right'],
        flat_left_mask=lh_mask,
        flat_right_mask=rh_mask,
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


@pytest.mark.parametrize(
    'cmap, cmap_name', [
        ('network', 'network'),
        ('modal', 'modal'),
        ('network', 'bone')
    ])
def test_parcellation(cmap, cmap_name):
    plot_f = plotdef(
        surf_from_gifti(projection='flat'),
        surf_scalars_from_cifti('parcellation', plot=True),
        parcellate_colormap('parcellation', cmap),
        vertex_to_face('parcellation'),
        plot_to_image(),
        save_snapshots(
            fname_spec=(
                'scalars-{surfscalars}_hemisphere-{hemisphere}_'
                f'cmap-{cmap_name}_projection-flat'
            ),
        ),
    )
    plot_f(
        flat_left_surf=surf['left'],
        flat_right_surf=surf['right'],
        flat_left_mask=lh_mask,
        flat_right_mask=rh_mask,
        parcellation_cifti=get_null400_cifti(),
        surf_scalars_cmap=cmap_name,
        hemisphere=['left', 'right'],
        views=['down'],
        output_dir='/tmp',
    )
