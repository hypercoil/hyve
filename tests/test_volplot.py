# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for elementary brain volume visualisations
"""
import pytest

from pkg_resources import resource_filename as pkgrf

import nibabel as nb

from hyve.flows import plotdef
from hyve.transforms import (
    surf_from_archive,
    points_scalars_from_nifti,
    plot_to_image,
    save_snapshots,
)


def test_vol_scalars():
    nii = nb.load('/Users/rastkociric/Downloads/pain_thresh_cFWE05.nii.gz')
    plot_f = plotdef(
        surf_from_archive(),
        points_scalars_from_nifti('pain'),
        plot_to_image(),
        save_snapshots(
            fname_spec=(
                'scalars-{pointsscalars}_view-{view}'
            ),
        ),
    )
    plot_f(
        template='fsaverage',
        surf_projection=('pial',),
        surf_alpha=0.3,
        pain_nifti=nii,
        points_scalars_cmap='magma',
        basename='/tmp/vol',
        views=('dorsal', 'left', 'anterior'),
        output_dir='/tmp',
    )
