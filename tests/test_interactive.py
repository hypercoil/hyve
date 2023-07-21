# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Tests that require an interactive plotter display window
"""
import pytest
import templateflow.api as tflow
from conveyant import ichain
from hyve.prim import automap_unified_plotter_p
from hyve.transforms import (
    plot_to_display,
    plot_final_image,
    save_snapshots,
    surf_from_archive,
    scalars_from_nifti,
    resample_to_surface,
)


@pytest.mark.ci_unsupported
def test_plotter_flow_syntax():
    chain = ichain(
        surf_from_archive(),
        resample_to_surface('gmdensity', template='fsaverage'),
        scalars_from_nifti('pain'),
        plot_to_display(),
    )
    plot_f = chain(automap_unified_plotter_p)
    plot_f(
        template='fsaverage',
        load_mask=True,
        gmdensity_nifti=tflow.get(
            template='MNI152NLin2009cAsym',
            suffix='probseg',
            label='GM',
            resolution=2
        ),
        pain_nifti='/Users/rastkociric/Downloads/pain_thresh_cFWE05.nii.gz',
        surf_projection=('pial',),
        surf_alpha=0.5,
    )


@pytest.mark.ci_unsupported
def test_plotter_final_capture():
    chain = ichain(
        surf_from_archive(),
        resample_to_surface('gmdensity', template='fsaverage'),
        scalars_from_nifti('pain'),
        plot_final_image(n_scenes=1), # n_scenes > 1 is not supported yet
        save_snapshots(
            fname_spec=(
                'scalars-{scalars}_hemisphere-{hemisphere}_view-{view}'
            ),
        ),
    )
    plot_f = chain(automap_unified_plotter_p)
    plot_f(
        template='fsaverage',
        load_mask=True,
        gmdensity_nifti=tflow.get(
            template='MNI152NLin2009cAsym',
            suffix='probseg',
            label='GM',
            resolution=2
        ),
        pain_nifti='/Users/rastkociric/Downloads/pain_thresh_cFWE05.nii.gz',
        surf_projection=('pial',),
        surf_alpha=0.5,
        output_dir='/tmp',
    )
