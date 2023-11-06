# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for figure builder mapping over both hemispheres
"""
import pytest
import pyvista as pv
import templateflow.api as tflow

from hyve.flows import plotdef
from hyve.layout import Cell, ColGroupSpec
from hyve.transforms import (
    surf_from_archive,
    surf_scalars_from_nifti,
    closest_ortho_camera,
    scalar_focus_camera,
    planar_sweep_camera,
    auto_camera,
    plot_to_image,
    save_grid,
    save_figure,
)


@pytest.mark.ci_unsupported
def test_panoptic():
    layout = Cell() | Cell() << (1 / 2)
    layout = layout * layout * layout
    annotations = {
        0: dict(
            hemisphere='left',
            view='lateral',
        ),
        1: dict(
            hemisphere='left',
            view='medial',
        ),
        2: dict(
            hemisphere='right',
            view='lateral',
        ),
        3: dict(
            hemisphere='right',
            view='medial',
        ),
        4: dict(
            view='dorsal',
        ),
        5: dict(
            view='ventral',
        ),
        6: dict(
            view='anterior',
        ),
        7: dict(
            view='posterior',
        ),
    }
    layout = layout.annotate(annotations)
    plot_f = plotdef(
        surf_from_archive(),
        surf_scalars_from_nifti(
            'difumo',
            template='fsaverage',
            select=list(range(60)),
            plot=True,
        ),
        plot_to_image(),
        save_figure(
            canvas_size=(3200, 4800),
            canvas_color=(0, 0, 0),
            sort_by=['surfscalars'],
            layout_kernel=layout,
            group_spec = [
                ColGroupSpec(
                    variable='surfscalars',
                    max_levels=16,
                ),
            ],
            fname_spec=(
                'scalars-difumo_view-all_page-{page}'
            ),
        ),
    )
    nifti = tflow.get(
        template='MNI152NLin2009cAsym',
        atlas='DiFuMo',
        resolution=2,
        desc='64dimensions'
    )
    plot_f(
        template='fsaverage',
        load_mask=True,
        difumo_nifti=nifti,
        surf_projection='pial',
        surf_scalars_cmap='viridis',
        surf_scalars_clim='robust',
        surf_scalars_below_color='#666666',
        #theme=pv.themes.DarkTheme(),
        window_size=(400, 300),
        hemisphere=['left', 'right', None],
        views={
            'left': ('medial', 'lateral'),
            'right': ('medial', 'lateral'),
            'both': ('dorsal', 'ventral', 'anterior', 'posterior'),
        },
        output_dir='/tmp',
    )


@pytest.mark.ci_unsupported
def test_focused_view_both_hemispheres():
    plot_f = plotdef(
        surf_from_archive(),
        surf_scalars_from_nifti(
            'difumo',
            template='fsaverage',
            select=list(range(60)),
            plot=True
        ),
        plot_to_image(),
        scalar_focus_camera(kind='centroid'),
        save_grid(
            n_cols=4, n_rows=8, padding=4,
            canvas_size=(1280, 1640),
            canvas_color=(0, 0, 0),
            fname_spec=(
                'scalars-difumo_view-focused_page-{page}'
            )
        ),
    )
    nifti = tflow.get(
        template='MNI152NLin2009cAsym',
        atlas='DiFuMo',
        resolution=2,
        desc='64dimensions'
    )
    plot_f(
        template='fsaverage',
        load_mask=True,
        difumo_nifti=nifti,
        surf_projection='pial',
        surf_scalars_cmap='viridis',
        surf_scalars_below_color='#333333',
        window_size=(400, 250),
        output_dir='/tmp',
    )

@pytest.mark.ci_unsupported
def test_ortho_views_both_hemispheres():
    plot_f = plotdef(
        surf_from_archive(),
        surf_scalars_from_nifti('difumo', template='fsaverage', plot=True),
        plot_to_image(),
        closest_ortho_camera(n_ortho=3),
        save_grid(
            n_cols=3, n_rows=8, padding=4,
            canvas_size=(960, 1640),
            canvas_color=(0, 0, 0),
            fname_spec=(
                'scalars-difumo_view-ortho_page-{page}'
            )
        ),
    )
    nifti = tflow.get(
        template='MNI152NLin2009cAsym',
        atlas='DiFuMo',
        resolution=2,
        desc='64dimensions'
    )
    plot_f(
        template='fsaverage',
        load_mask=True,
        difumo_nifti=nifti,
        surf_projection='pial',
        surf_scalars_cmap='Purples',
        surf_scalars_below_color='white',
        window_size=(400, 250),
        output_dir='/tmp',
        #hemisphere=['left', 'right'],
    )

@pytest.mark.ci_unsupported
def test_planar_sweep_both_hemispheres():
    plot_f = plotdef(
        surf_from_archive(),
        surf_scalars_from_nifti('difumo', template='fsaverage', plot=True),
        plot_to_image(),
        planar_sweep_camera(initial=(1, 0, 0), n_steps=10),
        save_grid(
            n_cols=10, n_rows=8, padding=4,
            canvas_size=(3200, 1640),
            canvas_color=(0, 0, 0),
            fname_spec=(
                'scalars-difumo_view-planar_page-{page}'
            )
        ),
    )
    nifti = tflow.get(
        template='MNI152NLin2009cAsym',
        atlas='DiFuMo',
        resolution=2,
        desc='64dimensions'
    )
    plot_f(
        template='fsaverage',
        load_mask=True,
        difumo_nifti=nifti,
        surf_projection='pial',
        surf_scalars_cmap='Purples',
        surf_scalars_below_color='white',
        window_size=(400, 250),
        output_dir='/tmp',
        hemisphere=['left', 'right'],
    )

@pytest.mark.ci_unsupported
def test_auto_view_both_hemispheres():
    plot_f = plotdef(
        surf_from_archive(),
        surf_scalars_from_nifti('difumo', template='fsaverage', plot=True),
        plot_to_image(),
        auto_camera(n_ortho=3, focus='peak', n_angles=3),
        save_grid(
            n_cols=7, n_rows=8, padding=4,
            canvas_size=(2240, 1640),
            canvas_color=(0, 0, 0),
            fname_spec=(
                'scalars-difumo_view-auto_page-{page}'
            )
        ),
    )
    nifti = tflow.get(
        template='MNI152NLin2009cAsym',
        atlas='DiFuMo',
        resolution=2,
        desc='64dimensions'
    )
    plot_f(
        template='fsaverage',
        load_mask=True,
        difumo_nifti=nifti,
        surf_projection='pial',
        surf_scalars_cmap='Purples',
        surf_scalars_below_color='white',
        window_size=(400, 250),
        output_dir='/tmp',
        hemisphere=['left', 'right'],
    )
