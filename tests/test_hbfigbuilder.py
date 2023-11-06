# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for figure builder where each scalar dataset is hemispherically
bound
"""
import pytest
import numpy as np

from hyve.flows import plotdef
from hyve.layout import Cell, ColGroupSpec, RowGroupSpec
from hyve.transforms import (
    surf_from_archive,
    surf_scalars_from_array,
    # surf_scalars_from_nifti,
    closest_ortho_camera,
    scalar_focus_camera,
    planar_sweep_camera,
    auto_camera,
    plot_to_image,
    save_figure,
    save_grid,
)


@pytest.mark.ci_unsupported
def test_focused_view_autoselect_hemisphere_groupspec0():
    plot_f = plotdef(
        surf_from_archive(),
        surf_scalars_from_array(
            'gaussiannoise',
            allow_multihemisphere=False,
            is_masked=False,
            apply_mask=True,
        ),
        plot_to_image(),
        scalar_focus_camera(kind='peak'),
        save_figure(
            padding=4,
            canvas_size=(2000, 100),
            canvas_color=(0, 0, 0),
            sort_by=['surfscalars'],
            group_spec = [
                RowGroupSpec(
                    variable='hemisphere',
                ),
                RowGroupSpec(variable='surfscalars'),
            ],
            fname_spec=(
                'scalars-gaussiannoise_view-focused3_page-{page}'
                #'scalars-gaussiannoise_view-focused2_page-{page}'
            )
        ),
    )

    array_left = np.random.randn(10, 32492)
    array_right = np.random.randn(10, 32492)
    plot_f(
        template="fsLR",
        load_mask=True,
        gaussiannoise_array_left=array_left,
        gaussiannoise_array_right=array_right,
        surf_projection='veryinflated',
        surf_scalars_cmap='RdYlBu',
        surf_scalars_below_color='#333333',
        window_size=(400, 250),
        output_dir='/tmp',
    )


@pytest.mark.ci_unsupported
def test_focused_view_autoselect_hemisphere_groupspec1():
    plot_f = plotdef(
        surf_from_archive(),
        surf_scalars_from_array(
            'gaussiannoise',
            allow_multihemisphere=False,
            is_masked=False,
            apply_mask=True,
        ),
        plot_to_image(),
        scalar_focus_camera(kind='peak'),
        save_figure(
            padding=4,
            canvas_size=(200, 1000),
            canvas_color=(0, 0, 0),
            sort_by=['surfscalars'],
            group_spec = [
                RowGroupSpec(
                    variable='hemisphere',
                    max_levels=1,
                ),
                ColGroupSpec(variable='surfscalars'),
            ],
            fname_spec=(
                'scalars-gaussiannoise_view-focused_hemi-{hemisphere}_page-{page}'
            )
        ),
    )

    array_left = np.random.randn(10, 32492)
    array_right = np.random.randn(10, 32492)
    plot_f(
        template="fsLR",
        load_mask=True,
        gaussiannoise_array_left=array_left,
        gaussiannoise_array_right=array_right,
        surf_projection='veryinflated',
        surf_scalars_cmap='RdYlBu',
        surf_scalars_below_color='#333333',
        window_size=(400, 250),
        output_dir='/tmp',
    )


@pytest.mark.ci_unsupported
def test_panoptic_groupspec_nohb():
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
        surf_scalars_from_array(
            'gaussiannoise',
            is_masked=False,
            apply_mask=True,
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
                    max_levels=3,
                ),
            ],
            fname_spec=(
                'scalars-gaussiannoise_view-all_page-{page}'
            ),
        ),
    )

    array_left = np.random.randn(12, 32492)
    array_right = np.random.randn(12, 32492)
    plot_f(
        template='fsLR',
        load_mask=True,
        gaussiannoise_array_left=array_left,
        gaussiannoise_array_right=array_right,
        surf_projection='veryinflated',
        surf_scalars_cmap='RdYlBu',
        surf_scalars_clim='robust',
        surf_scalars_below_color='#333333',
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
def test_focused_view_autoselect_hemisphere():
    plot_f = plotdef(
        surf_from_archive(),
        surf_scalars_from_array(
            'gaussiannoise',
            allow_multihemisphere=False,
            is_masked=False,
            apply_mask=True,
        ),
        plot_to_image(),
        scalar_focus_camera(kind='peak'),
        save_grid(
            n_cols=4, n_rows=5, padding=4,
            canvas_size=(1280, 1024),
            canvas_color=(0, 0, 0),
            sort_by=['surfscalars'],
            fname_spec=(
                'scalars-gaussiannoise_view-focused_page-{page}'
            )
        ),
    )

    array_left = np.random.randn(10, 32492)
    array_right = np.random.randn(10, 32492)
    plot_f(
        template="fsLR",
        load_mask=True,
        gaussiannoise_array_left=array_left,
        gaussiannoise_array_right=array_right,
        surf_projection='veryinflated',
        surf_scalars_cmap='RdYlBu',
        surf_scalars_below_color='#333333',
        window_size=(400, 250),
        output_dir='/tmp',
    )

@pytest.mark.ci_unsupported
def test_ortho_views_autoselect_hemisphere():
    selected = list(range(19)) + list(range(20, 39))
    plot_f = plotdef(
        surf_from_archive(),
        surf_scalars_from_array(
            'gaussiannoise',
            allow_multihemisphere=False,
            is_masked=False,
            apply_mask=True,
            select=selected,
        ),
        plot_to_image(),
        closest_ortho_camera(n_ortho=3),
        save_grid(
            n_cols=3, n_rows=10, padding=4,
            canvas_size=(960, 2048),
            canvas_color=(0, 0, 0),
            sort_by=['surfscalars'],
            fname_spec=(
                'scalars-gaussiannoise_view-ortho_page-{page}'
            )
        ),
    )

    array_left = np.random.randn(10, 32492)
    array_right = np.random.randn(10, 32492)
    plot_f(
        template="fsLR",
        load_mask=True,
        gaussiannoise_array_left=array_left,
        gaussiannoise_array_right=array_right,
        surf_projection='veryinflated',
        surf_scalars_cmap='RdYlBu',
        surf_scalars_below_color='#333333',
        window_size=(400, 250),
        output_dir='/tmp',
    )

@pytest.mark.ci_unsupported
def test_planar_sweep_autoselect_hemisphere():
    plot_f = plotdef(
        surf_from_archive(),
        surf_scalars_from_array(
            'gaussiannoise',
            allow_multihemisphere=False,
            is_masked=False,
            apply_mask=True,
        ),
        plot_to_image(),
        planar_sweep_camera(initial=(1, 0, 0), normal=(0, 0, 1), n_steps=10),
        save_grid(
            n_cols=10, n_rows=5, padding=4,
            canvas_size=(3200, 1024),
            canvas_color=(0, 0, 0),
            sort_by=['surfscalars'],
            fname_spec=(
                'scalars-gaussiannoise_view-planar_page-{page}'
            )
        ),
    )

    array_left = np.random.randn(10, 32492)
    array_right = np.random.randn(10, 32492)
    plot_f(
        template="fsLR",
        load_mask=True,
        gaussiannoise_array_left=array_left,
        gaussiannoise_array_right=array_right,
        surf_projection='veryinflated',
        surf_scalars_cmap='RdYlBu',
        surf_scalars_below_color='#333333',
        window_size=(400, 250),
        output_dir='/tmp',
    )

@pytest.mark.ci_unsupported
def test_auto_view_autoselect_hemisphere():
    plot_f = plotdef(
        surf_from_archive(),
        surf_scalars_from_array(
            'gaussiannoise',
            allow_multihemisphere=False,
            is_masked=False,
            apply_mask=True,
        ),
        plot_to_image(),
        auto_camera(n_ortho=3, focus='peak', n_angles=3),
        save_grid(
            n_cols=7, n_rows=10, padding=4,
            canvas_size=(2240, 2048),
            canvas_color=(0, 0, 0),
            sort_by=['surfscalars'],
            fname_spec=(
                'scalars-gaussiannoise_view-auto_page-{page}'
            )
        ),
    )

    array_left = np.random.randn(10, 32492)
    array_right = np.random.randn(10, 32492)
    plot_f(
        template="fsLR",
        load_mask=True,
        gaussiannoise_array_left=array_left,
        gaussiannoise_array_right=array_right,
        surf_projection='veryinflated',
        surf_scalars_cmap='RdYlBu',
        surf_scalars_below_color='#333333',
        window_size=(400, 250),
        output_dir='/tmp',
    )
