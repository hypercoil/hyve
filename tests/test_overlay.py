# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for plots that contain multiple overlays
"""
import pytest

import nibabel as nb
import numpy as np
import pyvista as pv

from hyve_examples import (
    get_null400_cifti,
    get_pain_thresh_nifti,
    get_salience_ic_nifti,
    get_schaefer400_cifti,
    get_schaefer400_gifti,
    get_svg_blend,
)
from hyve.elements import TextBuilder
from hyve.flows import plotdef
from hyve.layout import Cell
from hyve.transforms import (
    surf_from_archive,
    surf_scalars_from_nifti,
    surf_scalars_from_cifti,
    surf_scalars_from_gifti,
    points_scalars_from_array,
    plot_to_image,
    plot_to_html,
    auto_camera,
    parcellate_colormap,
    save_snapshots,
    save_grid,
    save_figure,
    svg_element,
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
            parcellate_colormap('parcellation', 'modal'),
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


def test_overlay_allview():
    annotations = {
        0: dict(
            hemisphere='left',
            view='lateral',
        ),
        1: dict(view='anterior'),
        2: dict(
            hemisphere='right',
            view='lateral',
        ),
        3: dict(view='dorsal'),
        4: dict(elements=['title', 'scalar_bar']),
        5: dict(view='ventral'),
        6: dict(
            hemisphere='left',
            view='medial',
        ),
        7: dict(view='posterior'),
        8: dict(
            hemisphere='right',
            view='medial',
        ),
    }
    annotations = {k + 1: v for k, v in annotations.items()}
    layout_r = Cell() | Cell() | Cell() << (1 / 3)
    layout_c = Cell() / Cell() / Cell() << (1 / 3)
    layout = layout_c * layout_r
    layout = Cell() | layout << (3 / 8)
    annotations[0] = dict(elements=['blend_insert'])
    layout = layout.annotate(annotations)
    fname_spec = 'scalars-{surfscalars}_hemisphere-{hemisphere}_mode-face'
    v2f_transform_sequence = (
        [vertex_to_face('parcellation')],
        [vertex_to_face('salience')],
        [vertex_to_face('pain')],
    )
    fname_spec += '_view-{view}'
    out_transform_sequence = [
        plot_to_image(),
        # save_grid(
        #     n_cols=3, n_rows=3, padding=10,
        #     canvas_size=(1800, 1500),
        #     canvas_color=(0, 0, 0),
        #     fname_spec=f'scalars-painoverlay_view-all_page-{{page}}',
        #     scalar_bar_action='collect',
        #     annotations=annotations,
        # ),
        save_figure(
            layout_kernel=layout,
            #padding=10,
            canvas_size=(2880, 1500),
            canvas_color=(0, 0, 0),
            fname_spec=f'scalars-painoverlay_view-all_page-{{page}}',
            scalar_bar_action='collect',
        ),
    ]

    plot_f = plotdef(
        surf_from_archive(),
        add_surface_overlay(
            'parcellation',
            surf_scalars_from_cifti('parcellation'),
            parcellate_colormap('parcellation', 'modal'),
            *v2f_transform_sequence[0],
        ),
        add_surface_overlay(
            'salience',
            surf_scalars_from_nifti('salience', template='fsLR'),
            *v2f_transform_sequence[1],
        ),
        add_surface_overlay(
            'pain',
            surf_scalars_from_nifti('pain', template='fsLR'),
            *v2f_transform_sequence[2],
        ),
        svg_element(
            name='blend_insert',
            src_file=get_svg_blend(),
            height=375,
            width=300,
        ),
        *out_transform_sequence,
    )
    plot_f(
        template="fsLR",
        load_mask=True,
        parcellation_cifti=get_schaefer400_cifti(),
        parcellation_alpha=0.5,
        parcellation_scalar_bar_style={
            'name': 'parcel',
            'orientation': 'h',
        },
        salience_nifti=get_salience_ic_nifti(),
        salience_cmap='magma',
        salience_clim=(3, 8),
        salience_cmap_negative='cool',
        salience_alpha=0.7,
        salience_scalar_bar_style={
            'name': 'salience',
            'orientation': 'h',
        },
        salience_below_color=(0, 0, 0, 0),
        pain_nifti=nb.load(get_pain_thresh_nifti()),
        pain_cmap='Reds',
        pain_clim=(0.3, 8),
        pain_alpha=0.7,
        pain_below_color=(0, 0, 0, 0),
        pain_scalar_bar_style={
            'name': 'zstat',
            'orientation': 'h',
        },
        surf_projection=('veryinflated',),
        hemisphere=['left', 'right', None],
        views={
            'left': ('medial', 'lateral'),
            'right': ('medial', 'lateral'),
            'both': ('dorsal', 'ventral', 'anterior', 'posterior'),
        },
        elements={
            'title': (
                TextBuilder(
                    content='pain',
                    bounding_box_height=192,
                    font_size_multiplier=0.25,
                    font_color='#cccccc',
                    priority=-1,
                ),
            ),
        },
        window_size=(600, 500),
        theme=pv.themes.DarkTheme(),
        output_dir='/tmp',
    )


def test_autocams():
    layout = Cell() | Cell() | Cell() << (1 / 3)
    layout = layout / layout << (1 / 2)
    layout = Cell() | layout << (2 / 5)
    bottom = Cell() | Cell() | Cell() | Cell() | Cell() << (1 / 5)
    layout = layout / bottom << (2 / 3)
    layout = layout | Cell() << (15 / 16)
    annotations = {
        **{0: {'view': 'focused'}},
        **{k: {'view': 'ortho'} for k in range(1, 4)},
        **{k: {'view': 'planar'} for k in range(4, 12)},
        **{12: {'elements': ['scalar_bar']}}
    }
    layout = layout.annotate(annotations)

    parcellation_gifti = get_schaefer400_gifti(tpl='fsaverage')
    plot_f = plotdef(
        surf_from_archive(),
        add_surface_overlay(
            'parcellation',
            #surf_scalars_from_gifti('parcellation', is_masked=False),
            surf_scalars_from_cifti('parcellation'),
            parcellate_colormap('parcellation', 'network', template='fsLR'),
            vertex_to_face('parcellation'),
        ),
        add_surface_overlay(
            'pain',
            surf_scalars_from_nifti('pain', template='fsLR'),
            vertex_to_face('pain'),
        ),
        plot_to_image(),
        auto_camera(n_ortho=3, focus='centroid', n_angles=8, surf_scalars='pain_points'),
        save_figure(
            layout_kernel=layout,
            #padding=10,
            canvas_size=(3200, 1500),
            canvas_color=(0, 0, 0),
            scalar_bar_action='collect',
        ),
    )
    plot_f(
        #template='fsaverage',
        template='fsLR',
        load_mask=True,
        parcellation_cifti=get_schaefer400_cifti(),
        # parcellation_gifti_left=parcellation_gifti['left'],
        # parcellation_gifti_right=parcellation_gifti['right'],
        parcellation_alpha=0.4,
        pain_nifti=nb.load(get_pain_thresh_nifti()),
        pain_cmap='inferno',
        pain_clim=(0.3, 8),
        pain_alpha=0.9,
        pain_below_color=(0, 0, 0, 0),
        surf_projection=('veryinflated',),
        hemisphere='left',
        window_size=(600, 500),
        theme=pv.themes.DarkTheme(),
        output_dir='/tmp',
        fname_spec=f'scalars-painoverlay_view-auto_page-{{page}}',
    )
