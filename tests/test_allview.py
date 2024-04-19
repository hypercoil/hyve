# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for all views on a brain surface
"""
import templateflow.api as tflow

from hyve_examples import get_null400_cifti
from hyve.flows import plotdef
from hyve.transforms import (
    surf_from_archive,
    surf_scalars_from_cifti,
    surf_scalars_from_nifti,
    add_surface_overlay,
    parcellate_colormap,
    vertex_to_face,
    save_grid,
    plot_to_image,
    text_element,
)

from hyve.elements import TextBuilder

COMMON_PARAMS = dict(
    load_mask=True,
    hemisphere=['left', 'right', None],
    views={
        'left': ('medial', 'lateral'),
        'right': ('medial', 'lateral'),
        'both': ('dorsal', 'ventral', 'anterior', 'posterior'),
    },
    output_dir='/tmp',
    window_size=(600, 500),
)


def get_annotations():
    return {
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
        4: dict(elements=['title']),
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


def test_allviews_scalars():
    annotations = get_annotations()
    annotations[4]['elements'] = ['scalar_bar', 'title']
    plot_f = plotdef(
        surf_from_archive(),
        add_surface_overlay(
            'GM Density',
            surf_scalars_from_nifti(
                'GM Density', template='fsaverage', plot=True
            ),
        ),
        plot_to_image(),
        save_grid(
            n_cols=3, n_rows=3, padding=10,
            canvas_size=(1800, 1500),
            canvas_color=(0, 0, 0),
            fname_spec='scalars-gmdensity_view-all_page-{page}',
            scalar_bar_action='collect',
            annotations=annotations,
        ),
        text_element(
            name='title',
            content='{surfscalars}',
            bounding_box_height=192,
            font_size_multiplier=0.2,
            font_color='#cccccc',
            priority=-1,
        ),
    )
    plot_f(
        template='fsaverage',
        gm_density_nifti=tflow.get(
            template='MNI152NLin2009cAsym',
            suffix='probseg',
            label='GM',
            resolution=2
        ),
        gm_density_clim=(0.2, 0.9),
        gm_density_below_color=None,
        gm_density_scalar_bar_style={
            'name': None,
            'orientation': 'h',
        },
        surf_projection=('pial',),
        title_element_content='Gray Matter Density',
        **COMMON_PARAMS,
    )


def test_allviews_parcellation():
    annotations = get_annotations()
    plot_f = plotdef(
        surf_from_archive(),
        surf_scalars_from_cifti(
            'parcellation',
            allow_multihemisphere=False,
        ),
        parcellate_colormap('parcellation', 'network'),
        vertex_to_face('parcellation'),
        plot_to_image(),
        save_grid(
            n_cols=3, n_rows=3, padding=10,
            canvas_size=(1800, 1500),
            canvas_color=(0, 0, 0),
            fname_spec='scalars-parcellation_view-all_page-{page}',
            scalar_bar_action='collect',
            annotations=annotations,
        ),
    )
    plot_f(
        template='fsLR',
        parcellation_cifti=get_null400_cifti(),
        surf_projection=('veryinflated',),
        elements={
            'title': (
                TextBuilder(
                    content='null parcellation',
                    bounding_box_height=128,
                    font_size_multiplier=0.2,
                    font_color='#cccccc',
                    priority=-1,
                ),
            ),
        },
        **COMMON_PARAMS,
    )
