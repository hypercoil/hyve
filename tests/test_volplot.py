# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for elementary brain volume visualisations
"""
import pytest
import pyvista as pv
import templateflow.api as tflow
from hyve_examples import get_pain_thresh_nifti
from hyve.flows import plotdef
from hyve.transforms import (
    surf_from_archive,
    points_scalars_from_nifti,
    select_internal_points,
    plot_to_image,
    save_snapshots,
    save_grid,
    add_points_overlay,
)


@pytest.mark.parametrize('overlay', [True, False])
@pytest.mark.parametrize('paramindef', [True, False])
@pytest.mark.parametrize('select_internal', [True, False])
@pytest.mark.parametrize('scalars_cmap', [
    ('parcellation', 'tab20b'),
    ('gmdensity', 'cividis'),
    ('wmdensity', 'cividis'),
    ('pain', 'magma'),
])
def test_points_snap_battery(
    overlay, paramindef, select_internal, scalars_cmap
):
    scalars, cmap = scalars_cmap
    inargs = {
        (
            f'{scalars}_cmap' if overlay else 'points_scalars_cmap'
        ): cmap,
    }
    fname_spec = (
        'scalars-{pointsscalars}_hemisphere-{hemisphere}_view-{view}_'
        f'cmap-{cmap}_internal-{select_internal}_'
        f'overlay-{overlay}_paramindef-{paramindef}'
    )
    inprims = [
        points_scalars_from_nifti(scalars, geom_name='MNIres2grid'),
    ]
    extra_inprims = []
    if scalars == 'parcellation':
        #extra_in = [points_scalars_from_nifti('gmdensity', plot=False)]
        inargs = {
            **inargs,
            'parcellation_nifti': tflow.get(
                'MNI152NLin2009cAsym',
                suffix='dseg',
                resolution=2,
                atlas='Schaefer2018',
                desc='400Parcels7Networks',
            ),
            (
                f'{scalars}_clim' if overlay else 'points_scalars_clim'
            ): (0.1, 400.9),
            # 'gmdensity_nifti': tflow.get(
            #     'MNI152NLin2009cAsym',
            #     suffix='probseg',
            #     label='GM',
            #     resolution=2,
            # ),
        }
        # extra_inprims += [
        #     select_internal_points(),
        #     points_scalars_from_nifti(
        #         'gmdensity',
        #         geom_name='MNIres2grid',
        #         plot=False,
        #     ),
        # ]
    elif scalars == 'pain':
        inargs = {
            **inargs,
            'pain_nifti': get_pain_thresh_nifti(),
        }
    else:
        label_key = 'GM' if scalars == 'gmdensity' else 'WM'
        inargs = {
            **inargs,
            f'{scalars}_nifti': tflow.get(
                'MNI152NLin2009cAsym',
                suffix='probseg',
                label=label_key,
                resolution=2,
            ),
            (
                f'{scalars}_clim' if overlay else 'points_scalars_clim'
            ): (0.3, 1.0),
        }
    if overlay:
        inprims = [
            add_points_overlay(scalars, *inprims)
        ]
    if paramindef:
        outprims = [save_snapshots(fname_spec=fname_spec)]
        outargs = {}
    else:
        outprims = [save_snapshots()]
        outargs = {'fname_spec': fname_spec}
    if select_internal:
        extra_inprims += [select_internal_points()]
    plot_f = plotdef(
        surf_from_archive(),
        *inprims,
        *extra_inprims,
        plot_to_image(),
        *outprims,
    )
    plot_f(
        template='fsaverage',
        surf_projection=('pial',),
        hemisphere=['left', 'right', 'both'],
        surf_alpha=0.3,
        **inargs,
        parallel_projection=True,
        window_size=(600, 400),
        output_dir='/tmp',
        views={
            'left': ('lateral',),
            'right': ('lateral',),
            'both': ('dorsal',),
        },
        **outargs,
        theme=pv.themes.DarkTheme(),
    )


@pytest.mark.parametrize('outprim', ['snapshots', 'grid'])
def test_points_scalars(outprim):
    nii = get_pain_thresh_nifti()
    additional = []
    match outprim:
        case 'snapshots':
            additional += [
                save_snapshots(
                    fname_spec=(
                        'scalars-{pointsscalars}_view-{view}'
                    ),
                )
            ]
            add_args = {}
        case 'grid':
            additional += [
                save_grid(
                    n_cols=3, n_rows=1,
                    canvas_size=(1800, 500),
                    canvas_color=(1, 1, 1),
                    fname_spec=f'scalars-pain_view-all_page-{{page}}',
                    scalar_bar_action='collect',
                )
            ]
            add_args = {'window_size': (1200, 1000)}
    plot_f = plotdef(
        surf_from_archive(),
        #points_scalars_from_nifti('pain'),
        add_points_overlay(
            'pain',
            points_scalars_from_nifti('pain'),
        ),
        plot_to_image(),
        *additional,
    )
    plot_f(
        template='fsaverage',
        surf_projection=('pial',),
        hemisphere='right',
        surf_alpha=0.3,
        pain_nifti=nii,
        #points_alpha=0.8,
        pain_cmap='magma',
        pain_alpha=0.8,
        parallel_projection=True,
        #points_style={'style': 'points', 'point_size': 10.0},
        views=('dorsal', 'lateral', 'anterior'),
        **add_args,
        output_dir='/tmp',
    )
