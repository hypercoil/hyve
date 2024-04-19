# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for elementary brain surface visualisations
"""
import pytest

import numpy as np
import pyvista as pv
import templateflow.api as tflow

from hyve_examples import (
    get_null400_cifti,
    get_null400_gifti,
    get_poldrack_freesurfer,
    get_pain_thresh_nifti,
)
from hyve.flows import plotdef
from hyve.transforms import (
    surf_from_archive,
    surf_from_freesurfer,
    surf_scalars_from_array,
    surf_scalars_from_cifti,
    surf_scalars_from_freesurfer,
    surf_scalars_from_gifti,
    surf_scalars_from_nifti,
    add_surface_overlay,
    parcellate_colormap,
    parcellate_surf_scalars,
    scatter_into_parcels,
    vertex_to_face,
    draw_surface_boundary,
    select_active_parcels,
    plot_to_html,
    plot_to_image,
    save_snapshots,
)

@pytest.mark.parametrize('hemi', ['left', 'right'])
@pytest.mark.parametrize('overlay', [True, False])
@pytest.mark.parametrize('paramindef', [True, False])
@pytest.mark.parametrize('scalars_cmap', [
    ('parcellation', 'network'),
    ('parcellation', 'modal'),
    ('parcellation', 'bone'),
    ('gmdensity', 'bone'),
    ('gmscatter', 'bone'),
    ('pain', 'magma'),
])
def test_surface_snap_battery(hemi, overlay, paramindef, scalars_cmap):
    scalars, cmap = scalars_cmap
    inargs = {}
    fname_spec = (
        'scalars-{surfscalars}_hemisphere-{hemisphere}_view-{view}_'
        f'cmap-{cmap}_overlay-{overlay}_paramindef-{paramindef}'
    )
    scatter = (scalars == 'gmscatter')
    if scatter:
        fname_spec = f'{fname_spec}_mode-scatter'
    if scalars == 'parcellation':
        inprims = [
            surf_scalars_from_cifti(scalars),
            parcellate_colormap(
                scalars,
                cmap_name=(
                    cmap if (paramindef and cmap != 'bone') else 'network'
                ),
            ),
            vertex_to_face(scalars),
        ]
        inargs = {**inargs, 'parcellation_cifti': get_null400_cifti()}
        if not paramindef or cmap == 'bone':
            inargs = {**inargs, **{(
                'parcellation_cmap' if overlay else 'surf_scalars_cmap'
            ): cmap}}
    elif scalars == 'gmscatter':
        defargs = {}
        if paramindef:
            defargs = {'plot': False}
        else:
            inargs = {
                **inargs,
                'gmdensity_plot': False,
                'parcellation_plot': False,
            }
        inprims = [
            surf_scalars_from_nifti('gmdensity', **defargs),
            surf_scalars_from_cifti('parcellation', **defargs),
            parcellate_surf_scalars('gmdensity', 'parcellation'),
            vertex_to_face('gmdensity', interpolation='mode'),
        ]
        scalars = 'gmdensity'
    else:
        defargs = {}
        if paramindef:
            defargs = {'interpolation': 'mean'}
        else:
            inargs = {**inargs, f'{scalars}_v2f_interpolation': 'mean'}
        inprims = [
            surf_scalars_from_nifti(scalars),
            vertex_to_face(scalars, **defargs),
        ]
    if scalars == 'gmdensity':
        inargs = {
            **inargs,
            f'gmdensity_nifti': tflow.get(
                template='MNI152NLin2009cAsym',
                suffix='probseg',
                label='GM',
                resolution=2
            ),
            **{(
                f'{scalars}_cmap' if overlay else 'surf_scalars_cmap'
            ): cmap},
        }
        if scatter:
            inargs = {**inargs, **{
                'parcellation_cifti': get_null400_cifti(),
            }}
    elif scalars == 'pain':
        inargs = {
            **inargs,
            'pain_nifti': get_pain_thresh_nifti(),
            (
                f'{scalars}_cmap' if overlay else 'surf_scalars_cmap'
            ): cmap,
            (
                f'{scalars}_below_color'
                if overlay
                else 'surf_scalars_below_color'
            ): (0.3, 0.3, 0.3, 0.3),
        }
    if overlay:
        inprims = [
            add_surface_overlay(scalars, *inprims)
        ]
    if paramindef:
        outprims = [save_snapshots(fname_spec=fname_spec)]
        outargs = {}
    else:
        outprims = [save_snapshots()]
        outargs = {'fname_spec': fname_spec}
    plot_f = plotdef(
        surf_from_archive(),
        *inprims,
        plot_to_image(),
        *outprims,
    )
    plot_f(
        template="fsLR",
        load_mask=True,
        **inargs,
        surf_projection=['veryinflated'],
        # surf_scalars_boundary_color='black',
        # surf_scalars_boundary_width=5,
        hemisphere=hemi,
        window_size=(600, 400),
        output_dir='/tmp',
        views=('lateral',),
        **outargs,
    )


def test_scalars():
    plot_f = plotdef(
        surf_from_archive(),
        surf_scalars_from_nifti('gmdensity', template='fsaverage', plot=True),
        surf_scalars_from_array('noise', plot=False),
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
        noise_array_left=np.random.randn(37476),
        noise_array_right=np.random.randn(37471),
        surf_style={
            'pbr': True,
            'metallic': 0.05,
            'roughness': 0.1,
            'specular': 0.5,
            'specular_power': 15,
            # 'diffuse': 1,
        },
        surf_scalars_color='gmdensity',
        surf_scalars_alpha='noise',
        surf_projection=('pial',),
        hemisphere=['left', 'right'],
        output_dir='/tmp',
    )


@pytest.mark.ci_unsupported
@pytest.mark.parametrize('cmap', ['network', 'modal'])
@pytest.mark.parametrize('v2f,imgtype', [(True, 'cifti'), (False, 'gifti')])
def test_parcellation(
    cmap: str,
    v2f: bool,
    imgtype: str,
):
    additional = []
    mode = 'vertex'
    if v2f:
        additional.append(vertex_to_face('parcellation'))
        mode = 'face'
    match imgtype:
        case 'cifti':
            inprim = [surf_scalars_from_cifti('parcellation', plot=True)]
            imgargs = {'parcellation_cifti': get_null400_cifti()}
        case 'gifti':
            inprim = [surf_scalars_from_gifti('parcellation', plot=True)]
            gifti = get_null400_gifti()
            imgargs = {
                'parcellation_gifti_left': gifti['left'],
                'parcellation_gifti_right': gifti['right'],
            }
    plot_f = plotdef(
        surf_from_archive(),
        *inprim,
        parcellate_colormap('parcellation', cmap),
        *additional,
        plot_to_image(),
        save_snapshots(
            fname_spec=(
                'scalars-{surfscalars}_hemisphere-{hemisphere}_view-{view}_'
                f'cmap-{cmap}_mode-{mode}'
            ),
        ),
    )
    plot_f(
        template="fsLR",
        load_mask=True,
        **imgargs,
        surf_projection=('veryinflated',),
        surf_color='black',
        surf_style={'lighting': False},
        parallel_projection=True,
        hemisphere=['left', 'right'],
        views=[
            'lateral', 'medial', 'ventral', 'dorsal', 'anterior', 'posterior',
            (-20, 0, 0), ((65, 65, 0), (0, 0, 0), (0, 0, 1))
        ],
        output_dir='/tmp',
    )


@pytest.mark.ci_unsupported
@pytest.mark.parametrize('cmap', ['network', 'modal'])
@pytest.mark.parametrize('v2f,imgtype', [(True, 'cifti'), (False, 'gifti')])
def test_boundary_maps(
    cmap: str,
    v2f: bool,
    imgtype: str,
):
    additional = []
    drawbound_args = {}
    mode = 'vertex'
    if v2f:
        additional.append(vertex_to_face('zstat', interpolation='mean'))
        mode = 'face'
        drawbound_args = {
            'target_domain': 'face',
            'num_steps': 1, # 0,
            'v2f_interpolation': 'mode',
        }
    match imgtype:
        case 'cifti':
            inprim = [surf_scalars_from_cifti('parcellation', plot=False)]
            imgargs = {'parcellation_cifti': get_null400_cifti()}
        case 'gifti':
            inprim = [surf_scalars_from_gifti('parcellation', plot=False)]
            gifti = get_null400_gifti()
            imgargs = {
                'parcellation_gifti_left': gifti['left'],
                'parcellation_gifti_right': gifti['right'],
            }
    plot_f = plotdef(
        surf_from_archive(),
        add_surface_overlay(
            'zstat',
            surf_scalars_from_nifti('zstat', plot=True),
            *additional,
        ),
        add_surface_overlay(
            'parcellation',
            *inprim,
            parcellate_colormap('parcellation'),
            draw_surface_boundary(
                'parcellation',
                'parcellation',
                copy_values_to_boundary=True,
                **drawbound_args,
            ),
        ),
        plot_to_image(),
        save_snapshots(
            fname_spec=(
                'scalars-{surfscalars}_hemisphere-{hemisphere}_view-{view}_'
                f'cmap-{cmap}_mode-{mode}'
            ),
        ),
    )
    plot_f(
        template="fsLR",
        load_mask=True,
        **imgargs,
        parcellation_cmap=cmap,
        parcellation_alpha=0.5,
        zstat_nifti=get_pain_thresh_nifti(),
        zstat_cmap='magma',
        zstat_below_color=(0, 0, 0, 0),
        surf_projection=('veryinflated',),
        surf_color=(0.3, 0.3, 0.3),
        surf_style={'lighting': False},
        parallel_projection=True,
        hemisphere=['left', 'right'],
        views=[
            'lateral', 'medial',
        ],
        output_dir='/tmp',
        empty_builders=True,
        window_size=(600, 400),
        theme=pv.themes.DarkTheme(),
    )


@pytest.mark.ci_unsupported
@pytest.mark.parametrize('cmap', ['network', 'modal'])
@pytest.mark.parametrize('v2f,imgtype', [(True, 'cifti'), (False, 'gifti')])
def test_boundary_maps2(
    cmap: str,
    v2f: bool,
    imgtype: str,
):
    additional_zstat = []
    additional = []
    drawbound_args = {}
    mode = 'vertex'
    if v2f:
        additional_zstat.append(vertex_to_face('zstat', interpolation='mean'))
        additional.append(vertex_to_face('parcellation', interpolation='mode'))
        mode = 'face'
        drawbound_args = {
            'target_domain': 'face',
            'num_steps': 0,
            'v2f_interpolation': 'mode',
        }
    match imgtype:
        case 'cifti':
            inprim = [surf_scalars_from_cifti('parcellation', plot=True)]
            imgargs = {'parcellation_cifti': get_null400_cifti()}
        case 'gifti':
            inprim = [surf_scalars_from_gifti('parcellation', plot=True)]
            gifti = get_null400_gifti()
            imgargs = {
                'parcellation_gifti_left': gifti['left'],
                'parcellation_gifti_right': gifti['right'],
            }
    plot_f = plotdef(
        surf_from_archive(),
        add_surface_overlay(
            'parcellation',
            *inprim,
            parcellate_colormap('parcellation'),
        ),
        add_surface_overlay(
            'parcellation_boundary',
            draw_surface_boundary(
                'parcellation',
                'parcellation_boundary',
                **drawbound_args,
            ),
        ),
        add_surface_overlay(
            'zstat',
            surf_scalars_from_nifti('zstat', plot=True),
            *additional_zstat,
        ),
        *additional,
        plot_to_image(),
        save_snapshots(
            fname_spec=(
                'scalars-{surfscalars}_hemisphere-{hemisphere}_view-{view}_'
                f'cmap-{cmap}_mode-{mode}'
            ),
        ),
    )
    plot_f(
        template="fsLR",
        load_mask=True,
        **imgargs,
        parcellation_cmap=cmap,
        parcellation_boundary_color='black',
        zstat_nifti=get_pain_thresh_nifti(),
        zstat_cmap='magma',
        zstat_alpha=0.8,
        zstat_below_color=(0, 0, 0, 0),
        surf_projection=('veryinflated',),
        surf_color=(0.3, 0.3, 0.3),
        surf_style={'lighting': False},
        parallel_projection=True,
        hemisphere=['left', 'right'],
        views=[
            'lateral', 'medial',
        ],
        output_dir='/tmp',
        empty_builders=True,
        window_size=(600, 400),
    )


@pytest.mark.ci_unsupported
#@pytest.mark.parametrize('cmap', ['network', 'modal'])
@pytest.mark.parametrize('v2f,imgtype', [(True, 'cifti'), (False, 'gifti')])
def test_active_selection(
    #cmap: str,
    v2f: bool,
    imgtype: str,
):
    additional = []
    drawbound_args = {}
    mode = 'vertex'
    if v2f:
        additional.append(vertex_to_face('zstat', interpolation='mean'))
        mode = 'face'
        drawbound_args = {
            'target_domain': 'face',
            'num_steps': 1, # 0,
            'v2f_interpolation': 'mode',
        }
    match imgtype:
        case 'cifti':
            inprim = [surf_scalars_from_cifti('parcellation', plot=False)]
            imgargs = {'parcellation_cifti': get_null400_cifti()}
        case 'gifti':
            inprim = [surf_scalars_from_gifti('parcellation', plot=False)]
            gifti = get_null400_gifti()
            imgargs = {
                'parcellation_gifti_left': gifti['left'],
                'parcellation_gifti_right': gifti['right'],
            }
    plot_f = plotdef(
        surf_from_archive(),
        add_surface_overlay(
            'zstat',
            surf_scalars_from_nifti('zstat', plot=True),
        ),
        add_surface_overlay(
            'parcellation',
            *inprim,
            #parcellate_colormap('parcellation'),
            select_active_parcels('parcellation', 'zstat', parcel_coverage_threshold=0.1),
            draw_surface_boundary(
                'parcellation',
                'parcellation',
                copy_values_to_boundary=True,
                **drawbound_args,
            ),
        ),
        *additional,
        plot_to_image(),
        save_snapshots(
            fname_spec=(
                'scalars-{surfscalars}_hemisphere-{hemisphere}_view-{view}_'
                f'mode-{mode}'
            ),
        ),
    )
    plot_f(
        template="fsLR",
        load_mask=True,
        **imgargs,
        parcellation_color='aqua',
        parcellation_alpha=0.8,
        zstat_nifti=get_pain_thresh_nifti(),
        zstat_cmap='magma',
        zstat_below_color=(0, 0, 0, 0),
        surf_projection=('veryinflated',),
        surf_color=(0.3, 0.3, 0.3),
        surf_style={'lighting': False},
        parallel_projection=True,
        hemisphere=['left', 'right'],
        views=[
            'lateral', 'medial',
        ],
        output_dir='/tmp',
        empty_builders=True,
        window_size=(600, 400),
        theme=pv.themes.DarkTheme(),
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
        vertex_to_face('gmdensity', interpolation='mode'),
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
        surf_style={'lighting': False},
        parallel_projection=True,
        output_dir='/tmp',
    )
