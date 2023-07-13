# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for elementary brain surface visualisations
"""
import pytest

from pkg_resources import resource_filename as pkgrf

import numpy as np
import templateflow.api as tflow

from conveyant import (
    ichain,
    ochain,
    iochain,
    split_chain,
    imap,
    omap,
)
from hyve.prim import automap_unified_plotter_p
from hyve.transforms import (
    surf_from_archive,
    resample_to_surface,
    plot_to_image,
    scalars_from_cifti,
    scalars_from_gifti,
    # parcellate_scalars,
    parcellate_colormap,
    # scatter_into_parcels,
    plot_to_html,
)


def print_params(**params):
    print(params)
    assert 0


@pytest.mark.ci_unsupported
def test_scalars():
    i_chain = ichain(
        surf_from_archive(),
        resample_to_surface('gm_density', template='fsaverage', plot=True),
    )
    o_chain = omap(
        plot_to_image(),
        mapping={
            'hemisphere': ('left', 'right'),
            'basename': ('/tmp/left_density', '/tmp/right_density'),
        }
    )
    f = iochain(automap_unified_plotter_p, i_chain, o_chain)
    out = f(
        template='fsaverage',
        load_mask=True,
        gm_density_nifti=tflow.get(
            template='MNI152NLin2009cAsym',
            suffix='probseg',
            label='GM',
            resolution=2
        ),
        surf_projection=('pial',),
        hemisphere=['left', 'right'],
        #off_screen=False,
    )
    assert len(out.keys()) == 1
    assert "screenshots" in out.keys()


@pytest.mark.ci_unsupported
def test_parcellation():
    i_chain = ichain(
        surf_from_archive(),
        scalars_from_cifti('parcellation', plot=True),
        parcellate_colormap('network', 'parcellation')
    )
    o_chain = ochain(
        split_chain(
            omap(
                plot_to_image(),
                mapping={
                    'hemisphere': ('left', 'right'),
                    'basename': ('/tmp/left', '/tmp/right'),
                }
            ),
            omap(
                plot_to_image(),
                mapping={
                    'hemisphere': ('left', 'right'),
                    'basename': ('/tmp/left', '/tmp/right'),
                    'views': (((-20, 0, 0),), (((65, 65, 0), (0, 0, 0), (0, 0, 1)),))
                }
            ),
        )
    )
    f = iochain(automap_unified_plotter_p, i_chain, o_chain)
    f(
        template="fsLR",
        load_mask=True,
        parcellation_cifti=pkgrf(
            'hyve',
            'data/examples/nullexample.nii'
        ),
        surf_projection=('veryinflated',),
        surf_scalars_boundary_color='black',
        surf_scalars_boundary_width=5,
        hemisphere=['left', 'right'],
    )

@pytest.mark.ci_unsupported
def test_parcellation_modal_cmap():
    i_chain = ichain(
        surf_from_archive(),
        scalars_from_gifti('parcellation', plot=True),
        parcellate_colormap('modal', 'parcellation')
    )
    o_chain = ochain(
        split_chain(
            omap(
                plot_to_image(),
                mapping={
                    'basename': ('/tmp/leftmodal', '/tmp/rightmodal'),
                    'hemisphere': ('left', 'right'),
                }
            ),
            omap(
                plot_to_image(),
                mapping={
                    'basename': ('/tmp/leftmodal', '/tmp/rightmodal'),
                    'hemisphere': ('left', 'right'),
                    'views': (((-20, 0, 0),), (((65, 65, 0), (0, 0, 0), (0, 0, 1)),))
                }
            ),
        )
    )
    f = iochain(automap_unified_plotter_p, i_chain, o_chain)
    f(
        template="fsLR",
        load_mask=True,
        parcellation_gifti_left=pkgrf(
            'hyve',
            'data/examples/nullexample_L.gii'
        ),
        parcellation_gifti_right=pkgrf(
            'hyve',
            'data/examples/nullexample_R.gii'
        ),
        surf_projection=('veryinflated',),
        surf_scalars_boundary_color='black',
        surf_scalars_boundary_width=5,
        hemisphere=['left', 'right'],
    )

@pytest.mark.ci_unsupported
def test_parcellation_html():
    i_chain = ichain(
        surf_from_archive(),
        scalars_from_cifti('parcellation', plot=True),
        parcellate_colormap('network', 'parcellation')
    )
    o_chain = ochain(
        omap(
            plot_to_html(backend="panel"),
            mapping={
                "filename": ('/tmp/left.html', '/tmp/right.html'),
            }
        ),
    )
    f = iochain(automap_unified_plotter_p, i_chain, o_chain)
    f(
        template="fsLR",
        load_mask=True,
        parcellation_cifti=pkgrf(
            'hyve',
            'data/examples/nullexample.nii'
        ),
        surf_projection=['veryinflated'],
        surf_scalars_boundary_color='black',
        surf_scalars_boundary_width=5,
        hemisphere=['left', 'right'],
    )

# @pytest.mark.ci_unsupported
# def test_parcellated_scalars(self):
#     i_chain = ichain(
#         surf_from_archive(),
#         resample_to_surface('gm_density', template='fsLR'),
#         scalars_from_cifti('parcellation'),
#         parcellate_scalars('gm_density', 'parcellation'),
#     )
#     o_chain = ochain(
#         map_over_sequence(
#             xfm=plot_and_save(),
#             mapping={
#                 "basename": ('/tmp/left_density_parc', '/tmp/right_density_parc'),
#                 "hemi": ('left', 'right'),
#             }
#         )
#     )
#     f = iochain(plot_surf_scalars, i_chain, o_chain)
#     out = f(
#         template="fsLR",
#         load_mask=True,
#         parcellation_cifti=pkgrf(
#             'hypercoil',
#             'viz/resources/nullexample.nii'
#         ),
#         gm_density_nifti=tflow.get(
#             template='MNI152NLin2009cAsym',
#             suffix='probseg',
#             label="GM",
#             resolution=2
#         ),
#         projection='inflated',
#         clim=(0.2, 0.9),
#     )
#     assert len(out.keys()) == 1
#     assert "screenshots" in out.keys()

#     parcellated = np.random.rand(400)
#     i_chain = ichain(
#         surf_from_archive(),
#         scalars_from_cifti('parcellation'),
#         scatter_into_parcels('scalars', 'parcellation'),
#     )
#     o_chain = ochain(
#         map_over_sequence(
#             xfm=plot_and_save(),
#             mapping={
#                 "basename": ('/tmp/left_noise_parc', '/tmp/right_noise_parc'),
#                 "hemi": ('left', 'right'),
#             }
#         )
#     )
#     f = iochain(plot_surf_scalars, i_chain, o_chain)
#     out = f(
#         template="fsLR",
#         load_mask=True,
#         parcellation_cifti=pkgrf(
#             'hypercoil',
#             'viz/resources/nullexample.nii'
#         ),
#         parcellated=parcellated,
#         projection='inflated',
#         clim=(0, 1),
#         cmap='inferno',
#     )
