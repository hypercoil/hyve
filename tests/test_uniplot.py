# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for elementary surfplot-based visualisations
"""
import pytest

from pkg_resources import resource_filename as pkgrf

import nibabel as nb
import numpy as np
import pandas as pd

from hyve.plot import unified_plotter
from hyve.surf import CortexTriSurface
from hyve.util import filter_adjacency_data, filter_node_data


@pytest.mark.ci_unsupported
def test_unified_plotter():
    surf = CortexTriSurface.from_nmaps(projections=('pial', 'inflated'))
    unified_plotter(
        surf=surf,
        surf_alpha=0.2,
        off_screen=False,
    )[0].show()
    unified_plotter(
        surf=surf,
        surf_alpha=0.2,
        off_screen=False,
        hemisphere='left',
        hemisphere_slack=1.2,
        surf_projection='inflated',
    )[0].show()
    surf.add_vertex_dataset(
        'data',
        data=np.random.rand(40962 * 2),
        apply_mask=False,
    )
    unified_plotter(
        surf=surf,
        surf_scalars='data',
        surf_scalars_cmap='magma',
        surf_alpha=0.2,
        hemisphere_slack=1.2,
        off_screen=False,
    )[0].show()
    vol = nb.load("/Users/rastkociric/Downloads/pain_thresh_cFWE05.nii.gz")
    vol_data = vol.get_fdata()
    vol_loc = np.where(vol_data > 0)
    vol_scalars = vol_data[vol_data > 0]
    vol_coor = np.stack(vol_loc)
    vol_coor = (vol.affine @ np.concatenate(
        (vol_coor, np.ones((1, vol_coor.shape[-1])))
    ))[:3].T
    vol_voxdim = vol.header.get_zooms()
    unified_plotter(
        vol_coor=vol_coor,
        vol_scalars=vol_scalars,
        vol_voxdim=vol_voxdim,
        hemisphere_slack=1.2,
        off_screen=False,
    )[0].show()
    unified_plotter(
        surf=surf,
        surf_scalars='data',
        surf_scalars_cmap='magma',
        surf_alpha=0.2,
        vol_coor=vol_coor,
        vol_scalars=vol_scalars,
        vol_voxdim=vol_voxdim,
        hemisphere_slack=1.2,
        off_screen=False,
    )[0].show()
    parcellation = '/Users/rastkociric/Downloads/desc-schaefer_res-0400_atlas.nii'
    surf_lr = CortexTriSurface.from_tflow(load_mask=True, projections=('inflated',))
    surf_lr.add_vertex_dataset(
        'parcellation',
        data=nb.load(parcellation).get_fdata().ravel(),
        is_masked=True,
    )
    surf_lr.add_vertex_dataset(
        'data',
        data=np.random.rand(32492 * 2),
        apply_mask=False,
    )
    node_coor = surf_lr.parcel_centres_of_mass('parcellation', 'inflated')
    cov = pd.read_csv(pkgrf(
        'hyve',
        'data/examples/atlas-schaefer400_desc-synth_cov.tsv',
    ), sep='\t', header=None).values
    vis_nodes_edge_selection = np.zeros(400, dtype=bool)
    vis_nodes_edge_selection[0:2] = True
    vis_nodes_edge_selection[200:202] = True
    node_data = filter_node_data(cov.sum(axis=0))
    edge_data = filter_adjacency_data(
        cov, connected_node_selection=vis_nodes_edge_selection)
    node_clim = (node_data['node_val'].min(), node_data['node_val'].max())
    edge_clim = (-1, 1)
    node_lh = np.zeros(400, dtype=bool)
    node_lh[:200] = True
    unified_plotter(
        node_values=node_data,
        node_coor=node_coor,
        node_clim=node_clim,
        node_color='node_val',
        node_lh=node_lh,
        edge_values=edge_data,
        edge_clim=edge_clim,
        hemisphere_slack=1.2,
        off_screen=False,
    )[0].show()
    unified_plotter(
        surf=surf_lr,
        surf_projection='inflated',
        surf_scalars='data',
        surf_scalars_cmap='magma',
        surf_alpha=0.2,
        vol_coor=vol_coor,
        vol_scalars=vol_scalars,
        vol_voxdim=vol_voxdim,
        node_values=node_data,
        node_coor=node_coor,
        node_clim=node_clim,
        node_color='node_val',
        node_lh=node_lh,
        edge_values=edge_data,
        edge_clim=edge_clim,
        hemisphere_slack=1.2,
        off_screen=False,
    )[0].show()
