# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for elementary brain network visualisations
"""
from pkg_resources import resource_filename as pkgrf

import numpy as np
import pandas as pd

from hyve.flows import plotdef
from hyve.flows import add_network_data, joindata
from hyve.transforms import (
    surf_from_archive,
    surf_scalars_from_cifti,
    parcellate_colormap,
    add_node_variable,
    add_edge_variable,
    plot_to_image,
    save_snapshots,
    node_coor_from_parcels,
    build_network,
    add_network_overlay,
)


def test_net():
    parcellation = '/Users/rastkociric/Downloads/desc-schaefer_res-0400_atlas.nii'
    cov = pd.read_csv(pkgrf(
        'hypercoil',
        'examples/synthetic/data/synth-regts/'
        f'atlas-schaefer400_desc-synth_cov.tsv'
    ), sep='\t', header=None).values

    vis_nodes_edge_selection = np.zeros(400, dtype=bool)
    vis_nodes_edge_selection[0:5] = True
    vis_nodes_edge_selection[200:205] = True

    plot_f = plotdef(
        surf_from_archive(),
        surf_scalars_from_cifti('parcellation', plot=False),
        add_network_data(
            add_node_variable('vis'),
            add_edge_variable(
                "vis_conn",
                threshold=10,
                topk_threshold_nodewise=True,
                absolute=True,
                incident_node_selection=vis_nodes_edge_selection,
                emit_degree=True,
            ),
            add_edge_variable(
                "vis_internal_conn",
                absolute=True,
                connected_node_selection=vis_nodes_edge_selection,
            ),
        ),
        node_coor_from_parcels('parcellation'),
        build_network('vis'),
        parcellate_colormap('network', 'parcellation', target='node'),
        plot_to_image(),
        save_snapshots(
            fname_spec=(
                'network-schaefer400_view-{view}'
            ),
        ),
    )
    plot_f(
        template='fsLR',
        surf_projection='inflated',
        surf_alpha=0.2,
        parcellation_cifti=parcellation,
        node_radius='vis_conn_degree',
        node_color='index',
        edge_color='vis_conn_sgn',
        edge_radius='vis_conn_val',
        vis_nodal=vis_nodes_edge_selection.astype(int),
        vis_conn_adjacency=cov,
        vis_internal_conn_adjacency=cov,
        views=('dorsal', 'left', 'posterior'),
        output_dir='/tmp',
    )


def test_net_highlight():
    parcellation = '/Users/rastkociric/Downloads/desc-schaefer_res-0400_atlas.nii'
    cov = pd.read_csv(pkgrf(
        'hypercoil',
        'examples/synthetic/data/synth-regts/'
        f'atlas-schaefer400_desc-synth_cov.tsv'
    ), sep='\t', header=None).values

    vis_nodes_edge_selection = np.zeros(400, dtype=bool)
    vis_nodes_edge_selection[0:2] = True
    vis_nodes_edge_selection[200:202] = True

    plot_f = plotdef(
        surf_from_archive(),
        surf_scalars_from_cifti('parcellation', plot=False),
        #build_network('vis'),
        add_network_overlay(
            'vis',
            joindata(fill_value=0., how="left")(
                add_edge_variable(
                    "vis_conn",
                    absolute=True,
                    incident_node_selection=vis_nodes_edge_selection,
                ),
                add_edge_variable(
                    "vis_internal_conn",
                    absolute=True,
                    connected_node_selection=vis_nodes_edge_selection,
                    emit_degree=True,
                    emit_incident_nodes=(0.2, 1),
                    removed_val=0.03,
                    surviving_val=1.0,
                ),
            ),
            parcellate_colormap('modal', 'parcellation', target='node'),
            node_coor_from_parcels('parcellation'),
        ),
        plot_to_image(),
        save_snapshots(
            fname_spec=(
                'network-schaefer400_desc-visual_view-{view}'
            ),
        ),
    )
    plot_f(
        template="fsLR",
        surf_projection='veryinflated',
        surf_alpha=0.2,
        parcellation_cifti=parcellation,
        vis_conn_adjacency=cov,
        vis_internal_conn_adjacency=cov,
        vis_node_radius='vis_internal_conn_degree',
        vis_node_color='index',
        vis_node_alpha='vis_internal_conn_incidents',
        vis_edge_color='vis_conn_sgn',
        vis_edge_radius='vis_conn_val',
        vis_edge_alpha='vis_internal_conn_val',
        views=("dorsal", "left", "posterior"),
        output_dir='/tmp',
    )
