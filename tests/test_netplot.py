# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for elementary brain network visualisations
"""
import pytest
import numpy as np
import pandas as pd
import pyvista as pv

from hyve_examples import (
    get_schaefer400_synthetic_conmat,
    get_schaefer400_cifti,
)
from hyve.flows import plotdef
from hyve.flows import add_network_data
from hyve.transforms import (
    surf_from_archive,
    surf_scalars_from_cifti,
    parcellate_colormap,
    add_node_variable,
    add_edge_variable,
    plot_to_image,
    save_snapshots,
    save_grid,
    node_coor_from_parcels,
    build_network,
    add_network_overlay,
)


def test_net():
    parcellation = get_schaefer400_cifti()
    cov = pd.read_csv(
        get_schaefer400_synthetic_conmat(), sep='\t', header=None
    ).values

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
        parcellate_colormap('parcellation', 'network', target='node'),
        plot_to_image(),
        save_snapshots(
            fname_spec=(
                'network-schaefer400_view-{view}'
            ),
        ),
    )
    plot_f(
        template='fsLR',
        hemisphere='right',
        surf_projection='inflated',
        surf_alpha=0.2,
        parcellation_cifti=parcellation,
        node_radius='vis_conn_degree',
        node_color='index',
        node_style={
            'pbr': True, 'metallic': 0.3, 'roughness': 0.1,
            'specular': 0.5, 'specular_power': 15,
        },
        edge_color='vis_conn_sgn',
        edge_radius='vis_conn_val',
        edge_style={
            'pbr': True, 'metallic': 0.3, 'roughness': 0.1,
            'specular': 0.5, 'specular_power': 15,
        },
        vis_nodal=vis_nodes_edge_selection.astype(int),
        vis_conn_adjacency=cov,
        vis_internal_conn_adjacency=cov,
        views=('dorsal', 'lateral', 'posterior'),
        output_dir='/tmp',
    )


def test_net_highlight_nooverlay():
    parcellation = get_schaefer400_cifti()
    cov = pd.read_csv(
        get_schaefer400_synthetic_conmat(), sep='\t', header=None
    ).values

    vis_nodes_edge_selection = np.zeros(400, dtype=bool)
    vis_nodes_edge_selection[0:2] = True
    vis_nodes_edge_selection[200:202] = True

    plot_f = plotdef(
        surf_from_archive(),
        surf_scalars_from_cifti('parcellation', plot=False),
        #build_network('vis'),
        add_network_data(
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
            how='left',
        ),
        node_coor_from_parcels('parcellation'),
        build_network('vis'),
        parcellate_colormap('parcellation', 'modal', target='node'),
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
        node_radius='vis_internal_conn_degree',
        node_color='index',
        node_alpha='vis_internal_conn_incidents',
        edge_color='vis_conn_sgn',
        edge_radius='vis_conn_val',
        edge_alpha='vis_internal_conn_val',
        views=("dorsal", "left", "posterior"),
        output_dir='/tmp',
    )


def test_net_fig():
    parcellation = get_schaefer400_cifti()
    cov = pd.read_csv(
        get_schaefer400_synthetic_conmat(), sep='\t', header=None
    ).values

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
        parcellate_colormap('parcellation', 'network', target='node'),
        plot_to_image(),
        save_grid(
            n_cols=3, n_rows=1,
            canvas_size=(1800, 500),
            canvas_color=(1, 1, 1),
            scalar_bar_action='collect',
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
        window_size=(1200, 1000),
        output_dir='/tmp',
        fname_spec=f'network-schaefer400_view-all_page-{{page}}',
    )


@pytest.mark.parametrize('cmap', ['network', 'modal'])
def test_net_highlight(cmap):
    parcellation = get_schaefer400_cifti()
    cov = pd.read_csv(
        get_schaefer400_synthetic_conmat(), sep='\t', header=None
    ).values

    vis_nodes_edge_selection = np.zeros(400, dtype=bool)
    vis_nodes_edge_selection[0:2] = True
    vis_nodes_edge_selection[200:202] = True

    plot_f = plotdef(
        surf_from_archive(),
        surf_scalars_from_cifti('parcellation', plot=False),
        #build_network('vis'),
        add_network_overlay(
            'vis',
            add_network_data(
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
                how='left',
            ),
            parcellate_colormap('parcellation', 'modal', target='node'),
            node_coor_from_parcels('parcellation'),
        ),
        plot_to_image(),
        save_snapshots(
            fname_spec=(
                f'network-schaefer400_desc-visual_cmap-{cmap}_view-{{view}}'
            ),
        ),
    )
    plot_f(
        template="fsLR",
        surf_projection='veryinflated',
        surf_alpha=0.2,
        parcellation_cifti=parcellation,
        surf_scalars_cmap=cmap,
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


def test_net_highlight_fig():
    parcellation = get_schaefer400_cifti()
    cov = pd.read_csv(
        get_schaefer400_synthetic_conmat(), sep='\t', header=None
    ).values

    vis_nodes_edge_selection = np.zeros(400, dtype=bool)
    vis_nodes_edge_selection[0:2] = True
    vis_nodes_edge_selection[200:202] = True

    plot_f = plotdef(
        surf_from_archive(),
        surf_scalars_from_cifti('parcellation', plot=False),
        #build_network('vis'),
        add_network_overlay(
            'vis',
            add_network_data(
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
                how='left',
            ),
            parcellate_colormap('parcellation', 'modal', target='node'),
            node_coor_from_parcels('parcellation'),
        ),
        plot_to_image(),
        save_grid(
            n_cols=3, n_rows=1,
            canvas_size=(1800, 500),
            canvas_color=(1, 1, 1),
            fname_spec=f'network-schaefer400_desc-visual_view-all_page-{{page}}',
            scalar_bar_action='collect',
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
        window_size=(1200, 1000),
        output_dir='/tmp',
    )
