# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from .elements import (
    ElementBuilder,
    RasterBuilder,
    ScalarBarBuilder,
    TextBuilder,
)
from .flows import plotdef, add_network_data
from .layout import (
    AnnotatedLayout,
    Cell,
    CellLayout,
    ColGroupSpec,
    GroupSpec,
    RowGroupSpec,
    grid,
    hsplit,
    vsplit,
)
from .plot import unified_plotter
from .transforms import (
    surf_from_archive,
    surf_from_freesurfer,
    surf_from_gifti,
    surf_scalars_from_cifti,
    surf_scalars_from_gifti,
    surf_scalars_from_array,
    surf_scalars_from_nifti,
    points_scalars_from_nifti,
    points_scalars_from_array,
    parcellate_colormap,
    parcellate_surf_scalars,
    scatter_into_parcels,
    vertex_to_face,
    add_surface_overlay,
    add_points_overlay,
    add_network_overlay,
    build_network,
    node_coor_from_parcels,
    add_node_variable,
    add_edge_variable,
    scalar_focus_camera,
    closest_ortho_camera,
    planar_sweep_camera,
    auto_camera,
    plot_to_image,
    plot_final_image,
    plot_to_html,
    plot_to_display,
    save_snapshots,
    save_figure,
    save_grid,
    svg_element,
    text_element,
    pyplot_element,
)
