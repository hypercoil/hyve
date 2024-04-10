# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Networks
~~~~~~~~
Network geometry data containers and geometric primitives.
"""
import dataclasses
from typing import Any, Mapping, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
import pyvista as pv

from .base import (
    _LayerBase,
    _property_vector,
    layer_rgba,
)
from ..elements import ScalarBarBuilder
from ..const import (
    Tensor,
    EDGE_ALPHA_DEFAULT_VALUE,
    EDGE_CLIM_DEFAULT_VALUE,
    EDGE_CMAP_DEFAULT_VALUE,
    EDGE_COLOR_DEFAULT_VALUE,
    EDGE_RADIUS_DEFAULT_VALUE,
    EDGE_RLIM_DEFAULT_VALUE,
    EDGE_RMAP_DEFAULT_VALUE,
    NETWORK_LAYER_BELOW_COLOR_DEFAULT_VALUE,
    NODE_ALPHA_DEFAULT_VALUE,
    NODE_CLIM_DEFAULT_VALUE,
    NODE_CMAP_DEFAULT_VALUE,
    NODE_COLOR_DEFAULT_VALUE,
    NODE_RADIUS_DEFAULT_VALUE,
    NODE_RMAP_DEFAULT_VALUE,
    NODE_RLIM_DEFAULT_VALUE,
)

RADIUS_GLYPH_SCALAR_EDGE = 0.01


@dataclasses.dataclass(frozen=True)
class EdgeLayer(_LayerBase):
    """Container for metadata to construct a single edge layer of a plot."""
    #TODO, maybe:
    # Right now, we neither use nor support negative parameters. Instead, it's
    # expected that the user will preprocess the input dataset, for instance
    # with a filter that separates positive and negative values into separate
    # columns.
    name: str
    color: str = EDGE_COLOR_DEFAULT_VALUE
    clim: Optional[Tuple[float, float]] = EDGE_CLIM_DEFAULT_VALUE
    radius: Union[float, str] = EDGE_RADIUS_DEFAULT_VALUE
    radius_negative: Optional[Union[float, str]] = None
    rlim: Tuple[float, float] = EDGE_RLIM_DEFAULT_VALUE
    rlim_negative: Optional[Tuple[float, float]] = None
    rlim_percentile: bool = False
    rmap: Optional[Union[callable, Tuple[float, float]]] = None
    rmap_negative: Optional[Union[callable, Tuple[float, float]]] = None
    alpha: Union[float, str] = EDGE_ALPHA_DEFAULT_VALUE
    below_color: Optional[Any] = NETWORK_LAYER_BELOW_COLOR_DEFAULT_VALUE


@dataclasses.dataclass(frozen=True)
class NodeLayer(_LayerBase):
    """Container for metadata to construct a single node layer of a plot."""
    #TODO, maybe:
    # Right now, we neither use nor support negative parameters. Instead, it's
    # expected that the user will preprocess the input dataset, for instance
    # with a filter that separates positive and negative values into separate
    # columns.
    name: str
    color: str = NODE_COLOR_DEFAULT_VALUE
    clim: Optional[Tuple[float, float]] = NODE_CLIM_DEFAULT_VALUE
    radius: Union[float, str] = NODE_RADIUS_DEFAULT_VALUE
    radius_negative: Optional[Union[float, str]] = None
    rlim: Tuple[float, float] = NODE_RLIM_DEFAULT_VALUE
    rlim_negative: Optional[Tuple[float, float]] = None
    rlim_percentile: bool = False
    rmap: Optional[Union[callable, Tuple[float, float]]] = None
    rmap_negative: Optional[Union[callable, Tuple[float, float]]] = None
    alpha: Union[float, str] = NODE_ALPHA_DEFAULT_VALUE
    below_color: Optional[Any] = NETWORK_LAYER_BELOW_COLOR_DEFAULT_VALUE
    edge_layers: Sequence[EdgeLayer] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(frozen=True)
class NetworkData:
    name: str
    coor: Tensor
    nodes: pd.DataFrame
    edges: Optional[pd.DataFrame] = None
    lh_mask: Optional[Tensor] = None

    def translate(
        self,
        translation: Sequence[float],
        condition: Optional[callable] = None,
    ) -> 'NetworkData':
        if not condition:
            coor = self.coor + np.array(translation)
        else:
            mask = condition(self.coor, self.nodes, self.lh_mask)
            coor = self.coor.copy()
            coor[mask] = coor[mask] + np.array(translation)
        return self.__class__(
            self.name,
            coor,
            self.nodes,
            self.edges,
            self.lh_mask,
        )


class NetworkDataCollection:
    def __init__(
        self,
        network_datasets: Optional[Sequence[NetworkData]] = None
    ):
        self.network_datasets = list(network_datasets) or []

    def add_network_dataset(self, network_dataset: NetworkData):
        return self.__class__(self.network_datasets + [network_dataset])

    def get_dataset(
        self,
        key: str,
        return_all: bool = False,
        strict: bool = True,
    ) -> Union[NetworkData, 'NetworkDataCollection']:
        indices = [
            i for i in range(len(self.network_datasets))
            if self.network_datasets[i].name == key
        ]
        if len(indices) == 0:
            raise KeyError(f'No node data with key {key}')
        if not return_all:
            if len(indices) > 1 and strict:
                raise KeyError(f'Multiple node data with key {key}')
            return self[indices[0]]
        else:
            return self.__class__([self[i] for i in indices])

    def get_node_dataset(
        self,
        key: str,
        return_all: bool = False,
        strict: bool = True,
    ) -> pd.DataFrame:
        node_datasets = [
            (i, ds.nodes)
            for (i, ds) in enumerate(self.network_datasets)
            if ds.nodes is not None
        ]
        ds = [
            (i, nodes) for (i, nodes) in node_datasets
            if key in nodes.columns
        ]
        if len(ds) == 0:
            raise KeyError(f'No node data with key {key}')
        if not return_all:
            if len(ds) > 1 and strict:
                raise KeyError(f'Multiple node data with key {key}')
            return ds[0][1][[key]]
        else:
            return [nodes[[key]] for _, nodes in ds]

    def get_edge_dataset(
        self,
        key: str,
        return_all: bool = False,
        strict: bool = True,
    ) -> Union[NetworkData, Sequence[NetworkData], 'NetworkDataCollection']:
        edge_datasets = [
            (i, ds.edges)
            for (i, ds) in enumerate(self.network_datasets)
            if ds.edges is not None
        ]
        ds = [
            (i, edges) for (i, edges) in edge_datasets
            if key in edges.columns
        ]
        if len(ds) == 0:
            raise KeyError(f'No edge data with key {key}')
        if not return_all:
            if len(ds) > 1 and strict:
                raise KeyError(f'Multiple edge data with key {key}')
            return ds[0][1][[key]]
        else:
            return [edges[[key]] for _, edges in ds]

    def translate(
        self,
        translation: Sequence[float],
        condition: Optional[callable] = None,
    ):
        return self.__class__([
            ds.translate(translation, condition=condition)
            for ds in self.network_datasets
        ])

    def present_in_hemisphere(self, hemi: str, key: str) -> bool:
        try:
            ds = self.get_dataset(key)
            mask = ds.lh_mask
        except KeyError:
            return False
        if mask is None or hemi == 'both':
            return True
        elif hemi == 'right':
            mask = ~mask
        return np.any(mask)

    def paint(
        self,
        plotter: pv.Plotter,
        layers: Sequence[NodeLayer],
        num_edge_radius_bins: int = 10,
        copy_actors: bool = False,
    ) -> Tuple[pv.Plotter, Sequence[ScalarBarBuilder]]:
        # TODO: See if we're better off merging the nodes and edges into a
        #       single mesh, or if there's any reason to keep them separate.
        scalar_bar_builders = ()
        for layer in layers:
            network = self.get_dataset(layer.name)
            node_coor = network.coor
            node_values = network.nodes
            edge_values = network.edges
            glyph, new_builder = build_nodes_mesh(
                node_values, node_coor, layer
            )
            plotter.add_mesh(
                glyph,
                scalars=f'{layer.name}_rgba',
                rgb=True,
                # shouldn't do anything here, but just in case
                copy_mesh=copy_actors,
            )

            layer_builders = new_builder
            for edge_layer in layer.edge_layers:
                glyphs, new_builder = build_edges_meshes(
                    edge_values,
                    node_coor,
                    edge_layer,
                    num_edge_radius_bins,
                )
                plotter.add_mesh(
                    glyphs,
                    scalars=f'{edge_layer.name}_rgba',
                    rgb=True,
                    copy_mesh=copy_actors,
                )
                if new_builder[0].name == layer_builders[0].name:
                    new_builder = (dataclasses.replace(
                        new_builder[0],
                        name=f'{layer.name} (edges)',
                    ),)
                layer_builders = layer_builders + new_builder
            scalar_bar_builders = scalar_bar_builders + layer_builders

        return plotter, scalar_bar_builders

    def __getitem__(self, key: int):
        return self.network_datasets[key]

    def __len__(self):
        return len(self.network_datasets)

    def __iter__(self):
        return iter(self.network_datasets)

    def __str__(self):
        return f'NetworkDataCollection({self.network_datasets})'

    def __repr__(self):
        return str(self)

    def __add__(self, other):
        return self.__class__(self.network_datasets + other.network_datasets)

    def __radd__(self, other):
        return self.__class__(other.network_datasets + self.network_datasets)


def build_edges_mesh(
    edge_values: pd.DataFrame,
    node_coor: np.ndarray,
    layer: EdgeLayer,
    radius: float,
    edges_noalpha: bool = False,
) -> Tuple[pv.PolyData, Sequence[ScalarBarBuilder]]:
    edge_values = edge_values.reset_index()
    # DataFrame indices begin at 1 after we filter them, but we need them to
    # begin at 0 for indexing into node_coor.
    target = edge_values.dst.values - 1
    source = edge_values.src.values - 1
    midpoints = (node_coor[target] + node_coor[source]) / 2
    orientations = node_coor[target] - node_coor[source]
    norm = np.linalg.norm(orientations, axis=-1)

    edges = pv.PolyData(midpoints)
    if layer.color not in edge_values.columns:
        scalars = None
        color = layer.color
    else:
        scalars = edge_values[layer.color].values
        color = None
    flayer = dataclasses.replace(layer, color=color)
    rgba, scalar_bar_builders = layer_rgba(flayer, scalars)
    if edges_noalpha:
        # We shouldn't have to do this, but for some reason either VTK or
        # PyVista is adding transparency even if we set alpha to 1 when we use
        # explicit RGBA to colour the mesh. Dropping the alpha channel
        # fixes this.
        rgba = rgba[:, :3]

    # TODO: This is a hack to get the glyphs to scale correctly.
    # The magic scalar is used to scale the glyphs to the correct radius.
    # Where does it come from? I have no idea. It's just what works. And
    # that, only roughly. No guarantees that it will work on new data.
    geom = pv.Cylinder(
        resolution=20,
        radius=RADIUS_GLYPH_SCALAR_EDGE * radius,
    )

    edges.point_data[f'{layer.name}_norm'] = norm
    edges.point_data[f'{layer.name}_vecs'] = orientations
    edges.point_data[f'{layer.name}_rgba'] = rgba
    glyph = edges.glyph(
        scale=f'{layer.name}_norm',
        orient=f'{layer.name}_vecs',
        geom=geom,
        factor=1,
    )
    return glyph, scalar_bar_builders


def build_edges_meshes(
    edge_values: pd.DataFrame,
    node_coor: np.ndarray,
    layer: EdgeLayer,
    num_radius_bins: int = 10,
) -> Tuple[pv.PolyData, Sequence[ScalarBarBuilder]]:
    if not isinstance(layer.radius, str):
        radius_str = 'edge_radius'
        edge_radius = np.full(len(edge_values), layer.radius)
        num_radius_bins = 1
    else:
        radius_str = layer.radius
        edge_radius = edge_values[radius_str].values
        edge_radius, _ = _property_vector(
            edge_radius,
            lim=layer.rlim,
            percentile=layer.rlim_percentile,
            mapper=layer.rmap,
        )
        num_radius_bins = min(num_radius_bins, len(edge_radius))
    bins = np.quantile(
        edge_radius,
        np.linspace(1 / num_radius_bins, 1, num_radius_bins),
    )
    # bins = np.linspace(
    #     edge_radius.min(),
    #     edge_radius.max(),
    #     num_radius_bins + 1,
    # )[1:]
    asgt = np.digitize(edge_radius, bins, right=True)
    #assert num_radius_bins == len(np.unique(bins)), (
    assert num_radius_bins == len(bins), (
        'Binning failed to produce the correct number of bins. '
        'This is likely a bug. Please report it at '
        'https://github.com/hypercoil/hyve/issues.'
    )
    #TODO: If colour is mapped by percentile, we should override it *before*
    # we loop over the bins, not after: the colour mapping should be
    # consistent across all bins.

    edges = pv.PolyData()
    for i in range(num_radius_bins):
        idx = asgt == i
        selected = edge_values[idx]
        if len(selected) == 0:
            continue
        # TODO: We're replacing the builders at every call. We definitely
        #       don't want to get multiple builders, but is this really the
        #       right way to do it? Double check to make sure it makes sense.
        mesh, scalar_bar_builders = build_edges_mesh(
            selected,
            node_coor,
            layer,
            bins[i],
        )
        edges = edges.merge(mesh)
    return edges, scalar_bar_builders


def build_nodes_mesh(
    node_values: pd.DataFrame,
    node_coor: np.ndarray,
    layer: NodeLayer,
) -> Tuple[pv.PolyData, Sequence[ScalarBarBuilder]]:
    node_values = node_values.reset_index()
    nodes = pv.PolyData(node_coor)
    if not isinstance(layer.radius, str):
        radius_str = 'node_radius'
        node_radius = np.full(len(node_coor), layer.radius)
    else:
        radius_str = layer.radius
        node_radius = node_values[radius_str].values
        node_radius, _ = _property_vector(
            node_radius,
            lim=layer.rlim,
            percentile=layer.rlim_percentile,
            mapper=layer.rmap,
        )

    if layer.color not in node_values.columns:
        scalars = None
        color = layer.color
    else:
        scalars = node_values[layer.color].values
        color = None
    if not isinstance(layer.alpha, str):
        alpha = layer.alpha
    else:
        alpha = None
    flayer = dataclasses.replace(layer, alpha=alpha, color=color)
    rgba, scalar_bar_builders = layer_rgba(flayer, scalars)
    if isinstance(layer.alpha, str):
        rgba[:, 3] = node_values[layer.alpha].values
    elif alpha == 1:
        # We shouldn't have to do this, but for some reason either VTK or
        # PyVista is adding transparency even if we set alpha to 1 when we use
        # explicit RGBA to colour the mesh. Dropping the alpha channel
        # fixes this.
        rgba = rgba[:, :3]

    nodes.point_data[radius_str] = node_radius
    nodes.point_data[f'{layer.name}_rgba'] = rgba
    glyph = nodes.glyph(
        scale=radius_str,
        orient=False,
        geom=pv.Icosphere(nsub=3),
    )
    return glyph, scalar_bar_builders


def plot_network_f(
    plotter: pv.Plotter,
    scalar_bar_builders: Sequence[ScalarBarBuilder],
    copy_actors: bool = False,
    *,
    networks: Optional[NetworkDataCollection] = None,
    node_color: Optional[str] = NODE_COLOR_DEFAULT_VALUE,
    node_radius: Union[float, str] = NODE_RADIUS_DEFAULT_VALUE,
    node_rmap: Optional[
        Union[callable, Tuple[float, float]]
    ] = NODE_RMAP_DEFAULT_VALUE,
    node_rlim: Tuple[float, float] = NODE_RLIM_DEFAULT_VALUE,
    node_cmap: Any = NODE_CMAP_DEFAULT_VALUE,
    node_clim: Tuple[float, float] = NODE_CLIM_DEFAULT_VALUE,
    node_alpha: Union[float, str] = NODE_ALPHA_DEFAULT_VALUE,
    edge_color: Optional[str] = EDGE_COLOR_DEFAULT_VALUE,
    edge_radius: Union[float, str] = EDGE_RADIUS_DEFAULT_VALUE,
    edge_rmap: Optional[
        Union[callable, Tuple[float, float]]
    ] = EDGE_RMAP_DEFAULT_VALUE,
    edge_rlim: Tuple[float, float] = EDGE_RLIM_DEFAULT_VALUE,
    edge_cmap: Any = EDGE_CMAP_DEFAULT_VALUE,
    edge_clim: Tuple[float, float] = EDGE_CLIM_DEFAULT_VALUE,
    edge_alpha: Union[float, str] = 1.0,
    num_edge_radius_bins: int = 10,
    network_layers: Optional[Sequence[NodeLayer]] = None,
) -> Tuple[pv.Plotter, Sequence[ScalarBarBuilder]]:
    if networks is not None:
        if network_layers is None or len(network_layers) == 0:
            # No point in multiple datasets without overlays, so we'll use the
            # first network's specifications to build the base layer.
            base_network = networks[0]
            network_name = base_network.name
            if base_network.edges is not None:
                base_edge_layers = [EdgeLayer(
                    name=network_name,
                    cmap=edge_cmap,
                    clim=edge_clim,
                    color=edge_color,
                    radius=edge_radius,
                    rmap=edge_rmap,
                    rlim=edge_rlim,
                    alpha=edge_alpha,
                )]
            else:
                base_edge_layers = []
            if base_network.coor is not None:
                base_layer = NodeLayer(
                    name=network_name,
                    cmap=node_cmap,
                    clim=node_clim,
                    color=node_color,
                    radius=node_radius,
                    rmap=node_rmap,
                    rlim=node_rlim,
                    alpha=node_alpha,
                    edge_layers=base_edge_layers,
                )
            network_layers = [base_layer]
        plotter, new_builders = networks.paint(
            plotter=plotter,
            layers=network_layers,
            num_edge_radius_bins=num_edge_radius_bins,
            copy_actors=copy_actors,
        )
        scalar_bar_builders = scalar_bar_builders + new_builders
    return plotter, scalar_bar_builders


def plot_network_aux_f(
    metadata: Mapping[str, Sequence[str]],
    *,
    networks: Optional[NetworkDataCollection] = None,
    node_color: Optional[str] = NODE_COLOR_DEFAULT_VALUE,
    node_radius: Union[float, str] = NODE_RADIUS_DEFAULT_VALUE,
    node_rmap: Optional[Union[callable, Tuple[float, float]]] = None,
    node_rlim: Tuple[float, float] = NODE_RLIM_DEFAULT_VALUE,
    node_cmap: Any = NODE_CMAP_DEFAULT_VALUE,
    node_clim: Tuple[float, float] = NODE_CLIM_DEFAULT_VALUE,
    node_alpha: Union[float, str] = NODE_ALPHA_DEFAULT_VALUE,
    edge_color: Optional[str] = EDGE_COLOR_DEFAULT_VALUE,
    edge_radius: Union[float, str] = EDGE_RADIUS_DEFAULT_VALUE,
    edge_rmap: Tuple[float, float] = None,
    edge_rlim: Tuple[float, float] = EDGE_RLIM_DEFAULT_VALUE,
    edge_cmap: Any = EDGE_CMAP_DEFAULT_VALUE,
    edge_clim: Tuple[float, float] = EDGE_CLIM_DEFAULT_VALUE,
    edge_alpha: Union[float, str] = 1.0,
    num_edge_radius_bins: int = 10,
    network_layers: Optional[Sequence[NodeLayer]] = None,
) -> Mapping[str, Sequence[str]]:
    node = networks is not None and any(
        network.coor is not None for network in networks
    )
    edge = networks is not None and any(
        network.edges is not None for network in networks
    )
    if node is not None:
        metadata['nodecolor'] = [node_color or None]
        metadata['noderadius'] = [node_radius or None]
        metadata['nodealpha'] = [node_alpha or None]
    if edge is not None:
        metadata['edgecolor'] = [edge_color or None]
        metadata['edgeradius'] = [edge_radius or None]
        metadata['edgealpha'] = [edge_alpha or None]
    if network_layers is not None:
        colors = '+'.join(layer.color for layer in network_layers)
        radii = '+'.join(layer.radius for layer in network_layers)
        alphas = '+'.join(layer.alpha for layer in network_layers)
        metadata['nodecolor'] = (
            [f'{metadata["nodecolor"][0]}+{colors}']
            if metadata['nodecolor'][0] is not None
            else [colors]
        )
        metadata['noderadius'] = (
            [f'{metadata["noderadius"][0]}+{radii}']
            if metadata['noderadius'][0] is not None
            else [radii]
        )
        metadata['nodealpha'] = (
            [f'{metadata["nodealpha"][0]}+{alphas}']
            if metadata['nodealpha'][0] is not None
            else [alphas]
        )

        if edge:
            colors = '+'.join(
                layer.edge_layers[0].color for layer in network_layers
            )
            radii = '+'.join(
                layer.edge_layers[0].radius for layer in network_layers
            )
            alphas = '+'.join(
                layer.edge_layers[0].alpha for layer in network_layers
            )
            metadata['edgecolor'] = (
                [f'{metadata["edgecolor"][0]}+{colors}']
                if metadata['edgecolor'][0] is not None
                else [colors]
            )
            metadata['edgeradius'] = (
                [f'{metadata["edgeradius"][0]}+{radii}']
                if metadata['edgeradius'][0] is not None
                else [radii]
            )
            metadata['edgealpha'] = (
                [f'{metadata["edgealpha"][0]}+{alphas}']
                if metadata['edgealpha'][0] is not None
                else [alphas]
            )
    return metadata
