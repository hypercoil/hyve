# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Constants
~~~~~~~~~
Plotting constants
"""
import inspect
from textwrap import dedent
from typing import Any, Optional

from lytemaps.datasets import fetch_fsaverage
from matplotlib import rcParams

DEFAULT_CMAP = None
DEFAULT_COLOR = 'white'
DEFAULT_WINDOW_SIZE = (1920, 1080)
TYPICAL_DPI = round(rcParams['figure.dpi'])

LAYER_CLIM_DEFAULT_VALUE = None
LAYER_CMAP_NEGATIVE_DEFAULT_VALUE = None
LAYER_CLIM_NEGATIVE_DEFAULT_VALUE = None
LAYER_COLOR_DEFAULT_VALUE = None
LAYER_ALPHA_DEFAULT_VALUE = 1.0
LAYER_BELOW_COLOR_DEFAULT_VALUE = None
LAYER_BLEND_MODE_DEFAULT_VALUE = 'source_over'

NETWORK_LAYER_BELOW_COLOR_DEFAULT_VALUE = None

SURF_SCALARS_DEFAULT_VALUE = None
SURF_SCALARS_CMAP_DEFAULT_VALUE = (None, None)
SURF_SCALARS_CLIM_DEFAULT_VALUE = 'robust'
SURF_SCALARS_BELOW_COLOR_DEFAULT_VALUE = None
SURF_SCALARS_LAYERS_DEFAULT_VALUE = None

POINTS_SCALARS_DEFAULT_VALUE = None
POINTS_SCALARS_CMAP_DEFAULT_VALUE = None
POINTS_SCALARS_CLIM_DEFAULT_VALUE = None
POINTS_SCALARS_BELOW_COLOR_DEFAULT_VALUE = (0.0, 0.0, 0.0, 0.0)
POINTS_SCALARS_LAYERS_DEFAULT_VALUE = None

NODE_COLOR_DEFAULT_VALUE = 'black'
NODE_RADIUS_DEFAULT_VALUE = 3.0
NODE_RLIM_DEFAULT_VALUE = (2, 10)
NODE_CMAP_DEFAULT_VALUE = DEFAULT_CMAP
NODE_CLIM_DEFAULT_VALUE = 'robust'
NODE_ALPHA_DEFAULT_VALUE = 1.0

EDGE_COLOR_DEFAULT_VALUE = 'edge_sgn'
EDGE_RADIUS_DEFAULT_VALUE = 'edge_val'
EDGE_RLIM_DEFAULT_VALUE = (0.1, 1.8)
EDGE_CMAP_DEFAULT_VALUE = 'RdYlBu'
EDGE_CLIM_DEFAULT_VALUE = (-1, 1)
EDGE_ALPHA_DEFAULT_VALUE = 1.0

TEXT_DEFAULT_CONTENT = ''
TEXT_DEFAULT_FONT = 'futura'
TEXT_DEFAULT_FONT_SIZE_MULTIPLIER = 0.2
TEXT_DEFAULT_FONT_COLOR = 'white'
TEXT_DEFAULT_FONT_OUTLINE_COLOR = 'black'
TEXT_DEFAULT_FONT_OUTLINE_MULTIPLIER = 0.5
TEXT_DEFAULT_BOUNDING_BOX_WIDTH = 256
TEXT_DEFAULT_BOUNDING_BOX_HEIGHT = 24
TEXT_DEFAULT_ANGLE = 0

RASTER_DEFAULT_BOUNDING_BOX_HEIGHT = 512
RASTER_DEFAULT_BOUNDING_BOX_WIDTH = 512
RASTER_DEFAULT_FORMAT = 'png'

SCALAR_BAR_DEFAULT_NAME = None
SCALAR_BAR_DEFAULT_BELOW_COLOR = None
SCALAR_BAR_DEFAULT_LENGTH = 256
SCALAR_BAR_DEFAULT_WIDTH = 24
SCALAR_BAR_DEFAULT_ORIENTATION = 'v'
SCALAR_BAR_DEFAULT_NUM_SIG_FIGS = 3
SCALAR_BAR_DEFAULT_FONT = 'futura'
# TODO: These are multiplied by the width of the scalar bar. This isn't great
#       because the actual width of the scalar bar is determined by figure
#       parameters rather than by the parameters here, which really only
#       configure the aspect ratio of the scalar bar. We should change this to
#       a fraction of the figure width, but this is difficult to do because
#       we don't have access to the figure width until we've already created
#       the figure.
SCALAR_BAR_DEFAULT_NAME_FONTSIZE_MULTIPLIER = 0.4
SCALAR_BAR_DEFAULT_LIM_FONTSIZE_MULTIPLIER = 0.35
SCALAR_BAR_DEFAULT_FONT_COLOR = 'white'
SCALAR_BAR_DEFAULT_FONT_OUTLINE_COLOR = 'black'
SCALAR_BAR_DEFAULT_FONT_OUTLINE_MULTIPLIER = 0.2
SCALAR_BAR_DEFAULT_LOC = None
SCALAR_BAR_DEFAULT_SIZE = None
SCALAR_BAR_DEFAULT_SPACING = 0.02

Tensor = Any

REQUIRED = inspect._empty


def descr(desc: str):
    return dedent(desc).strip()

DOCBASE = """
Create a visualisation of a surface, points, and/or networks.
""".strip()
RETBASE = """
visualisation : Any
    The visualisation.
""".strip()
DOCBUILDER = {
    'surf' : {
        'type': 'cortex.CortexTriSurface',
        'default': None,
        'desc': descr(
            'A surface to plot. If not specified, no surface will be plotted.'
        ),
    },
    'surf_projection' : {
        'type': 'str',
        'default': 'pial',
        'desc': descr(
            """
            The projection of the surface to plot. The projection must be
            available in the surface's ``projections`` attribute. For typical
            surfaces, available projections might include ``'pial'``,
            ``'inflated'``, ``veryinflated``, ``'white'``, and ``'sphere'``.
            """
        ),
    },
    'surf_scalars_nifti' : {
        'type': 'str',
        'default': None,
        'desc': descr(
            """
            The path to a NIfTI file containing scalar-valued data to be
            resampled over a surface geometry. The resampled data will be
            loaded into the surface's ``point_data`` attribute as {scalars_name}.
            """
        ),
    },
    'surf_alpha' : {
        'type': 'float',
        'default': 1.0,
        'desc': descr('The opacity of the surface.'),
    },
    'surf_scalars' : {
        'type': 'str',
        'default': None,
        'desc': descr(
            """
            The name of the scalars to plot on the surface as the base layer.
            The scalars must be available in the surface's ``point_data``
            attribute. If not specified, no scalars will be plotted on the
            surface as the base layer.
            """
        ),
    },
    'surf_scalars_boundary_color' : {
        'type': 'str',
        'default': 'black',
        'desc': descr(
            """
            The color of the boundary between the surface and the background.
            Note that this boundary is only visible if
            ``surf_scalars_boundary_width`` is greater than 0.
            """
        ),
    },
    'surf_scalars_boundary_width' : {
        'type': 'int',
        'default': 0,
        'desc': descr(
            """
            The width of the boundary between the surface and the background.
            If set to 0, no boundary will be plotted.
            """
        ),
    },
    'surf_scalars_cmap' : {
        'type': 'str',
        'default': (None, None),
        'desc': descr(
            """
            The colormap to use for the surface scalars. If a tuple is
            specified, the first element is the colormap to use for the left
            hemisphere, and the second element is the colormap to use for the
            right hemisphere. If a single colormap is specified, it will be
            used for both hemispheres.
            """
        ),
    },
    'surf_scalars_clim' : {
        'type': 'str or tuple',
        'default': 'robust',
        'desc': descr(
            """
            The colormap limits to use for the surface scalars. If a tuple is
            specified, the first element is the colormap limits to use for the
            left hemisphere, and the second element is the colormap limits to
            use for the right hemisphere. If a single value is specified, it
            will be used for both hemispheres. If set to ``'robust'``, the
            colormap limits will be set to the 5th and 95th percentiles of
            the data.
            .. warning::
                If the colormap limits are set to ``'robust'``, the colormap
                limits will be calculated based on the data in the surface
                scalars, separately for each hemisphere. This means that the
                colormap limits may be different for each hemisphere, and the
                colors in the colormap may not be aligned between hemispheres.
            """
        ),
    },
    'surf_scalars_below_color' : {
        'type': 'str',
        'default': 'black',
        'desc': descr(
            'The color to use for values below the colormap limits.'
        ),
    },
    'surf_scalars_layers' : {
        'type': 'list of Layer',
        'default': None,
        'desc': descr(
            """
            A list of layers to plot on the surface. Each layer is defined by
            a ``Layer`` object, which specifies the name of the layer, the
            colormap to use, the colormap limits, the color, the opacity, and
            the blend mode. If not specified, no layers will be plotted.
            """
        ),
    },
    'points' : {
        'type': 'PointDataCollection',
        'default': None,
        'desc': descr(
            """
            A collection of points to plot. If not specified, no points will
            be plotted.
            """
        ),
    },
    'points_scalars' : {
        'type': 'str',
        'default': None,
        'desc': descr(
            """
            The name of the scalars to plot on the points. The scalars must be
            available in the points' ``point_data`` attribute. If not
            specified, no scalars will be plotted.
            """
        ),
    },
    'points_alpha' : {
        'type': 'float',
        'default': 1.0,
        'desc': descr('The opacity of the points.'),
    },
    'points_scalars_cmap' : {
        'type': 'str',
        'default': None,
        'desc': descr('The colormap to use for the points scalars.'),
    },
    'points_scalars_clim' : {
        'type': 'tuple',
        'default': None,
        'desc': descr('The colormap limits to use for the points scalars.'),
    },
    'points_scalars_below_color' : {
        'type': 'str',
        'default': 'black',
        'desc': descr(
            'The color to use for values below the colormap limits.'
        ),
    },
    'points_scalars_layers' : {
        'type': 'list of Layer',
        'default': None,
        'desc': descr(
            """
            A list of layers to plot on the points. Each layer is defined by
            a ``Layer`` object, which specifies the name of the layer, the
            colormap to use, the colormap limits, the color, the opacity, and
            the blend mode. If not specified, no layers will be plotted.
            """
        ),
    },
    'networks' : {
        'type': 'NetworkDataCollection',
        'default': None,
        'desc': descr(
            """
            A collection of networks to plot. If not specified, no networks
            will be plotted. Each network in the collection must include a
            ``'coor'`` attribute, which specifies the coordinates of the
            nodes in the network. The coordinates must be specified as a
            ``(N, 3)`` array, where ``N`` is the number of nodes in the
            network. The collection may also optionally include a ``'nodes'``
            attribute, The node attributes must be specified as a
            ``pandas.DataFrame`` with ``N`` rows, where ``N`` is the number
            of nodes in the network. The collection may also optionally
            include an ``'edges'`` attribute, which specifies the edges in
            the network. The edge attributes must be specified as a
            ``pandas.DataFrame`` with ``M`` rows, where ``M`` is the number
            of edges in the network. Finally, the collection may also
            optionally include a ``lh_mask`` attribute, which is a
            boolean-valued array indicating which nodes belong to the left
            hemisphere.
            """
        ),
    },
    'node_color' : {
        'type': 'str or colour specification',
        'default': 'black',
        'desc': descr(
            """
            The color of the nodes. If ``node_values`` is specified, this
            argument can be used to specify a column in the table to use for
            the node colors.
            """
        ),
    },
    'node_radius' : {
        'type': 'float or str',
        'default': 3.0,
        'desc': descr(
            """
            The radius of the nodes. If ``node_values`` is specified, this
            argument can be used to specify a column in the table to use for
            the node radii.
            """
        ),
    },
    'node_radius_range' : {
        'type': 'tuple',
        'default': (2, 10),
        'desc': descr(
            """
            The range of node radii to use. The values in ``node_radius`` will
            be linearly scaled to this range.
            """
        ),
    },
    'node_cmap' : {
        'type': 'str or matplotlib colormap',
        'default': 'viridis',
        'desc': descr(
            """
            The colormap to use for the nodes. If ``node_values`` is
            specified, this argument can be used to specify a column in the
            table to use for the node colors.
            """
        ),
    },
    'node_clim' : {
        'type': 'tuple',
        'default': (0, 1),
        'desc': descr(
            """
            The range of values to map into the dynamic range of the colormap.
            """
        ),
    },
    'node_alpha' : {
        'type': 'float or str',
        'default': 1.0,
        'desc': descr(
            """
            The opacity of the nodes. If ``node_values`` is specified, this
            argument can be used to specify a column in the table to use for
            the node opacities.
            """
        ),
    },
    'edge_color' : {
        'type': 'str or colour specification',
        'default': 'edge_sgn',
        'desc': descr(
            """
            The color of the edges. If ``edge_values`` is specified, this
            argument can be used to specify a column in the table to use for
            the edge colors. By default, edges are colored according to the
            value of the ``'edge_sgn'`` column in ``edge_values``, which is 1
            for positive edges and -1 for negative edges when the edges are
            digested by the ``filter_adjacency_data`` function using the
            default settings.
            """
        ),
    },
    'edge_radius' : {
        'type': 'float or str',
        'default': 'edge_val',
        'desc': descr(
            """
            The radius of the edges. If ``edge_values`` is specified, this
            argument can be used to specify a column in the table to use for
            the edge radii. By default, edges are sized according to the value
            of the ``'edge_val'`` column in ``edge_values``, which is the
            absolute value of the edge weight when the edges are digested by
            the ``filter_adjacency_data`` function using the default settings.
            """
        ),
    },
    'edge_radius_range' : {
        'type': 'tuple',
        'default': (0.1, 1.8),
        'desc': descr(
            """
            The range of edge radii to use. The values in ``edge_radius`` will
            be linearly scaled to this range.
            """
        ),
    },
    'edge_cmap' : {
        'type': 'str or matplotlib colormap',
        'default': 'RdYlBu',
        'desc': descr(
            """
            The colormap to use for the edges. If ``edge_values`` is
            specified, this argument can be used to specify a column in the
            table to use for the edge colors.
            """
        ),
    },
    'edge_clim' : {
        'type': 'tuple',
        'default': (-1, 1),
        'desc': descr(
            """
            The range of values to map into the dynamic range of the colormap.
            """
        ),
    },
    'edge_alpha' : {
        'type': 'float or str',
        'default': 1.0,
        'desc': descr(
            """
            The opacity of the edges. If ``edge_values`` is specified, this
            argument can be used to specify a column in the table to use for
            the edge opacities.
            """
        ),
    },
    'num_edge_radius_bins' : {
        'type': 'int',
        'default': 10,
        'desc': descr(
            """
            The number of bins to use when binning the edges by radius. Because
            edges are intractable to render when there are many of them, this
            argument can be used to bin the edges by radius and render each bin
            separately. This can significantly improve performance when there
            are many edges but will result in a loss of detail.
            """
        ),
    },
    'network_layers' : {
        'type': 'list of NodeLayer',
        'default': None,
        'desc': descr(
            """
            A list of layers to plot on the networks. Each layer is defined by
            a ``NodeLayer`` object, which specifies the name of the layer,
            together with various parameters for the nodes and edges in the
            layer. If not specified, a single layer will be created for the
            first network in ``networks``.
            """
        ),
    },
    'layer_cmap' : {
        'type': 'str',
        'default': None,
        'desc': descr('The colormap to use for the {layer_name} scalars.'),
    },
    'layer_clim' : {
        'type': 'tuple',
        'default': None,
        'desc': descr(
            """
            The colormap limits to use for the {layer_name} scalars. The
            specified values determine the dynamic range of the colormap. If
            set to ``'robust'``, the colormap limits will be set to the 5th
            and 95th percentiles of the data (not supported for all
            geometries).
            """
        ),
    },
    'layer_cmap_negative' : {
        'type': 'str',
        'default': None,
        'desc': descr(
            """
            The colormap to use for negative values of the {layer_name}
            scalars. If this is specified, then separate colormaps will be
            used for positive and negative values of the {layer_name}
            scalars. If not specified, the same colormap will be used for
            all values of the {layer_name} scalars.
            """
        ),
    },
    'layer_clim_negative' : {
        'type': 'tuple',
        'default': None,
        'desc': descr(
            """
            The colormap limits to use for negative values of the {layer_name}
            scalars.
            """
        ),
    },
    'layer_color' : {
        'type': 'str',
        'default': None,
        'desc': descr(
            """
            If specified, use a solid color for all values for the scalars in
            layer {layer_name}. If not specified, the colormap will be used.
            """
        ),
    },
    'layer_alpha' : {
        'type': 'str',
        'default': None,
        'desc': descr(
            """
            The opacity to use for the scalars in layer {layer_name}. If not
            specified, the opacity will be set to 1.0.
            """
        ),
    },
    'layer_below_color' : {
        'type': 'str',
        'default': 'black',
        'desc': descr(
            """
            The color to use for values below the colormap limits for the
            layer {layer_name}. Specify an alpha of 0 to hide values below the
            colormap limits (e.g., for thresholding).
            """
        ),
    },
    'hemisphere' : {
        'type': 'str',
        'default': None,
        'desc': descr(
            """
            The hemisphere to plot. If not specified, both hemispheres will be
            plotted.
            """
        ),
    },
    'hemisphere_slack' : {
        'type': "float, None, or ``'default'``",
        'default': 'default',
        'desc': descr(
            """
            The amount of slack to add between the hemispheres when plotting
            both hemispheres. This argument is ignored if ``hemisphere`` is
            not specified. The slack is specified in units of hemisphere
            width. Thus, a slack of 1.0 means that the hemispheres will be
            plotted without any extra space or overlap between them. When the
            slack is greater than 1.0, the hemispheres will be plotted with
            extra space between them. When the slack is less than 1.0, the
            hemispheres will be plotted with some overlap between them. If the
            slack is set to ``'default'``, the slack will be set to 1.1 for
            projections that have overlapping hemispheres and None for
            projections that do not have overlapping hemispheres.
            """
        ),
    },
    'off_screen' : {
        'type': 'bool',
        'default': True,
        'desc': descr(
            """
            Whether to render the plot off-screen. If ``False``, a window will
            appear containing an interactive plot.
            """
        ),
    },
    'copy_actors' : {
        'type': 'bool',
        'default': True,
        'desc': descr(
            """
            Whether to copy the actors before returning them. If ``False``, the
            actors will be modified in-place.
            """
        ),
    },
    'theme' : {
        'type': 'PyVista plotter theme',
        'default': None,
        'desc': descr(
            """
            The PyVista plotter theme to use. If not specified, the default
            DocumentTheme will be used.
            """
        ),
    },
    'template' : {
        'type': 'str',
        'default': 'fsLR',
        'desc': descr(
            """
            String that identifies the template space to load the surface from.
            """
        ),
    },
    'load_mask' : {
        'type': 'bool',
        'default': True,
        'desc': descr(
            """
            Indicates whether the surface mask should be loaded (e.g., for
            medial wall removal).
            """
        ),
    },
    'left_surf' : {
        'type': 'str',
        'default': None,
        'desc': descr(
            'Path or reference to the left hemisphere surface file.'
        ),
    },
    'right_surf' : {
        'type': 'str',
        'default': None,
        'desc': descr(
            'Path or reference to the right hemisphere surface file.'
        ),
    },
    'left_mask' : {
        'type': 'str',
        'default': None,
        'desc': descr(
            'Path or reference to the left hemisphere mask file.'
        ),
    },
    'right_mask' : {
        'type': 'str',
        'default': None,
        'desc': descr(
            'Path or reference to the right hemisphere mask file.'
        ),
    },
    'use_single_plotter' : {
        'type': 'bool',
        'default': True,
        'desc': descr(
            """
            Indicates whether a single plotter should be used for all
            visualisations. This prevents segmentation faults when a large
            number of scenes are visualised, but might rarely cause problems.
            If ``False``, a new plotter will be created for each
            visualisation.
            """
        ),
    },
    'postprocessors' : {
        'type': 'mapping or None',
        'default': None,
        'desc': descr(
            """
            Scene postprocessors, for instance for capturing snapshots. It is
            strongly advised you do not use this argument directly; it is
            typically configured by transformations.
            """
        )
    },
    'elements' : {
        'type': 'sequence',
        'default': None,
        'desc': descr(
            'SVG elements to be added to the index of scenes'
        ),
    },
    'map_spec' : {
        'type': 'sequence of nested lists or tuples of strings',
        'default': None,
        'desc': descr(
            """
            Specifies the way that the scene plotter function should be mapped
            over sequential arguments. A list of argument names indicates that
            a new scene plotter should be called for every possible
            combination of the values of those arguments. A tuple of argument
            names indicates that the values of those arguments should be
            broadcast to the maximal shape and zipped together, and the scene
            plotter should be called once for each element of the resulting
            sequence. Lists and tuples can be nested arbitrarily deeply.
            """
        )
    },
}

def docbuilder(strip_defaults: bool = True):
    """Filter and return the DOCBUILDER dictionary."""
    if strip_defaults:
        return {
            k: {
                k2: v2
                for k2, v2 in v.items()
                if k2 != 'default'
            }
            for k, v in DOCBUILDER.items()
        }
    else:
        return DOCBUILDER


def neuromaps_fetch_fn(
    nmaps_fetch_fn: callable,
    density: str,
    suffix: str,
    hemi: Optional[int] = None,
):
    template = nmaps_fetch_fn(density=density)[suffix]
    if hemi is not None:
        hemi_map = {
            'L': 0,
            'R': 1,
        }
        return template[hemi_map[hemi]]
    else:
        return template


def template_dict():
    _fsLR = fsLR()
    _fsaverage = fsaverage()
    return {
        'fsLR': _fsLR,
        'fsaverage': _fsaverage,
        'fsaverage5': _fsaverage,
    }


class fsaverage:
    NMAPS_MASK_QUERY = {
        'nmaps_fetch_fn': fetch_fsaverage,
        'density': '41k',
        'suffix': 'medial',
    }
    NMAPS_COOR_QUERY = {
        'nmaps_fetch_fn': fetch_fsaverage,
        'density': '41k',
        'suffix': 'sphere',
    }
    NMAPS_COMPARTMENTS = {
        'L': 0,
        'R': 1,
    }
    TFLOW_COOR_QUERY = {
        'template': 'fsLR',
        'space': None,
        'density': '32k',
        'suffix': 'sphere',
    }
    TFLOW_COMPARTMENTS = {
        'L': {'hemi': 'lh'},
        'R': {'hemi': 'rh'},
    }


class fsLR:
    TFLOW_MASK_QUERY = {
        'template': 'fsLR',
        'density': '32k',
        'desc': 'nomedialwall',
    }
    TFLOW_COOR_QUERY = {
        'template': 'fsLR',
        'space': None,
        'density': '32k',
        'suffix': 'sphere',
    }
    TFLOW_COMPARTMENTS = {
        'L': {'hemi': 'L'},
        'R': {'hemi': 'R'},
    }


class CIfTIStructures:
    LEFT = 'CIFTI_STRUCTURE_CORTEX_LEFT'
    RIGHT = 'CIFTI_STRUCTURE_CORTEX_RIGHT'
