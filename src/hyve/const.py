# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Constants
~~~~~~~~~
Plotting constants
"""
from typing import Any, Optional

from lytemaps.datasets import fetch_fsaverage

DEFAULT_CMAP = 'viridis'
DEFAULT_COLOR = 'white'
TYPICAL_DPI = 96

LAYER_CLIM_DEFAULT_VALUE = None
LAYER_CMAP_NEGATIVE_DEFAULT_VALUE = None
LAYER_CLIM_NEGATIVE_DEFAULT_VALUE = None
LAYER_COLOR_DEFAULT_VALUE = None
LAYER_ALPHA_DEFAULT_VALUE = 1.0
LAYER_BELOW_COLOR_DEFAULT_VALUE = (0.0, 0.0, 0.0, 0.0)
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
NODE_CLIM_DEFAULT_VALUE = (0, 1)
NODE_ALPHA_DEFAULT_VALUE = 1.0

EDGE_COLOR_DEFAULT_VALUE = 'edge_sgn'
EDGE_RADIUS_DEFAULT_VALUE = 'edge_val'
EDGE_RLIM_DEFAULT_VALUE = (0.1, 1.8)
EDGE_CMAP_DEFAULT_VALUE = 'RdYlBu'
EDGE_CLIM_DEFAULT_VALUE = (0, 1)
EDGE_ALPHA_DEFAULT_VALUE = 1.0

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
SCALAR_BAR_DEFAULT_NAME_FONTSIZE_MULTIPLIER = 0.5
SCALAR_BAR_DEFAULT_LIM_FONTSIZE_MULTIPLIER = 0.4
SCALAR_BAR_DEFAULT_FONT_COLOR = 'w'
SCALAR_BAR_DEFAULT_FONT_OUTLINE_COLOR = 'k'
SCALAR_BAR_DEFAULT_FONT_OUTLINE_WIDTH = 1.0

Tensor = Any


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
