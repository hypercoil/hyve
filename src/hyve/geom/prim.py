# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Geometric primitives
~~~~~~~~~~~~~~~~~~~~
Geometric primitives containers.
"""
from typing import NamedTuple, Mapping, Tuple, Optional

from .network import (
    plot_network_aux_f,
    plot_network_f,
)
from .points import (
    plot_points_aux_f,
    plot_points_f,
)
from .surf import (
    plot_surf_aux_f,
    plot_surf_f,
)
from .transforms import (
    hemisphere_select_fit_surf,
    hemisphere_select_transform_surf,
    hemisphere_select_fit_points,
    hemisphere_select_transform_points,
    hemisphere_select_fit_network,
    hemisphere_select_transform_network,
    hemisphere_slack_fit_surf,
    hemisphere_slack_transform_surf,
    hemisphere_slack_fit_points,
    hemisphere_slack_transform_points,
    hemisphere_slack_fit_network,
    hemisphere_slack_transform_network,
)


class GeomPrimitive(NamedTuple):
    name: str
    plot: callable
    meta: callable
    xfms: Mapping[str, Tuple[Optional[callable], Optional[callable]]]


plot_surf_p = GeomPrimitive(
    name='surf',
    plot=plot_surf_f,
    meta=plot_surf_aux_f,
    xfms={
        'hemisphere_select': (
            hemisphere_select_fit_surf,
            hemisphere_select_transform_surf,
        ),
        'hemisphere_slack': (
            hemisphere_slack_fit_surf,
            hemisphere_slack_transform_surf,
        ),
    },
)


plot_points_p = GeomPrimitive(
    name='points',
    plot=plot_points_f,
    meta=plot_points_aux_f,
    xfms={
        'hemisphere_select': (
            hemisphere_select_fit_points,
            hemisphere_select_transform_points,
        ),
        'hemisphere_slack': (
            hemisphere_slack_fit_points,
            hemisphere_slack_transform_points,
        ),
    },
)


plot_network_p = GeomPrimitive(
    name='network',
    plot=plot_network_f,
    meta=plot_network_aux_f,
    xfms={
        'hemisphere_select': (
            hemisphere_select_fit_network,
            hemisphere_select_transform_network,
        ),
        'hemisphere_slack': (
            hemisphere_slack_fit_network,
            hemisphere_slack_transform_network,
        ),
    },
)