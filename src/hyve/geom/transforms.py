# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Geometric transformations
~~~~~~~~~~~~~~~~~~~~~~~~~
Geometric transformations containers.
"""
from typing import (
    Any,
    Literal,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

from ..const import Tensor
from .base import SubgeometryParameters
from .network import NetworkDataCollection
from .points import PointDataCollection
from .surf import CortexTriSurface


class GeomTransform(NamedTuple):
    name: str
    fit: callable
    transform: callable
    meta: Optional[callable] = None


def hemisphere_select_fit(
    key_geom: Optional[str] = 'surf',
    key_scalars: Optional[str] = None,
    *,
    hemisphere: Optional[str] = None,
    **params,
) -> Mapping[str, Any]:
    # by convention, None denotes both hemispheres (no selection)
    if hemisphere == 'both':
        hemisphere = None
    hemispheres = (
        (hemisphere,) if hemisphere is not None else ('left', 'right')
    )
    key_geom = params.get(key_geom, None)
    if key_geom is not None and key_scalars is not None:
        hemispheres = tuple(
            hemi for hemi in hemispheres
            if key_scalars is None or key_geom.present_in_hemisphere(
                hemi, key_scalars
            )
        )
    if len(hemispheres) == 1:
        hemispheres_str = hemispheres[0]
    else:
        hemispheres_str = 'both'
    return {
        'hemispheres': hemispheres,
        'hemispheres_str': hemispheres_str,
    }


def hemisphere_select_transform(
    fit_params: Mapping[str, Any],
    **params,
) -> Mapping[str, Any]:
    return {
        'hemispheres': fit_params.get('hemispheres', ('left', 'right')),
        'hemispheres_str': fit_params.get('hemispheres_str', 'both'),
    }


def hemisphere_select_meta(
    hemisphere: Optional[str] = None,
    key_geom: Optional[str] = 'surf',
    key_scalars: Optional[str] = None,
    **params,
) -> Mapping[str, Any]:
    return {
        'hemisphere': [
            hemisphere_select_fit(
                hemisphere=hemisphere,
                key_geom=key_geom,
                key_scalars=key_scalars,
                meta_call=True,
                **params,
            )['hemispheres_str']
        ],
    }


def hemisphere_select_assign_parameters(
    *,
    surf_scalars_cmap: Any,
    surf_scalars_clim: Any,
    surf_scalars_layers: Any,
    subgeom_params: Optional[SubgeometryParameters] = None,
) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    left = {}
    right = {}

    def assign_tuple(arg, name):
        left[name], right[name] = arg

    def assign_scalar(arg, name):
        left[name] = right[name] = arg

    def conditional_assign(condition, arg, name):
        if arg is not None and condition(arg):
            assign_tuple(arg, name)
        else:
            assign_scalar(arg, name)

    conditional_assign(
        lambda x: len(x) == 2,
        surf_scalars_cmap,
        'surf_scalars_cmap',
    )
    conditional_assign(
        lambda x: len(x) == 2 and isinstance(x[0], (tuple, list)),
        surf_scalars_clim,
        'surf_scalars_clim',
    )
    conditional_assign(
        lambda x: len(x) == 2 and isinstance(x[0], (tuple, list)),
        surf_scalars_layers,
        'surf_scalars_layers',
    )
    if subgeom_params is None:
        subgeom_params = {}
    if subgeom_params.get('left') is not None:
        left = {**subgeom_params.params.pop('left'), **left}
    if subgeom_params.get('right') is not None:
        right = {**subgeom_params.params.pop('right'), **right}
    return SubgeometryParameters(left=left, right=right, **subgeom_params)


def hemisphere_select_fit_surf(f: callable) -> callable:

    def _hemisphere_select_fit_surf(
        surf: Optional[CortexTriSurface] = None,
        surf_projection: Optional[str] = None,
        **params,
    ) -> Mapping[str, Any]:
        fit_params = f(**params, surf=surf, surf_projection=surf_projection)

        surf_hemi_params = hemisphere_select_assign_parameters(
            surf_scalars_cmap=params.get('surf_scalars_cmap'),
            surf_scalars_clim=params.get('surf_scalars_clim'),
            surf_scalars_layers=params.get('surf_scalars_layers'),
        )
        fit_params['hemisphere_parameters'] = surf_hemi_params

        return fit_params
    return _hemisphere_select_fit_surf


def hemisphere_select_fit_network(f: callable) -> callable:
    def _hemisphere_select_fit_network(**params) -> Mapping[str, Any]:
        fit_params = f(**params)
        return fit_params
    return _hemisphere_select_fit_network


def hemisphere_select_fit_points(f: callable) -> callable:
    def _hemisphere_select_fit_points(**params) -> Mapping[str, Any]:
        fit_params = f(**params)
        return fit_params
    return _hemisphere_select_fit_points


def hemisphere_select_transform_surf(f: callable) -> callable:
    # This function doesn't actually select anything, but it prepares the
    # hemisphere parameters for the surface geometric primitive to perform
    # any specified selection later.
    def _hemisphere_select_transform_surf(
        fit_params: Optional[Mapping[str, Any]] = None,
        **params,
    ) -> Mapping[str, Any]:
        result = f(fit_params=fit_params, **params)
        hemisphere_parameters = fit_params.get('hemisphere_parameters', None)
        if hemisphere_parameters is None:
            return result

        result['hemisphere_parameters'] = hemisphere_parameters
        return result
    return _hemisphere_select_transform_surf


def hemisphere_select_transform_network(f: callable) -> callable:
    def _hemisphere_select_transform_network(
        fit_params: Optional[Mapping[str, Any]] = None,
        **params,
    ) -> Mapping[str, Any]:
        result = f(fit_params=fit_params, **params)
        hemispheres_str = fit_params.get('hemispheres_str', 'both')
        networks = params.get('networks', None)
        if networks is not None:
            condition = None
            if hemispheres_str == 'left':
                if any([ds.lh_mask is None for ds in networks]):
                    condition = lambda coor, _, __: coor[:, 0] < 0
                else:
                    condition = lambda _, __, lh_mask: lh_mask
            elif hemispheres_str == 'right':
                if any([ds.lh_mask is None for ds in networks]):
                    condition = lambda coor, _, __: coor[:, 0] > 0
                else:
                    condition = lambda _, __, lh_mask: ~lh_mask
            if condition is not None:
                networks = networks.__class__(
                    ds.select(condition=condition) for ds in networks
                )
                result['networks'] = networks
        return result
    return _hemisphere_select_transform_network


def hemisphere_select_transform_points(f: callable) -> callable:
    def _hemisphere_select_transform_points(
        fit_params: Optional[Mapping[str, Any]] = None,
        **params,
    ) -> Mapping[str, Any]:
        result = f(fit_params=fit_params, **params)
        hemispheres_str = fit_params.get('hemispheres_str', 'both')
        points = params.get('points', None)
        if points is not None:
            condition = None
            if hemispheres_str == 'left':
                condition = lambda coor, _: coor[:, 0] < 0
            elif hemispheres_str == 'right':
                condition = lambda coor, _: coor[:, 0] > 0
            if condition is not None:
                points = points.__class__(
                    ds.select(condition=condition) for ds in points
                )
                result['points'] = points
        return result
    return _hemisphere_select_transform_points


def hemisphere_slack_fit(
    hemispheres: Sequence[str],
    surf_projection: Optional[str] = None,
    *,
    hemisphere_slack: Optional[Union[float, Literal['default']]] = 'default',
    **params,
) -> Mapping[str, Any]:
    if hemisphere_slack == 'default':
        proj_require_slack = {'inflated', 'veryinflated', 'sphere'}
        if surf_projection in proj_require_slack:
            hemisphere_slack = 1.1
        else:
            hemisphere_slack = None
    return {
        'hemispheres': hemispheres,
        'hemisphere_slack': hemisphere_slack,
    }


def hemisphere_slack_transform(
    fit_params: Mapping[str, Any],
    **params,
) -> Mapping[str, Any]:
    return {}


def hemisphere_slack_get_displacement(
    hemisphere_slack: float,
    hemi_gap: float,
    hw_left: float,
    hw_right: float,
) -> float:
    min_gap = hw_left + hw_right
    target_gap = min_gap * hemisphere_slack
    displacement = (target_gap - hemi_gap) / 2
    return displacement


def hemisphere_slack_get_displacement_from_coor(
    hemisphere_slack: float,
    coor: Tensor,
    left_mask: Tensor,
) -> float:
    hw_left = (
        coor[left_mask, 0].max()
        - coor[left_mask, 0].min()
    ) / 2
    hw_right = (
        coor[~left_mask, 0].max()
        - coor[~left_mask, 0].min()
    ) / 2
    hemi_gap = (
        coor[~left_mask, 0].max()
        + coor[~left_mask, 0].min()
    ) / 2 - (
        coor[left_mask, 0].max()
        + coor[left_mask, 0].min()
    ) / 2
    return hemisphere_slack_get_displacement(
        hemisphere_slack=hemisphere_slack,
        hemi_gap=hemi_gap,
        hw_left=hw_left,
        hw_right=hw_right,
    )


def hemisphere_slack_fit_surf(f: callable) -> callable:
    def _hemisphere_slack_fit_surf(
        surf: Optional[CortexTriSurface] = None,
        surf_projection: Optional[str] = None,
        **params,
    ) -> Mapping[str, Any]:
        fit_params = f(surf_projection=surf_projection, **params)
        displacement = fit_params.get('displacement', None)
        if displacement is not None:
            return fit_params

        hemispheres = fit_params.get('hemispheres', ())
        hemisphere_slack = fit_params.get('hemisphere_slack', None)
        if len(hemispheres) == 2 and hemisphere_slack is not None:
            if surf is not None:
                surf.left.project(surf_projection)
                surf.right.project(surf_projection)
                hw_left = (surf.left.bounds[1] - surf.left.bounds[0]) / 2
                hw_right = (surf.right.bounds[1] - surf.right.bounds[0]) / 2
                hemi_gap = surf.right.center[0] - surf.left.center[0]
                displacement = hemisphere_slack_get_displacement(
                    hemisphere_slack=hemisphere_slack,
                    hemi_gap=hemi_gap,
                    hw_left=hw_left,
                    hw_right=hw_right,
                )
                fit_params['displacement'] = displacement

        return fit_params
    return _hemisphere_slack_fit_surf


def hemisphere_slack_fit_network(f: callable) -> callable:
    def _hemisphere_slack_fit_network(
        networks: Optional[Sequence[NetworkDataCollection]] = None,
        **params,
    ) -> Mapping[str, Any]:
        fit_params = f(**params)
        displacement = fit_params.get('displacement', None)
        if displacement is not None:
            return fit_params

        hemispheres = fit_params.get('hemispheres', ())
        hemisphere_slack = fit_params.get('hemisphere_slack', None)
        if len(hemispheres) == 2 and hemisphere_slack is not None:
            if networks is not None:
                ref_coor = np.concatenate([n.coor for n in networks])
                if any([n.lh_mask is None for n in networks]):
                    left_mask = ref_coor[:, 0] < 0
                else:
                    left_mask = np.concatenate([n.lh_mask for n in networks])
                displacement = hemisphere_slack_get_displacement_from_coor(
                    hemisphere_slack=hemisphere_slack,
                    coor=ref_coor,
                    left_mask=left_mask,
                )
                fit_params['displacement'] = displacement

        return fit_params
    return _hemisphere_slack_fit_network


def hemisphere_slack_fit_points(f: callable) -> callable:
    def _hemisphere_slack_fit_points(
        points: Optional[Sequence[PointDataCollection]] = None,
        **params,
    ) -> Mapping[str, Any]:
        fit_params = f(**params)
        displacement = fit_params.get('displacement', None)
        if displacement is not None:
            return fit_params

        hemispheres = fit_params.get('hemispheres', ())
        hemisphere_slack = fit_params.get('hemisphere_slack', None)
        if len(hemispheres) == 2 and hemisphere_slack is not None:
            if points is not None:
                ref_coor = np.concatenate([p.points.points for p in points])
                left_mask = ref_coor[:, 0] < 0
                displacement = hemisphere_slack_get_displacement_from_coor(
                    hemisphere_slack=hemisphere_slack,
                    coor=ref_coor,
                    left_mask=left_mask,
                )
                fit_params['displacement'] = displacement

        return fit_params
    return _hemisphere_slack_fit_points


def hemisphere_slack_transform_surf(f: callable) -> callable:
    def _hemisphere_slack_transform_surf(
        surf: Optional[CortexTriSurface] = None,
        surf_projection: Optional[str] = None,
        fit_params: Optional[Mapping[str, Any]] = None,
        **params,
    ) -> Mapping[str, Any]:
        result = f(fit_params=fit_params, **params)
        displacement = fit_params.get('displacement', None)
        if displacement is None:
            return result

        hemispheres = fit_params.get('hemispheres', ())
        hemisphere_slack = fit_params.get('hemisphere_slack', None)
        if len(hemispheres) == 2 and hemisphere_slack is not None:
            if surf is not None:
                left = surf.left.translate((-displacement, 0, 0))
                right = surf.right.translate((displacement, 0, 0))
                surf = CortexTriSurface(left=left, right=right, mask=surf.mask)
                result['surf'] = surf
                result['surf_projection'] = f'{surf_projection}_translated'

        return result
    return _hemisphere_slack_transform_surf


def hemisphere_slack_transform_network(f: callable) -> callable:
    def _hemisphere_slack_transform_network(
        networks: Optional[Sequence[NetworkDataCollection]] = None,
        fit_params: Optional[Mapping[str, Any]] = None,
        **params,
    ) -> Mapping[str, Any]:
        result = f(fit_params=fit_params, **params)
        displacement = fit_params.get('displacement', None)
        if displacement is None:
            return result

        hemispheres = fit_params.get('hemispheres', ())
        hemisphere_slack = fit_params.get('hemisphere_slack', None)
        if len(hemispheres) == 2 and hemisphere_slack is not None:
            if networks is not None:
                if any([n.lh_mask is None for n in networks]):
                    def lh_condition(coor, _, __):
                        return coor[:, 0] < 0
                    def rh_condition(coor, _, __):
                        return coor[:, 0] > 0
                else:
                    def lh_condition(_, __, lh_mask):
                        return lh_mask
                    def rh_condition(_, __, lh_mask):
                        return ~lh_mask
                networks = networks.translate(
                    (-displacement, 0, 0),
                    condition=lh_condition,
                )
                networks = networks.translate(
                    (displacement, 0, 0),
                    condition=rh_condition,
                )
                result['networks'] = networks

        return result
    return _hemisphere_slack_transform_network


def hemisphere_slack_transform_points(f: callable) -> callable:
    def _hemisphere_slack_transform_points(
        points: Optional[Sequence[PointDataCollection]] = None,
        fit_params: Optional[Mapping[str, Any]] = None,
        **params,
    ) -> Mapping[str, Any]:
        result = f(fit_params=fit_params, **params)
        displacement = fit_params.get('displacement', None)
        if displacement is None:
            return result

        hemispheres = fit_params.get('hemispheres', ())
        hemisphere_slack = fit_params.get('hemisphere_slack', None)
        if len(hemispheres) == 2 and hemisphere_slack is not None:
            if points is not None:
                left_points = points.translate(
                    (-displacement, 0, 0),
                    condition=lambda coor, _: coor[:, 0] < 0,
                )
                right_points = points.translate(
                    (displacement, 0, 0),
                    condition=lambda coor, _: coor[:, 0] > 0
                )
                points = PointDataCollection(
                    lp + rp for lp, rp in zip(left_points, right_points)
                )
                result['points'] = points

        return result
    return _hemisphere_slack_transform_points


hemisphere_select = GeomTransform(
    name='hemisphere_select',
    fit=hemisphere_select_fit,
    transform=hemisphere_select_transform,
    meta=hemisphere_select_meta,
)


hemisphere_slack = GeomTransform(
    name='hemisphere_slack',
    fit=hemisphere_slack_fit,
    transform=hemisphere_slack_transform,
)
