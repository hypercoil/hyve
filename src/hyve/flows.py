# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Visualisation control flow
~~~~~~~~~~~~~~~~~~~~~~~~~~
Functions for transforming the control flow of visualisation functions.
See also ``transforms.py`` for functions that transform the input and output
flows of visualisation functions.
"""
from typing import Any, Literal, Optional, Sequence

from conveyant import ichain
from conveyant.flows import join

from .prim import automap_unified_plotter_p


def plotdef(*pparams: Sequence[callable]) -> callable:
    return ichain(*pparams)(automap_unified_plotter_p)


def joindata(
    join_vars: Optional[Sequence[str]] = None,
    how: Literal['outer', 'inner'] = 'outer',
    fill_value: Any = None,
) -> callable:
    def joining_f(arg):
        arg = list(a for a in arg if a is not None)
        out = arg[0].join(arg[1:], how=how)
        if fill_value is not None:
            out = out.fillna(fill_value)
        return out

    return join(joining_f, join_vars)


def add_network_data(
    *pparams: Sequence[callable],
    how: Literal['outer', 'inner'] = 'outer',
    fill_value: Any = 0.0,
) -> callable:
    return joindata(
        join_vars= ('edge_values', 'node_values'),
        how=how,
        fill_value=fill_value,
    )(*pparams)
