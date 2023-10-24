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
import inspect
from typing import Any, Literal, Optional, Sequence

from conveyant import emulate_assignment, ichain, join, splice_docstring

from .const import DOCBASE, RETBASE, docbuilder
from .prim import automap_unified_plotter_p


def plotdef(*pparams: Sequence[callable]) -> callable:
    plot_f = ichain(*pparams)(automap_unified_plotter_p)
    # drop variadic parameters
    plot_f.__signature__ = inspect.signature(plot_f).replace(
        parameters=tuple(
            p for p in plot_f.__signature__.parameters.values()
            if p.kind != p.VAR_POSITIONAL
        )
    )
    plot_f = emulate_assignment()(plot_f)
    # TODO: build docstring here when it's done
    return splice_docstring(
        f=plot_f,
        template=docbuilder(),
        base_str=DOCBASE,
        returns=RETBASE,
    )


def _get_unique_parameters_and_make_signature(
    joined: callable,
    fs: Sequence[callable],
) -> callable:
    seen_params = set()
    unique_params = []
    for f in fs:
        f_params = []
        for name, param in f.__signature__.parameters.items():
            if name not in seen_params:
                seen_params.add(name)
                f_params.append(param)
        unique_params = f_params + unique_params
    joined.__signature__ = inspect.signature(joined).replace(
        parameters=unique_params
    )
    return joined


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

    return join(
        joining_f,
        join_vars,
        postprocess=_get_unique_parameters_and_make_signature,
    )


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
