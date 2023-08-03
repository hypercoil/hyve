# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for layout representation
"""
from hyve.layout import CellLayout, Cell, hsplit, vsplit, grid


def test_simple_layout():
    layout = vsplit(1 / 3,
        hsplit(1 / 2,
            Cell(),
            vsplit(1 / 4,
                Cell(),
                Cell(),
            ),
        ),
        Cell(),
        Cell(),
    ).partition(120, 120)
    cells = list(layout)
    assert cells[0].cell_loc == (0, 0)
    assert cells[0].cell_dim == (40, 60)
    assert cells[1].cell_loc == (0, 60)
    assert cells[1].cell_dim == (10, 60)
    assert cells[2].cell_loc == (10, 60)
    assert cells[2].cell_dim == (30, 60)
    assert cells[3].cell_loc == (40, 0)
    assert cells[3].cell_dim == (40, 120)
    assert cells[4].cell_loc == (80, 0)
    assert cells[4].cell_dim == (40, 120)


def test_grid_layout():
    gridlayout = grid(n_cols=6, n_rows=5).partition(1200, 900)
    cells = list(gridlayout)
    assert all([cells[i].cell_dim == (200, 180) for i in range(30)])
    for i in range(5):
        for j in range(6):
            assert cells[6 * i + j].cell_loc == (200 * j, 180 * i)

    gridlayout = grid(n_cols=6, n_rows=5, order='col').partition(1200, 900)
    cells = list(gridlayout)
    assert all([cells[i].cell_dim == (200, 180) for i in range(30)])
    for i in range(5):
        for j in range(6):
            assert cells[5 * j + i].cell_loc == (200 * j, 180 * i)
