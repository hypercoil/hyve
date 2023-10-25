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


def test_layout_substitute():
    layout = (
        (Cell() | (1 / 2) | (
            Cell() / (1 / 4) /
            Cell()
        )) / (1 / 3) /
        (
            Cell(),
            Cell(),
        )
    )
    layout_inner = Cell() | (1 / 4) | (Cell(), Cell())
    layout_sub = (layout % 1 % layout_inner).partition(120, 120)
    cells = list(layout_sub)
    assert cells[0].cell_loc == (0, 0)
    assert cells[0].cell_dim == (40, 60)
    assert cells[1].cell_loc == (0, 60)
    assert cells[1].cell_dim == (10, 15)
    assert cells[2].cell_loc == (0, 75)
    assert cells[2].cell_dim == (10, 15)
    assert cells[3].cell_loc == (0, 90)
    assert cells[3].cell_dim == (10, 30)
    assert cells[4].cell_loc == (10, 60)
    assert cells[4].cell_dim == (30, 60)
    assert cells[5].cell_loc == (40, 0)
    assert cells[5].cell_dim == (40, 120)
    assert cells[6].cell_loc == (80, 0)
    assert cells[6].cell_dim == (40, 120)


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

def test_layout_product():
    layout0 = Cell() | (1 / 3) | Cell()
    layout1 = Cell() / (1 / 4) / Cell()
    layout01 = (layout0 * layout1).partition(120, 120)
    cells = list(layout01)
    assert cells[0].cell_loc == (0, 0)
    assert cells[0].cell_dim == (30, 40)
    assert cells[1].cell_loc == (30, 0)
    assert cells[1].cell_dim == (90, 40)
    assert cells[2].cell_loc == (0, 40)
    assert cells[2].cell_dim == (30, 80)
    assert cells[3].cell_loc == (30, 40)
    assert cells[3].cell_dim == (90, 80)

    layout10 = (layout1 * layout0).partition(120, 120)
    cells = list(layout10)
    assert cells[0].cell_loc == (0, 0)
    assert cells[0].cell_dim == (30, 40)
    assert cells[1].cell_loc == (0, 40)
    assert cells[1].cell_dim == (30, 80)
    assert cells[2].cell_loc == (30, 0)
    assert cells[2].cell_dim == (90, 40)
    assert cells[3].cell_loc == (30, 40)
    assert cells[3].cell_dim == (90, 80)

    annotation0 = {
        0: {'x': 0},
        1: {'x': 1},
    }
    annotation1 = {
        0: {'y': 0},
        1: {'y': 1},
    }
    annotated0 = layout0.annotate(annotation0)
    annotated1 = layout1.annotate(annotation1)
    annotated01 = (annotated0 * annotated1).partition(120, 120)
    assert annotated01.annotations == {
        0: {'x': 0, 'y': 0},
        1: {'x': 0, 'y': 1},
        2: {'x': 1, 'y': 0},
        3: {'x': 1, 'y': 1},
    }
    for cell_a, cell_b in zip(annotated01.layout, layout01):
        assert cell_a.cell_loc == cell_b.cell_loc
        assert cell_a.cell_dim == cell_b.cell_dim

    annotated10 = (annotated1 * annotated0).partition(120, 120)
    assert annotated10.annotations == {
        0: {'x': 0, 'y': 0},
        1: {'x': 1, 'y': 0},
        2: {'x': 0, 'y': 1},
        3: {'x': 1, 'y': 1},
    }
    for cell_a, cell_b in zip(annotated10.layout, layout10):
        assert cell_a.cell_loc == cell_b.cell_loc
        assert cell_a.cell_dim == cell_b.cell_dim
