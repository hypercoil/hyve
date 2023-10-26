# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for layout representation
"""
from hyve.layout import CellLayout, Cell, hsplit, vsplit, grid, float_layout


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
    # layout = (
    #     (Cell() | (1 / 2) | (Cell() /
    #                          (1 / 4) /
    #                          Cell())
    #     ) / (1 / 3) / (
    #         Cell(),
    #         Cell(),
    #     )
    # )
    # layout = (
    #     (Cell() | (Cell() /
    #                Cell() <<
    #                (1 / 4)) << (1 / 2)) /
    #      Cell() /
    #      Cell() <<
    #      (1 / 3)
    # )
    layout = ((
        Cell() /
        (Cell() | Cell() << (1 / 4)) <<
        (1 / 2))| Cell() | Cell() << (1 / 3)
    )
    layout_inner = Cell() / Cell() / Cell() << (1 / 4)
    layout_sub = (layout % layout_inner << 1).partition(120, 120)
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
    layout0 = Cell() / Cell() << (1 / 3)
    layout1 = Cell() | Cell() << (1 / 4)
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


def test_layout_floating():
    anchor = Cell() | Cell() | Cell() | Cell() << (1 / 5)
    floating = Cell() / Cell() << (1 / 3)
    floating_inner = Cell() | Cell() << (1 / 2)
    layout_inner = float_layout(
        floating=floating_inner,
        anchor=floating,
        loc_rel=(0.1, 0.4),
        dim_rel=(0.8, 0.2),
    )
    layout = float_layout(
        floating=layout_inner,
        anchor=anchor,
        loc_rel=(0.7, 0.1),
        dim_rel=(0.2, 0.8),
    )
    for i, cell in enumerate(layout):
        if i < 4:
            assert cell.root is layout
        elif i < 6:
            assert cell.root is layout.floating[0]
        else:
            assert cell.root is layout.floating[1]

    layout.partition(500, 500)
    cells = list(layout)
    assert cells[0].cell_loc == (0, 0)
    assert cells[0].cell_dim == (100, 500)
    assert cells[1].cell_loc == (100, 0)
    assert cells[1].cell_dim == (100, 500)
    assert cells[2].cell_loc == (200, 0)
    assert cells[2].cell_dim == (100, 500)
    assert cells[3].cell_loc == (300, 0)
    assert cells[3].cell_dim == (200, 500)

    # Total area is (100, 400)
    assert cells[4].cell_loc == (350, 50)
    assert cells[4].cell_dim == (100, 133)
    assert cells[5].cell_loc == (350, 183)
    assert cells[5].cell_dim == (100, 267)

    # Total area is (80, 80)
    assert cells[6].cell_loc == (360, 210)
    assert cells[6].cell_dim == (40, 80)
    assert cells[7].cell_loc == (400, 210)
    assert cells[7].cell_dim == (40, 80)

    anchor = Cell() | Cell() << (1 / 2)
    floating = Cell() / Cell() << (1 / 4)
    base = float_layout(
        floating=floating,
        anchor=anchor,
        loc_rel=(0.1, 0.1),
        dim_rel=(0.8, 0.8),
    )
    layout = (
        base | base << (1 / 2)) / (
        base | base << (1 / 2)) << (1 / 2)
    layout.partition(400, 400)
    cells = list(layout)
    assert cells[0].cell_loc == (0, 0)
    assert cells[0].cell_dim == (100, 200)
    assert cells[1].cell_loc == (100, 0)
    assert cells[1].cell_dim == (100, 200)
    assert cells[2].cell_loc == (200, 0)
    assert cells[2].cell_dim == (100, 200)
    assert cells[3].cell_loc == (300, 0)
    assert cells[3].cell_dim == (100, 200)
    assert cells[4].cell_loc == (0, 200)
    assert cells[4].cell_dim == (100, 200)
    assert cells[5].cell_loc == (100, 200)
    assert cells[5].cell_dim == (100, 200)
    assert cells[6].cell_loc == (200, 200)
    assert cells[6].cell_dim == (100, 200)
    assert cells[7].cell_loc == (300, 200)
    assert cells[7].cell_dim == (100, 200)

    # Total area is (160, 160)
    assert cells[8].cell_loc == (20, 20)
    assert cells[8].cell_dim == (160, 40)
    assert cells[9].cell_loc == (20, 60)
    assert cells[9].cell_dim == (160, 120)
    assert cells[10].cell_loc == (220, 20)
    assert cells[10].cell_dim == (160, 40)
    assert cells[11].cell_loc == (220, 60)
    assert cells[11].cell_dim == (160, 120)
    assert cells[12].cell_loc == (20, 220)
    assert cells[12].cell_dim == (160, 40)
    assert cells[13].cell_loc == (20, 260)
    assert cells[13].cell_dim == (160, 120)
    assert cells[14].cell_loc == (220, 220)
    assert cells[14].cell_dim == (160, 40)
    assert cells[15].cell_loc == (220, 260)
    assert cells[15].cell_dim == (160, 120)
