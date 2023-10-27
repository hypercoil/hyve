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
    layout_inner = floating + floating_inner << ((0.1, 0.4), (0.8, 0.2))
    layout = anchor + layout_inner << ((0.7, 0.1), (0.2, 0.8))
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
    base = anchor + floating << ((0.1, 0.1), (0.8, 0.8))
    layout = (
        base | base << (1 / 2)) / (
        base | base << (1 / 2)) << (1 / 2)
    layout.partition(400, 400)
    cells = list(layout)
    def check_cells(cells):
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

    check_cells(cells)

    annotated0 = base.annotate(
        {0: {'i': 0}, 1: {'i': 1}, 2: {'i': 8}, 3: {'i': 9}}
    )
    annotated1 = base.annotate(
        {0: {'i': 2}, 1: {'i': 3}, 2: {'i': 10}, 3: {'i': 11}}
    )
    annotated2 = base.annotate(
        {0: {'i': 4}, 1: {'i': 5}, 2: {'i': 12}, 3: {'i': 13}}
    )
    annotated3 = base.annotate(
        {0: {'i': 6}, 1: {'i': 7}, 2: {'i': 14}, 3: {'i': 15}}
    )
    annotated = (
        annotated0 | annotated1 << (1 / 2)) / (
        annotated2 | annotated3 << (1 / 2)) << (1 / 2)
    annotated.partition(400, 400)
    cells = list(annotated)
    check_cells(cells)
    assert len(annotated) == 16
    for i in range(len(annotated)):
        assert annotated.annotations[i]['i'] == i

    annotated0 = base.annotate(
        {0: {'i': 0}, 1: {'i': None}, 2: {'i': 3}, 3: {'i': 4}}
    )
    annotated1 = base.annotate(
        {0: {'i': 1}, 1: {'i': 2}, 2: {'i': 5}, 3: {'i': 6}}
    )
    annotated = annotated0 % annotated1 << 1
    annotated.partition(400, 400)
    assert len(annotated) == 7
    for i in range(len(annotated)):
        assert annotated.annotations[i]['i'] == i
        if i > 2:
            assert type(annotated[i].root).__name__ == 'FloatingCellLayout'
    cells = list(annotated)
    assert cells[0].cell_loc == (0, 0)
    assert cells[0].cell_dim == (200, 400)
    assert cells[1].cell_loc == (200, 0)
    assert cells[1].cell_dim == (100, 400)
    assert cells[2].cell_loc == (300, 0)
    assert cells[2].cell_dim == (100, 400)
    assert cells[3].cell_loc == (40, 40)
    assert cells[3].cell_dim == (320, 80)
    assert cells[4].cell_loc == (40, 120)
    assert cells[4].cell_dim == (320, 240)
    assert cells[5].cell_loc == (220, 40)
    assert cells[5].cell_dim == (160, 80)
    assert cells[6].cell_loc == (220, 120)
    assert cells[6].cell_dim == (160, 240)

    annotated_a0 = anchor.annotate({0: {'i': 0}, 1: {'i': 1}})
    annotated_a1 = anchor.annotate({0: {'i': 2}, 1: {'i': 3}})
    annotated_a2 = anchor.annotate({0: {'i': 4}, 1: {'i': 5}})
    annotated_a3 = anchor.annotate({0: {'i': 6}, 1: {'i': 7}})
    annotated_f0 = floating.annotate({0: {'i': 8}, 1: {'i': 9}})
    annotated_f1 = floating.annotate({0: {'i': 10}, 1: {'i': 11}})
    annotated_f2 = floating.annotate({0: {'i': 12}, 1: {'i': 13}})
    annotated_f3 = floating.annotate({0: {'i': 14}, 1: {'i': 15}})
    annotated0 = annotated_a0 + annotated_f0 << ((0.1, 0.1), (0.8, 0.8))
    annotated1 = annotated_a1 + annotated_f1 << ((0.1, 0.1), (0.8, 0.8))
    annotated2 = annotated_a2 + annotated_f2 << ((0.1, 0.1), (0.8, 0.8))
    annotated3 = annotated_a3 + annotated_f3 << ((0.1, 0.1), (0.8, 0.8))
    annotated = (
        (annotated0 | annotated1 << (1 / 2)) /
        (annotated2 | annotated3 << (1 / 2)) << (1 / 2)
    )
    annotated.partition(400, 400)
    cells = list(annotated)
    check_cells(cells)
    assert len(annotated) == 16
    for i in range(len(annotated)):
        assert annotated.annotations[i]['i'] == i

    layout0 = Cell() | Cell() << (1 / 2)
    layout1 = Cell() / Cell() << (1 / 4)
    onecell = Cell()
    annotated0 = layout0.annotate({0: {'x': 0}, 1: {'x': 1}})
    annotated1 = layout1.annotate({0: {'y': 0}, 1: {'y': 1}})
    annotated2 = onecell.annotate({0: {'float0': True}})
    annotated3 = onecell.annotate({0: {'float1': True}})
    annotated12 = annotated1 + annotated2 << ((0.1, 0.6), (0.8, 0.3))
    annotated03 = annotated0 + annotated3 << ((0.6, 0.1), (0.3, 0.8))
    annotated = annotated0 * annotated12
    annotated.partition(400, 400)
    cells = list(annotated)
    annotations = annotated.annotations
    assert cells[0].cell_loc == (0, 0)
    assert cells[0].cell_dim == (200, 100)
    assert annotations[0] == {'x': 0, 'y': 0}
    assert cells[1].cell_loc == (0, 100)
    assert cells[1].cell_dim == (200, 300)
    assert annotations[1] == {'x': 0, 'y': 1}
    assert cells[2].cell_loc == (200, 0)
    assert cells[2].cell_dim == (200, 100)
    assert annotations[2] == {'x': 1, 'y': 0}
    assert cells[3].cell_loc == (200, 100)
    assert cells[3].cell_dim == (200, 300)
    assert annotations[3] == {'x': 1, 'y': 1}
    assert cells[4].cell_loc == (20, 240)
    assert cells[4].cell_dim == (160, 120)
    assert annotations[4] == {'float0': True, 'x': 0}
    assert cells[5].cell_loc == (220, 240)
    assert cells[5].cell_dim == (160, 120)
    assert annotations[5] == {'float0': True, 'x': 1}

    # Really I don't think you should do this in practice, but it's possible
    annotated = annotated03 * annotated12
    annotations = annotated.annotations
    cells = list(annotated.partition(400, 400))
    assert cells[0].cell_loc == (0, 0)
    assert cells[0].cell_dim == (200, 100)
    assert annotations[0] == {'x': 0, 'y': 0}
    assert cells[1].cell_loc == (0, 100)
    assert cells[1].cell_dim == (200, 300)
    assert annotations[1] == {'x': 0, 'y': 1}
    assert cells[2].cell_loc == (200, 0)
    assert cells[2].cell_dim == (200, 100)
    assert annotations[2] == {'x': 1, 'y': 0}
    assert cells[3].cell_loc == (200, 100)
    assert cells[3].cell_dim == (200, 300)
    assert annotations[3] == {'x': 1, 'y': 1}
    assert cells[4].cell_loc == (20, 240)
    assert cells[4].cell_dim == (160, 120)
    assert annotations[4] == {'x': 0, 'float0': True}
    assert cells[5].cell_loc == (220, 240)
    assert cells[5].cell_dim == (160, 120)
    assert annotations[5] == {'x': 1, 'float0': True}
    # TODO: need to double check the below -- something here isn't right
    assert cells[6].cell_loc == (240, 40)
    assert cells[6].cell_dim == (120, 80)
    assert annotations[6] == {'y': 0, 'float1': True}
    assert cells[7].cell_loc == (240, 120)
    assert cells[7].cell_dim == (120, 240)
    assert annotations[7] == {'y': 1, 'float1': True}
    assert cells[8].cell_loc == (252, 232)
    assert cells[8].cell_dim == (96, 96)
    assert annotations[8] == {'float0': True, 'float1': True}


def test_break():
    layout0 = Cell() | Cell() | Cell() << (1 / 3)
    layout1 = Cell() | Cell() << (1 /2)
    layout2 = Cell() / Cell() << (1 / 2)
    layout = layout0 * layout1 * layout2
    left, right = layout @ 0
    assert len(left), len(right) == (2, 10)
    left, right = layout @ 1
    assert len(left), len(right) == (4, 8)
    left, right = layout @ 2
    assert len(left), len(right) == (6, 6)
    left, right = layout @ 3
    assert len(left), len(right) == (8, 4)
    left, right = layout @ 4
    assert len(left), len(right) == (10, 2)

    assert len(list(layout.breakpoints)) == 5
    assert len(list(layout[0].parent.breakpoints)) == 1
    assert len(list(layout[0].parent.parent.breakpoints)) == 1

    layout = layout1 * layout0 * layout2
    breakpoints = list(layout.breakpoints)
    bpdict = dict(zip(breakpoints, range(len(breakpoints))))
    # not intuitive, because of operator precedence
    assert breakpoints[0] is layout.left
    assert breakpoints[1] is layout.left.right
    assert breakpoints[2] is layout
    assert breakpoints[3] is layout.right
    assert breakpoints[4] is layout.right.right
    left, right = layout @ 0
    assert len(left), len(right) == (2, 10)
    left, right = layout @ 1
    assert len(left), len(right) == (4, 8)
    left, right = layout @ 2
    assert len(left), len(right) == (6, 6)
    left, right = layout @ 3
    assert len(left), len(right) == (8, 4)
    left, right = layout @ 4
    assert len(left), len(right) == (10, 2)

    layout = (
        layout1 * layout1 * layout1 * layout0 * layout1 * layout2 * layout1
    )
    for i in range(47):
        left, right = layout @ i
        assert len(left), len(right) == (6 * (i + 1), 192 - 6 * (i + 1))
