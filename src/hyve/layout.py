# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Layout representation
~~~~~~~~~~~~~~~~~~~~~
Classes for representing layouts of data
"""
from typing import Literal, Optional


class CellLayout:
    """
    Binary tree layout of cells, addressable and iterable in hierarchical
    left-to-right and top-to-bottom order
    """
    # It's mutable and TBH I hate it, but it makes implementing the
    # iteration logic much simpler
    def __init__(
        self,
        left: Optional['CellLayout'],
        right: Optional['CellLayout'],
        split_orientation: Optional[Literal['h', 'v']],
        split_position: Optional[float],
    ):
        self.left = left
        self.right = right
        self.split_orientation = split_orientation
        self.split_position = split_position
        self.parent = None
        self.cell_loc = None
        self.cell_dim = None

    def _leftmost(self) -> 'CellLayout':
        if self.left is None:
            return self
        return self.left._leftmost()

    def _next(self) -> Optional['CellLayout']:
        if self.right is not None:
            return self.right._leftmost()
        elif self.parent is None:
            return None
        elif self.parent.left is self:
            return self.parent
        else:
            cur_node = self
            while (
                cur_node.parent is not None
                and cur_node.parent.right is cur_node
            ):
                cur_node = cur_node.parent
            return cur_node.parent

    def _next_leaf(self) -> Optional['CellLayout']:
        candidate = self._next()
        while candidate is not None and not isinstance(candidate, Cell):
            candidate = candidate._next()
        return candidate

    def __getitem__(self, index):
        cur_idx = 0
        cur_cell = self._leftmost()
        while cur_idx < index:
            cur_cell = cur_cell._next_leaf()
            cur_idx += 1
        return cur_cell

    def __next__(self):
        return self._next_leaf()

    def __iter__(self):
        cur_cell = self._leftmost()
        while cur_cell is not None:
            yield cur_cell
            cur_cell = cur_cell._next_leaf()

    def __repr__(self):
        return (
            f'CellLayout(left={self.left}, right={self.right}, '
            f'orientation={self.split_orientation}, '
            f'position={self.split_position}'
            ')'
        )

    def __len__(self):
        return sum(1 for _ in self)

    def partition(self, width, height, x_offset=0, y_offset=0, padding=0):
        """
        Partition a canvas into a grid of cells specified by the layout
        """
        if self.split_orientation is None:
            assert self.split_position is None
        elif self.split_orientation == 'v':
            left_width = round(self.split_position * width)
            right_width = width - left_width
            left_height = right_height = height
            left_x_offset = x_offset
            right_x_offset = x_offset + left_width
            left_y_offset = right_y_offset = y_offset
        elif self.split_orientation == 'h':
            left_height = round(self.split_position * height)
            right_height = height - left_height
            left_width = right_width = width
            left_y_offset = y_offset
            right_y_offset = y_offset + left_height
            left_x_offset = right_x_offset = x_offset
        else:
            raise ValueError(
                f'Invalid split orientation: {self.split_orientation}'
            )
        if self.left is not None:
            self.left.partition(
                width=left_width,
                height=left_height,
                x_offset=left_x_offset,
                y_offset=left_y_offset,
                padding=padding,
            )
        if self.right is not None:
            self.right.partition(
                width=right_width,
                height=right_height,
                x_offset=right_x_offset,
                y_offset=right_y_offset,
                padding=padding,
            )
        self.cell_loc = (x_offset + padding, y_offset + padding)
        self.cell_dim = (width - 2 * padding, height - 2 * padding)
        return self


class Cell(CellLayout):
    """
    Leaf cell in a binary layout tree
    """
    def __init__(self):
        super().__init__(
            left=None,
            right=None,
            split_orientation=None,
            split_position=None
        )

    def __repr__(self):
        return f'Cell(loc={self.cell_loc}, dim={self.cell_dim})'


def split(
    orientation: Literal['h', 'v'],
    position: float,
    *cells: CellLayout,
) -> CellLayout:
    """
    Create a split layout
    """
    if len(cells) == 1:
        return cells[0]
    elif len(cells) == 2:
        left, right = cells
        layout = CellLayout(
            left=left,
            right=right,
            split_orientation=orientation,
            split_position=position,
        )
        if left is not None:
            left.parent = layout
        if right is not None:
            right.parent = layout
        return layout
    else:
        new_position = position / (1 - position)
        return split(
            orientation,
            position,
            cells[0],
            split(orientation, new_position, *cells[1:])
        )


def vsplit(
    position: float,
    *cells: CellLayout,
) -> CellLayout:
    """
    Create a vertical split layout
    """
    return split('v', position, *cells)


def hsplit(
    position: float,
    *cells: CellLayout,
) -> CellLayout:
    """
    Create a horizontal split layout
    """
    return split('h', position, *cells)


def grid(
    nrows: int,
    ncols: int,
    order: Literal['row', 'col'] = 'row',
) -> CellLayout:
    """
    Create a grid layout
    """
    if order == 'row':
        layout = hsplit(
            1 / nrows,
            *[
                vsplit(
                    1 / ncols,
                    *[
                        Cell()
                        for _ in range(ncols)
                    ]
                )
                for _ in range(nrows)
            ]
        )
    elif order == 'col':
        layout = vsplit(
            1 / ncols,
            *[
                hsplit(
                    1 / nrows,
                    *[
                        Cell()
                        for _ in range(nrows)
                    ]
                )
                for _ in range(ncols)
            ]
        )
    else:
        raise ValueError(f'Invalid grid order: {order}')
    return layout
