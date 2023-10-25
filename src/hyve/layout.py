# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Layout representation
~~~~~~~~~~~~~~~~~~~~~
Classes for representing layouts of data
"""
import dataclasses
from typing import Literal, Mapping, Optional, Sequence, Tuple, Union


@dataclasses.dataclass(frozen=True)
class CellLayoutSubstitution:
    layout: 'CellLayout'
    index: int

    def __mod__(self, other: 'CellLayout'):
        return self.layout.substitute(self.index, other)


@dataclasses.dataclass(frozen=True)
class CellLayoutVerticalSplit:
    layout: 'CellLayout'
    position: float

    def __truediv__(self, other: Union['CellLayout', Sequence['CellLayout']]):
        if isinstance(other, CellLayout):
            other = [other]
        return vsplit(self.position, self.layout, *other)


@dataclasses.dataclass(frozen=True)
class CellLayoutHorizontalSplit:
    layout: 'CellLayout'
    position: float

    def __or__(self, other: Union['CellLayout', Sequence['CellLayout']]):
        if isinstance(other, CellLayout):
            other = [other]
        return hsplit(self.position, self.layout, *other)


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

    def __getitem__(self, index: int) -> 'CellLayout':
        cur_idx = 0
        cur_cell = self._leftmost()
        while cur_idx < index:
            cur_cell = cur_cell._next_leaf()
            cur_idx += 1
        return cur_cell

    def __next__(self) -> 'CellLayout':
        return self._next_leaf()

    def __iter__(self):
        cur_cell = self._leftmost()
        while cur_cell is not None:
            yield cur_cell
            cur_cell = cur_cell._next_leaf()

    def __repr__(self) -> str:
        return (
            f'CellLayout(left={self.left}, right={self.right}, '
            f'orientation={self.split_orientation}, '
            f'position={self.split_position}'
            ')'
        )

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def __mod__(self, other: int) -> 'CellLayoutSubstitution':
        return CellLayoutSubstitution(
            layout=self,
            index=other,
        )

    def __truediv__(self, other: float) -> 'CellLayoutVerticalSplit':
        return CellLayoutVerticalSplit(
            layout=self,
            position=other,
        )

    def __or__(self, other: float) -> 'CellLayoutHorizontalSplit':
        return CellLayoutHorizontalSplit(
            layout=self,
            position=other,
        )

    def __mul__(self, other: 'CellLayout') -> 'CellLayout':
        """Note that this 'product' is not commutative"""
        layout = self
        n = len(layout)
        for i in range(n, 0, -1):
            layout = layout.substitute(i - 1, other)
        return layout

    def copy(self) -> 'CellLayout':
        left = self.left.copy() if self.left is not None else None
        right = self.right.copy() if self.right is not None else None
        copy = self.__class__(
            left=left,
            right=right,
            split_orientation=self.split_orientation,
            split_position=self.split_position,
        )
        copy.cell_loc = self.cell_loc
        copy.cell_dim = self.cell_dim
        if left is not None:
            left.parent = copy
        if right is not None:
            right.parent = copy
        return copy

    def substitute(self, index: int, other: 'CellLayout') -> 'CellLayout':
        """Substitute a cell in the layout with another layout"""
        if len(self) <= 1:
            return other
        new = self.copy()
        other = other.copy()
        cur_idx = 0
        cur_cell = new._leftmost()
        while cur_idx < index:
            cur_cell = cur_cell._next_leaf()
            cur_idx += 1
        if cur_cell.parent.left is cur_cell:
            cur_cell.parent.left = other
        else:
            cur_cell.parent.right = other
        other.parent = cur_cell.parent
        return new

    def partition(
        self,
        width: int,
        height: int,
        x_offset: int = 0,
        y_offset: int = 0,
        padding: int = 0,
    ) -> 'CellLayout':
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

    def annotate(
        self,
        annotations: Mapping[int, Optional[Mapping]],
    ) -> 'AnnotatedLayout':
        """
        Annotate a layout with a dictionary of annotations
        """
        assert len(annotations) == len(self)
        return AnnotatedLayout(
            layout=self.copy(),
            annotations=annotations,
        )


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

    def __repr__(self) -> str:
        return f'Cell(loc={self.cell_loc}, dim={self.cell_dim})'

    def copy(self) -> 'Cell':
        return self.__class__()


@dataclasses.dataclass
class AnnotatedLayout(CellLayout):
    """
    Layout with annotations
    """
    layout: CellLayout
    annotations: Mapping[int, Optional[Mapping]] = dataclasses.field(
        default_factory=dict
    )
    assigned: Optional[Sequence[bool]] = None

    def __post_init__(self):
        if self.assigned is None:
            self.assigned = [False] * len(self.layout)
        assert len(self.layout) == len(self.assigned)

    def _leftmost(self) -> CellLayout:
        return self.layout._leftmost()

    def _next(self) -> Optional[CellLayout]:
        return self.layout._next()

    def _next_leaf(self) -> Optional[CellLayout]:
        return self.layout._next_leaf()

    def __getitem__(self, index: int) -> CellLayout:
        return self.layout[index]

    def __next__(self) -> CellLayout:
        return self.layout.__next__()

    def __iter__(self):
        return self.layout.__iter__()

    def __repr__(self) -> str:
        return (
            f'AnnotatedLayout(layout={self.layout}, '
            f'annotations={self.annotations})'
        )

    def __len__(self) -> int:
        return len(self.layout)

    def __mul__(self, other: 'AnnotatedLayout') -> 'AnnotatedLayout':
        layout = self.layout * other.layout
        len_l = len(self)
        len_r = len(other)
        annotations = {
            i * (len_r) + j: {
                **self.annotations[i],
                **other.annotations[j],
            }
            for i in range(len_l) for j in range(len_r)
        }
        return AnnotatedLayout(
            layout=layout,
            annotations=annotations,
        )

    def partition(
        self,
        width: int,
        height: int,
        x_offset: int = 0,
        y_offset: int = 0,
        padding: int = 0,
    ) -> 'AnnotatedLayout':
        layout = self.layout.partition(
            width=width,
            height=height,
            x_offset=x_offset,
            y_offset=y_offset,
            padding=padding,
        )
        return AnnotatedLayout(
            layout=layout,
            annotations=self.annotations,
            assigned=self.assigned,
        )

    def annotate_cell(
        self,
        index: int,
        annotation: Optional[Mapping] = None,
        collision_policy: Literal[
            'error', 'overwrite', 'coalesce', 'ignore'
        ] = 'coalesce',
    ) -> 'AnnotatedLayout':
        if annotation is None:
            annotation = {}
        assert index < len(self), f'Invalid cell index: {index}'
        current_annotation = self.annotations.get(index, None)
        if current_annotation is not None:
            if collision_policy == 'error':
                raise ValueError(
                    f'Cell {index} already has an annotation: '
                    f'{current_annotation}'
                )
            elif collision_policy == 'overwrite':
                pass
            elif collision_policy == 'coalesce':
                annotation = {**current_annotation, **annotation}
            elif collision_policy == 'ignore':
                return self
        annotations = {**self.annotations, **{index: annotation}}
        return AnnotatedLayout(
            layout=self.layout,
            annotations=annotations,
            assigned=self.assigned,
        )

    def match_and_assign(
        self,
        query: Mapping,
        match_to_unannotated: bool = False,
    ) -> Tuple['AnnotatedLayout', int]:
        """
        Match annotations against a query and assign the first available cell
        to the query
        """
        assigned = self.assigned.copy()
        matched = False
        for index, annotation in self.annotations.items():
            if assigned[index]:
                continue
            if annotation is None:
                if match_to_unannotated:
                    assigned[index] = True
                    matched = True
                    break
                else:
                    continue
            if all(
                (
                    query.get(key, None) == value
                    or query.get(key, None) in value
                )
                for key, value in annotation.items()
            ):
                assigned[index] = True
                matched = True
                break
        if not matched:
            index = len(self)
        return AnnotatedLayout(
            layout=self.layout,
            annotations=self.annotations,
            assigned=assigned,
        ), index

    def match_and_assign_all(
        self,
        queries: Sequence[Mapping],
        force_unmatched: bool = False,
    ) -> Tuple['AnnotatedLayout', Sequence[int]]:
        """
        Match annotations against a sequence of queries and assign the first
        available cell to each query
        """
        layout = self
        matched = [False] * len(queries)
        indices = [None] * len(queries)
        for i, query in enumerate(queries):
            layout, index = layout.match_and_assign(
                query, match_to_unannotated=False
            )
            if index < len(layout):
                indices[i] = index
                matched[i] = True
        for i, query in enumerate(queries):
            if matched[i]:
                continue
            layout, index = layout.match_and_assign(
                query, match_to_unannotated=True
            )
            if index < len(layout):
                indices[i] = index
                matched[i] = True
            elif force_unmatched:
                assigned = layout.assigned.copy()
                indices[i] = assigned.index(False)
                matched[i] = True
                assigned[indices[i]] = True
                layout = AnnotatedLayout(
                    layout=layout.layout,
                    annotations=layout.annotations,
                    assigned=assigned,
                )
        return layout, indices


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
    n_rows: int,
    n_cols: int,
    order: Literal['row', 'col'] = 'row',
    kernel: callable = Cell,
) -> CellLayout:
    """
    Create a grid layout
    """
    if order == 'row':
        layout = hsplit(
            1 / n_rows,
            *[
                vsplit(
                    1 / n_cols,
                    *[
                        kernel()
                        for _ in range(n_cols)
                    ]
                )
                for _ in range(n_rows)
            ]
        )
    elif order == 'col':
        layout = vsplit(
            1 / n_cols,
            *[
                hsplit(
                    1 / n_rows,
                    *[
                        kernel()
                        for _ in range(n_rows)
                    ]
                )
                for _ in range(n_cols)
            ]
        )
    else:
        raise ValueError(f'Invalid grid order: {order}')
    return layout
