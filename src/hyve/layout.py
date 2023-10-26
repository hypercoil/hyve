# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Layout representation
~~~~~~~~~~~~~~~~~~~~~
Classes for representing layouts of data
"""
import dataclasses
from typing import Any, Literal, Mapping, Optional, Sequence, Tuple, Union


@dataclasses.dataclass(frozen=True)
class CellLayoutArgument:
    # We use this as a buffer for operators with precedence slower than <<,
    # i.e. | (__or__)
    layout: 'CellLayout'
    argument: Any


@dataclasses.dataclass(frozen=True)
class CellLayoutSubstitution:
    layout: 'CellLayout'
    substitute: 'CellLayout'

    def __lshift__(self, other: int):
        return self.layout.substitute(other, self.substitute)


@dataclasses.dataclass(frozen=True)
class CellLayoutVerticalChain:
    chain: Sequence['CellLayout']

    def __or__(self, other: Union[float, 'CellLayoutArgument']):
        if isinstance(other, CellLayoutArgument):
            # We already have everything required to evaluate the chain
            chain = self.chain + [other.layout]
            return vsplit(other.argument, *chain)
        return CellLayoutVerticalChain(
            chain=self.chain + [other]
        )

    def __lshift__(self, other: float):
        return vsplit(other, *self.chain)


@dataclasses.dataclass(frozen=True)
class CellLayoutHorizontalChain:
    chain: Sequence['CellLayout']

    def __truediv__(self, other: 'CellLayout'):
        return CellLayoutHorizontalChain(
            chain=self.chain + [other]
        )

    def __lshift__(self, other: float):
        return hsplit(other, *self.chain)


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
        self.floating = None

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
        for i, cell in enumerate(self):
            if i == index:
                return cell

    def __next__(self) -> 'CellLayout':
        return self._next_leaf()

    def __iter__(self):
        cur_cell = self._leftmost()
        while cur_cell is not None:
            yield cur_cell
            cur_cell = cur_cell._next_leaf()
        if self.floating is not None:
            for f in self.floating:
                yield from f

    def __repr__(self) -> str:
        return self._repr()

    def __len__(self) -> int:
        if self.parent is not None:
            raise ValueError(
                'Cannot get length of a non-root layout, use '
                'len(layout.root) instead'
            )
        return sum(1 for _ in self)

    def __mod__(self, other: 'CellLayout') -> 'CellLayoutSubstitution':
        return CellLayoutSubstitution(
            layout=self,
            substitute=other,
        )

    def __truediv__(self, other: 'CellLayout') -> 'CellLayoutHorizontalChain':
        return CellLayoutHorizontalChain(
            chain=[self, other]
        )

    def __or__(
        self, other: Union['CellLayout', 'CellLayoutArgument']
    ) -> Union['CellLayoutVerticalChain', 'CellLayout']:
        if isinstance(other, CellLayoutArgument):
            return vsplit(other.argument, self, other.layout)
        return CellLayoutVerticalChain(
            chain=[self, other]
        )

    def __mul__(self, other: 'CellLayout') -> 'CellLayout':
        """Note that this 'product' is not commutative"""
        layout = self
        n = len(layout)
        for i in range(n, 0, -1):
            layout = layout.substitute(i - 1, other)
        return layout

    def __lshift__(self, other: Any):
        return CellLayoutArgument(
            layout=self,
            argument=other,
        )

    @property
    def root(self) -> 'CellLayout':
        """Get the root of the layout tree"""
        if self.parent is None:
            return self
        return self.parent.root

    def copy(self, **extra_params) -> 'CellLayout':
        left = self.left.copy() if self.left is not None else None
        right = self.right.copy() if self.right is not None else None
        floating = (
            [f.copy() for f in self.floating]
            if self.floating is not None
            else None
        )
        copy = self.__class__(
            left=left,
            right=right,
            split_orientation=self.split_orientation,
            split_position=self.split_position,
            **extra_params,
        )
        copy.cell_loc = self.cell_loc
        copy.cell_dim = self.cell_dim
        if left is not None:
            left.parent = copy
        if right is not None:
            right.parent = copy
        if floating is not None:
            copy.floating = floating
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
        if self.floating is not None:
            for f in self.floating:
                floating_width = round(f.float_dim_rel[0] * width)
                floating_height = round(f.float_dim_rel[1] * height)
                floating_x_offset = round(f.float_loc_rel[0] * width)
                floating_y_offset = round(f.float_loc_rel[1] * height)
                f.partition(
                    width=floating_width,
                    height=floating_height,
                    x_offset=floating_x_offset,
                    y_offset=floating_y_offset,
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

    def _repr(self, inject: Optional[str] = None) -> str:
        inject = '' if inject is None else inject
        if self.floating:
            floating = f', floating={self.floating}'
        else:
            floating = ''
        return (
            f'{self.__class__.__name__}'
            f'(left={self.left}, right={self.right}{floating}, '
            f'orientation={self.split_orientation}, '
            f'position={self.split_position}'
            f'{inject})'
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


class FloatingCellLayout(CellLayout):
    """
    A cell layout that floats on top of other cells
    """
    def __init__(
        self,
        float_loc_rel: Tuple[float, float],
        float_dim_rel: Tuple[float, float],
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
        self.float_loc_rel = float_loc_rel
        self.float_dim_rel = float_dim_rel
        self.floating = None

    def __repr__(self) -> str:
        return self._repr(
            f', float_loc_rel={self.float_loc_rel}, '
            f'float_dim_rel={self.float_dim_rel}'
        )

    def copy(self, **extra_params) -> 'FloatingCellLayout':
        return super().copy(
            float_loc_rel=self.float_loc_rel,
            float_dim_rel=self.float_dim_rel,
            **extra_params,
        )


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
        left = left.copy() if left is not None else None
        right = right.copy() if right is not None else None
        left_floating = left.floating or []
        right_floating = right.floating or []
        left.floating = None
        right.floating = None
        layout = CellLayout(
            left=left,
            right=right,
            split_orientation=orientation,
            split_position=position,
        )
        if left is not None:
            left.parent = layout
            left_floating = [
                refloat_layout(
                    e,
                    layout,
                    loc_rel=(0, 0),
                    dim_rel=(
                        (1, position)
                        if orientation == 'h'
                        else (position, 1)
                    ),
                    inner_loc_rel=e.float_loc_rel,
                    inner_dim_rel=e.float_dim_rel,
                )
                for e in left_floating
            ]
        if right is not None:
            right.parent = layout
            right_floating = [
                refloat_layout(
                    e,
                    layout,
                    loc_rel=(
                        (0, position)
                        if orientation == 'h'
                        else (position, 0)
                    ),
                    dim_rel=(
                        (1, 1 - position)
                        if orientation == 'h'
                        else (1 - position, 1)
                    ),
                    inner_loc_rel=e.float_loc_rel,
                    inner_dim_rel=e.float_dim_rel,
                )
                for e in right_floating
            ]
        floating = left_floating + right_floating or None
        layout.floating = floating
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


def refloat_layout(
    floating: CellLayout,
    anchor: CellLayout,
    loc_rel: Tuple[float, float],
    dim_rel: Tuple[float, float],
    inner_loc_rel: Tuple[float, float],
    inner_dim_rel: Tuple[float, float],
) -> CellLayout:
    new_dim_rel = tuple(
        i * j for i, j in zip(inner_dim_rel, dim_rel)
    )
    new_loc_rel = tuple(
        i + j * k
        for i, j, k in zip(loc_rel, inner_loc_rel, dim_rel)
    )
    return float_layout(
        floating,
        anchor,
        loc_rel=new_loc_rel,
        dim_rel=new_dim_rel,
    ).floating[-1]


def float_layout(
    floating: CellLayout,
    anchor: CellLayout,
    loc_rel: Tuple[float, float],
    dim_rel: Tuple[float, float],
) -> CellLayout:
    refloated = []
    if floating.floating is not None:
        inner = floating.floating
        for e in inner:
            refloated.append(
                refloat_layout(
                    e,
                    anchor,
                    loc_rel=loc_rel,
                    dim_rel=dim_rel,
                    inner_loc_rel=e.float_loc_rel,
                    inner_dim_rel=e.float_dim_rel,
                )
            )
    to_float = floating.copy()
    to_float.floating = None
    floating = FloatingCellLayout(
        float_loc_rel=loc_rel,
        float_dim_rel=dim_rel,
        left=to_float.left,
        right=to_float.right,
        split_orientation=to_float.split_orientation,
        split_position=to_float.split_position,
    )
    floating.left.parent = floating
    floating.right.parent = floating
    anchor = anchor.copy()
    if anchor.floating is None:
        anchor.floating = [floating] + refloated
    else:
        anchor.floating += [floating] + refloated
    return anchor
