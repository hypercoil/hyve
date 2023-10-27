# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Layout representation
~~~~~~~~~~~~~~~~~~~~~
Classes for representing layouts of data
"""
import dataclasses
from functools import cached_property, singledispatch
from typing import Any, Literal, Mapping, Optional, Sequence, Tuple, Union


BIG = 2**63 - 1


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
        return substitute(self.layout, self.substitute, other)


@dataclasses.dataclass(frozen=True)
class CellLayoutFloat:
    anchor: 'CellLayout'
    floating: 'CellLayout'

    def __lshift__(self, other: Tuple[Tuple[float, float], Tuple[float, float]]):
        return float_layout(
            self.floating,
            self.anchor,
            loc_rel=other[0],
            dim_rel=other[1],
        )


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

    def __add__(self, other: 'CellLayout') -> 'CellLayoutFloat':
        """Note that this 'sum' is not commutative"""
        return CellLayoutFloat(
            anchor=self,
            floating=other,
        )

    def __mul__(self, other: 'CellLayout') -> 'CellLayout':
        """Note that this 'product' is not commutative"""
        return product(self, other)

    def __matmul__(self, other: int) -> Tuple['CellLayout', 'CellLayout']:
        return break_at(self, other)

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

    @property
    def direct_size(self):
        """The direct size of a layout excludes any floating cells"""
        return sum(1 for e in self if e.root is self)

    @property
    def breakpoints_left(self):
        """Get valid breakpoints on the left side of the layout tree."""
        self.split_orientation
        if self.left.split_orientation == self.split_orientation:
            yield from self.left.breakpoints

    @property
    def breakpoints_right(self):
        """Get valid breakpoints on the right side of the layout tree."""
        if self.right.split_orientation == self.split_orientation:
            yield from self.right.breakpoints

    @property
    def breakpoints(self):
        yield from self.breakpoints_left
        yield self
        yield from self.breakpoints_right

    @property
    def is_left(self):
        return self.parent is not None and self.parent.left is self

    @property
    def is_right(self):
        return self.parent is not None and self.parent.right is self

    @property
    def count(self):
        left_count = self.left.count if self.left is not None else 0
        right_count = self.right.count if self.right is not None else 0
        return left_count + right_count

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

    @property
    def index(self):
        return list(self.root).index(self)

    @property
    def root_loc(self):
        # Neither the most efficient nor the most flexible way to do this,
        # but we'll take it for now
        loc = self.root.copy().partition(BIG, BIG)[self.index].cell_loc
        return (loc[0] / BIG, loc[1] / BIG)

    @property
    def root_dim(self):
        # Neither the most efficient nor the most flexible way to do this,
        # but we'll take it for now
        dim = self.root.copy().partition(BIG, BIG)[self.index].cell_dim
        return (dim[0] / BIG, dim[1] / BIG)

    @property
    def count(self):
        return 1


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

    def defloat(self) -> 'CellLayout':
        if len(self) == 1:
            return Cell()
        else:
            new = self.copy()
            return CellLayout(
                left=new.left,
                right=new.right,
                split_orientation=self.split_orientation,
                split_position=self.split_position,
            )


class FloatingCell(Cell):
    def __init__(
        self,
        float_loc_rel: Tuple[float, float],
        float_dim_rel: Tuple[float, float],
    ):
        super().__init__()
        self.float_loc_rel = float_loc_rel
        self.float_dim_rel = float_dim_rel

    def __repr__(self) -> str:
        return (
            'FloatingCell('
            f'loc={self.cell_loc}, dim={self.cell_dim}, '
            f'float_loc_rel={self.float_loc_rel}, '
            f'float_dim_rel={self.float_dim_rel}'
            ')'
        )

    def copy(self, **extra_params) -> 'FloatingCell':
        return self.__class__(
            float_loc_rel=self.float_loc_rel,
            float_dim_rel=self.float_dim_rel,
        )

    def defloat(self) -> 'Cell':
        return Cell()


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

    @property
    def root(self) -> 'CellLayout':
        raise NotImplementedError(
            'AnnotatedLayout does not support root property'
        )

    @property
    def direct_size(self):
        """The direct size of a layout excludes any floating cells"""
        return sum(1 for e in self.layout if e.root is self.layout)

    @property
    def floating(self):
        return self.layout.floating

    @cached_property
    def decomposition(self):
        if self.floating is None:
            return self, []
        else:
            max_annotation = self.direct_size
            root_layout = self.layout.copy()
            root_layout.floating = None
            root_layout = AnnotatedLayout(
                layout=root_layout,
                annotations={
                    i: annotation
                    for i, annotation in self.annotations.items()
                    if i < max_annotation
                },
            )
            floating_layouts = []
            for floating in self.floating:
                min_annotation = max_annotation
                max_annotation += len(floating)
                floating_layouts.append(
                    AnnotatedLayout(
                        layout=floating.defloat(),
                        annotations={
                            i - min_annotation: annotation
                            for i, annotation in self.annotations.items()
                            if i < max_annotation and i >= min_annotation
                        },
                    )
                )
            return root_layout, floating_layouts

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


def reindex_annotations(*cells: AnnotatedLayout) -> Mapping[int, Mapping]:
    aggregate_size = 0
    floating_size = 0
    root_annotations = {}
    floating_annotations = {}
    for cell in cells:
        direct_size = cell.direct_size
        root_annotations = {
            **root_annotations,
            **{
                aggregate_size + i: annotation
                for i, annotation in cell.annotations.items()
                if i < direct_size
            },
        }
        floating_annotations = {
            **floating_annotations,
            **{
                floating_size + i - direct_size: annotation
                for i, annotation in cell.annotations.items()
                if i >= direct_size
            },
        }
        aggregate_size += direct_size
        floating_size += len(cell) - direct_size
    floating_annotations = {
        aggregate_size + i: annotation
        for i, annotation in floating_annotations.items()
    }
    return {
        **root_annotations,
        **floating_annotations,
    }


@singledispatch
def _split(
    *cells: CellLayout,
    split_orientation: Literal['h', 'v'],
    split_position: float,
) -> CellLayout:
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
            split_orientation=split_orientation,
            split_position=split_position,
        )
        if left is not None:
            left.parent = layout
            left_floating = [
                refloat_layout(
                    e,
                    layout,
                    loc_rel=(0, 0),
                    dim_rel=(
                        (1, split_position)
                        if split_orientation == 'h'
                        else (split_position, 1)
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
                        (0, split_position)
                        if split_orientation == 'h'
                        else (split_position, 0)
                    ),
                    dim_rel=(
                        (1, 1 - split_position)
                        if split_orientation == 'h'
                        else (1 - split_position, 1)
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
        new_position = split_position / (1 - split_position)
        return split(
            split_orientation,
            split_position,
            cells[0],
            split(split_orientation, new_position, *cells[1:])
        )


@_split.register
def _(
    *cells: AnnotatedLayout,
    split_orientation: Literal['h', 'v'],
    split_position: float,
) -> AnnotatedLayout:
    layouts = [cell.layout for cell in cells]
    layout = _split(
        *layouts,
        split_orientation=split_orientation,
        split_position=split_position,
    )
    annotations = reindex_annotations(*cells)
    return AnnotatedLayout(
        layout=layout,
        annotations=annotations,
    )


def split(
    orientation: Literal['h', 'v'],
    position: float,
    *cells: CellLayout,
) -> CellLayout:
    """
    Create a split layout
    """
    return _split(
        *cells,
        split_orientation=orientation,
        split_position=position,
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


@singledispatch
def _float_layout(
    anchor: CellLayout,
    floating: CellLayout,
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
    if isinstance(to_float, Cell):
        floating = FloatingCell(
            float_loc_rel=loc_rel,
            float_dim_rel=dim_rel,
        )
    else:
        floating = FloatingCellLayout(
            float_loc_rel=loc_rel,
            float_dim_rel=dim_rel,
            left=to_float.left,
            right=to_float.right,
            split_orientation=to_float.split_orientation,
            split_position=to_float.split_position,
        )
    if floating.left is not None:
        floating.left.parent = floating
    if floating.right is not None:
        floating.right.parent = floating
    anchor = anchor.copy()
    if anchor.floating is None:
        anchor.floating = [floating] + refloated
    else:
        anchor.floating += [floating] + refloated
    return anchor


@_float_layout.register
def _(
    anchor: AnnotatedLayout,
    floating: AnnotatedLayout,
    loc_rel: Tuple[float, float],
    dim_rel: Tuple[float, float],
) -> AnnotatedLayout:
    layout = _float_layout(
        anchor.layout,
        floating.layout,
        loc_rel=loc_rel,
        dim_rel=dim_rel,
    )
    anchor_size = len(anchor)
    floating_annotations = {
        anchor_size + i: annotation
        for i, annotation in floating.annotations.items()
    }
    annotations = {
        **anchor.annotations,
        **floating_annotations,
    }
    return AnnotatedLayout(
        layout=layout,
        annotations=annotations,
    )


def float_layout(
    floating: CellLayout,
    anchor: CellLayout,
    loc_rel: Tuple[float, float],
    dim_rel: Tuple[float, float],
) -> CellLayout:
    return _float_layout(
        anchor,
        floating,
        loc_rel=loc_rel,
        dim_rel=dim_rel,
    )


@singledispatch
def _substitute(
    orig: CellLayout,
    substitute: CellLayout,
    index: int,
) -> CellLayout:
    if len(orig) <= 1:
        return substitute
    new = orig.copy()
    substitute = substitute.copy()

    cur_idx = 0
    cur_cell = new._leftmost()
    while cur_idx < index:
        cur_cell = cur_cell._next_leaf()
        cur_idx += 1
    cell_root_loc = cur_cell.root_loc
    cell_root_dim = cur_cell.root_dim
    if cur_cell.parent.left is cur_cell:
        cur_cell.parent.left = substitute
    else:
        cur_cell.parent.right = substitute
    substitute.parent = cur_cell.parent

    if substitute.floating is not None:
        substitute_floating = [
            refloat_layout(
                e,
                substitute,
                loc_rel=cell_root_loc,
                dim_rel=cell_root_dim,
                inner_loc_rel=e.float_loc_rel,
                inner_dim_rel=e.float_dim_rel,
            )
            for e in substitute.floating
        ]
        new.floating = (
            new.floating + substitute_floating
            if new.floating is not None
            else substitute_floating
        )
        substitute.floating = None
    return new


@_substitute.register
def _(
    orig: AnnotatedLayout,
    substitute: AnnotatedLayout,
    index: int,
) -> AnnotatedLayout:
    layout = _substitute(
        orig.layout,
        substitute.layout,
        index=index,
    )
    substitute_size = substitute.direct_size
    orig_size = orig.direct_size
    new_size = layout.direct_size
    incr = substitute_size - 1
    root_annotations = {
        i: annotation
        for i, annotation in orig.annotations.items()
        if (i < orig_size and i != index)
    }
    root_annotations = {
        **{
            i if i < index else i + incr: annotation
            for i, annotation in root_annotations.items()
        },
        **{
            i + index: annotation
            for i, annotation in substitute.annotations.items()
            if i < substitute_size
        },
    }
    if layout.floating is not None:
        orig_floating_size = len(orig) - orig_size
        floating_annotations = {
            **{
                i - orig_size + new_size: annotation
                for i, annotation in orig.annotations.items()
                if i >= orig_size
            },
            **{
                (
                    i - substitute_size + new_size + orig_floating_size
                ): annotation
                for i, annotation in substitute.annotations.items()
                if i >= substitute_size
            },
        }
    else:
        floating_annotations = {}
    annotations = {
        **root_annotations,
        **floating_annotations,
    }
    return AnnotatedLayout(
        layout=layout,
        annotations=annotations,
    )


def substitute(
    orig: CellLayout,
    substitute: CellLayout,
    index: int,
) -> CellLayout:
    """Substitute a cell in the layout with another layout"""
    return _substitute(
        orig,
        substitute,
        index=index,
    )


@singledispatch
def _product(
    outer: CellLayout,
    inner: CellLayout,
) -> CellLayout:
    if outer.floating is not None:
        # Try to lift the floating cells and recompose
        outer_base = outer.copy()
        outer_base.floating = None
        layout = product(outer_base, inner)
        float_loc_rel = [e.float_loc_rel for e in outer.floating]
        float_dim_rel = [e.float_dim_rel for e in outer.floating]
        floating_product = [
            product(e, inner)
            for e in outer.floating
        ]
        for floating, loc, dim in zip(
            floating_product,
            float_loc_rel,
            float_dim_rel,
        ):
            layout = float_layout(
                floating,
                layout,
                loc_rel=loc,
                dim_rel=dim,
            )
        return layout
    assert outer.floating is None, (
        'Floating cells are not supported in the outer layout'
    )
    layout = outer
    inner_size = inner.direct_size
    for i in range(len(layout)):
        layout = substitute(layout, inner, i * inner_size)
    return layout


@_product.register
def _(
    outer: AnnotatedLayout,
    inner: AnnotatedLayout
) -> AnnotatedLayout:
    if outer.floating is not None:
        # Try to lift the floating cells and recompose
        outer_base, outer_floating = outer.decomposition
        layout = product(outer_base, inner)
        float_loc_rel = [e.float_loc_rel for e in outer.floating]
        float_dim_rel = [e.float_dim_rel for e in outer.floating]
        floating_product = [
            product(e, inner)
            for e in outer_floating
        ]
        for floating, loc, dim in zip(
            floating_product,
            float_loc_rel,
            float_dim_rel,
        ):
            layout = float_layout(
                floating,
                layout,
                loc_rel=loc,
                dim_rel=dim,
            )
        return layout
    assert outer.floating is None, (
        'Floating cells are not supported in the outer layout'
    )
    layout = outer.layout * inner.layout
    outer_size = outer.direct_size
    inner_size = inner.direct_size
    fixed_size = outer_size * inner_size
    inner_total_size = len(inner)
    root_annotations = {
        i * inner_size + j: {
            **outer.annotations.get(i, {}),
            **inner.annotations.get(j, {})
        }
        for i in range(outer_size) for j in range(inner_size)
    }
    floating_annotations = {
        i * inner_total_size - (i + 1) * inner_size + j + fixed_size: {
            **outer.annotations.get(i, {}),
            **inner.annotations.get(j, {})
        }
        for i in range(outer_size)
        for j in range(inner_size, inner_total_size)
    }

    return AnnotatedLayout(
        layout=layout,
        annotations={
            **root_annotations,
            **floating_annotations,
        },
    )


def product(
    inner: CellLayout,
    outer: CellLayout,
) -> CellLayout:
    """Product of two layouts"""
    return _product(
        inner,
        outer,
    )


@singledispatch
def _break_at(
    layout: CellLayout,
    index: int,
) -> Tuple[CellLayout, CellLayout]:
    to_break = layout.copy()
    broken = None
    breakpoints_left = list(to_break.breakpoints_left)
    breakpoints_right = list(to_break.breakpoints_right)
    breakpoints = (
        breakpoints_left + [to_break] + breakpoints_right
    )
    break_point = breakpoints[index]
    breakpoints = dict(zip(breakpoints, range(len(breakpoints))))
    if break_point is to_break:
        left = to_break.left
        right = to_break.right
        left.parent = None
        right.parent = None
        return left, right

    # By the way we've defined valid break points, we can only traverse
    # valid break points when moving from the root to any valid break
    # point. This means that we can use the index of the break point
    # to determine whether we need to traverse left or right to get to
    # it.
    pointer = to_break
    leftward = index < breakpoints[pointer]
    parent = None
    break_leftward = None
    while pointer != break_point:
        previous = pointer
        pointer = pointer.left if leftward else pointer.right
        if ((index < breakpoints[pointer]) != leftward):
            if leftward:
                previous.left = None
            else:
                previous.right = None
            pointer.parent = None
            if broken is None:
                break_leftward = leftward
                broken = pointer
            elif parent is not None:
                break_leftward = leftward
                if leftward:
                    parent.right = pointer
                else:
                    parent.left = pointer
                pointer.parent = parent
            leftward = not leftward
            parent = previous
    if broken is None:
        parent = previous
        break_leftward = leftward
        broken = pointer

    attach_left = breakpoints[parent] < index
    branch_upper = (
        break_point.left if attach_left else break_point.right
    )
    branch_lower = (
        break_point.right if attach_left else break_point.left
    )
    # break_point.left = None
    # break_point.right = None
    branch_upper.parent = None
    branch_lower.parent = None
    match break_leftward:
        case True:
            parent.left = branch_upper
            branch_upper.parent = parent
            if break_point.parent is None or break_point.parent is parent:
                broken = branch_lower
            elif break_point.is_left:
                break_point.parent.left = branch_lower
                branch_lower.parent = break_point.parent
            elif break_point.is_right:
                break_point.parent.right = branch_lower
                branch_lower.parent = break_point.parent
        case False:
            parent.right = branch_upper
            branch_upper.parent = parent
            if break_point.parent is None or break_point.parent is parent:
                broken = branch_lower
            elif break_point.is_left:
                break_point.parent.left = branch_lower
                branch_lower.parent = break_point.parent
            elif break_point.is_right:
                break_point.parent.right = branch_lower
                branch_lower.parent = break_point.parent
    #break_point.parent = None
    if index < breakpoints[to_break]:
        return broken, to_break
    else:
        return to_break, broken


def break_at(
    layout: CellLayout,
    index: int,
) -> Tuple[CellLayout, CellLayout]:
    """Break a layout at a given index"""
    return _break_at(
        layout,
        index=index,
    )
