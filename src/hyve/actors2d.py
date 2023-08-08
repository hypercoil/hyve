# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
2-dimensional plot actors
~~~~~~~~~~~~~~~~~~~~~~~~~
Operations for creating and tiling 2-dimensional plots.
"""
import dataclasses
import io
from abc import abstractmethod
from collections.abc import Mapping as MappingABC
from typing import Any, Literal, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors, patheffects
from matplotlib.figure import Figure
from PIL import Image

from .const import (
    SCALAR_BAR_DEFAULT_BELOW_COLOR,
    SCALAR_BAR_DEFAULT_FONT,
    SCALAR_BAR_DEFAULT_FONT_COLOR,
    SCALAR_BAR_DEFAULT_FONT_OUTLINE_COLOR,
    SCALAR_BAR_DEFAULT_FONT_OUTLINE_MULTIPLIER,
    SCALAR_BAR_DEFAULT_LENGTH,
    SCALAR_BAR_DEFAULT_LIM_FONTSIZE_MULTIPLIER,
    SCALAR_BAR_DEFAULT_NAME,
    SCALAR_BAR_DEFAULT_NAME_FONTSIZE_MULTIPLIER,
    SCALAR_BAR_DEFAULT_NUM_SIG_FIGS,
    SCALAR_BAR_DEFAULT_ORIENTATION,
    SCALAR_BAR_DEFAULT_SPACING,
    SCALAR_BAR_DEFAULT_WIDTH,
    TEXT_DEFAULT_ANGLE,
    TEXT_DEFAULT_BOUNDING_BOX_HEIGHT,
    TEXT_DEFAULT_BOUNDING_BOX_WIDTH,
    TEXT_DEFAULT_CONTENT,
    TEXT_DEFAULT_FONT,
    TEXT_DEFAULT_FONT_COLOR,
    TEXT_DEFAULT_FONT_OUTLINE_COLOR,
    TEXT_DEFAULT_FONT_OUTLINE_MULTIPLIER,
    TEXT_DEFAULT_FONT_SIZE_MULTIPLIER,
    TYPICAL_DPI,
    Tensor,
)


@dataclasses.dataclass(frozen=True)
class Actors2DBuilder(MappingABC):
    """Addressable container for 2D actors."""

    def __getitem__(self, key: str):
        return self.__getattribute__(key)

    def __iter__(self):
        return iter(dataclasses.asdict(self))

    def __len__(self) -> int:
        return len(dataclasses.asdict(self))

    @abstractmethod
    def __eq__(self, other) -> bool:
        pass

    @abstractmethod
    def __hash__(self) -> int:
        pass

    @abstractmethod
    def __call__(self) -> Any:
        pass

    @property
    @abstractmethod
    def canvas_height(self) -> int:
        pass

    @property
    @abstractmethod
    def canvas_width(self) -> int:
        pass

    @abstractmethod
    def set_canvas_size(self, height: int, width: int) -> 'Actors2DBuilder':
        pass

    def eval_spec(self, metadata: Mapping[str, str]) -> 'Actors2DBuilder':
        return self


@dataclasses.dataclass(frozen=True)
class ScalarBarBuilder(Actors2DBuilder):
    """Addressable container for scalar bar parameters."""
    mapper: Optional[cm.ScalarMappable]
    name: Optional[str] = SCALAR_BAR_DEFAULT_NAME
    below_color: Optional[str] = SCALAR_BAR_DEFAULT_BELOW_COLOR
    length: int = SCALAR_BAR_DEFAULT_LENGTH
    width: int = SCALAR_BAR_DEFAULT_WIDTH
    orientation: Literal['h', 'v'] = SCALAR_BAR_DEFAULT_ORIENTATION
    num_sig_figs: int = SCALAR_BAR_DEFAULT_NUM_SIG_FIGS
    font: str = SCALAR_BAR_DEFAULT_FONT
    name_fontsize_multiplier: float = (
        SCALAR_BAR_DEFAULT_NAME_FONTSIZE_MULTIPLIER
    )
    lim_fontsize_multiplier: float = (
        SCALAR_BAR_DEFAULT_LIM_FONTSIZE_MULTIPLIER
    )
    font_color: Any = SCALAR_BAR_DEFAULT_FONT_COLOR
    font_outline_color: Any = (
        SCALAR_BAR_DEFAULT_FONT_OUTLINE_COLOR
    )
    font_outline_multiplier: float = (
        SCALAR_BAR_DEFAULT_FONT_OUTLINE_MULTIPLIER
    )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ScalarBarBuilder):
            return False
        mapper = self.mapper
        if mapper is None or mapper.norm is None:
            return self.name == other.name
        other_mapper = other.mapper
        if other_mapper is None or other_mapper.norm is None:
            return self.name == other.name
        return (
            self.name == other.name
            and np.isclose(mapper.norm.vmin, other.mapper.norm.vmin)
            and np.isclose(mapper.norm.vmax, other_mapper.norm.vmax)
        )

    def __hash__(self) -> int:
        mapper = self.mapper
        if mapper is None or mapper.norm is None:
            return hash((self.name, 'ScalarBarBuilder'))
        return hash((
            self.name,
            mapper.norm.vmin,
            mapper.norm.vmax,
            'ScalarBarBuilder'
        ))

    def __call__(self) -> Any:
        return build_scalar_bar(**dataclasses.asdict(self))

    @property
    def canvas_height(self) -> int:
        if self.orientation == 'h':
            return self.width
        else:
            return self.length

    @property
    def canvas_width(self) -> int:
        if self.orientation == 'h':
            return self.length
        else:
            return self.width

    def set_canvas_size(self, height: int, width: int) -> 'ScalarBarBuilder':
        if self.orientation == 'h':
            return dataclasses.replace(self, width=height, length=width)
        else:
            return dataclasses.replace(self, width=width, length=height)

    def eval_spec(self, metadata: Mapping[str, str]) -> 'ScalarBarBuilder':
        if self.name is None:
            return self
        return dataclasses.replace(self, name=self.name.format(**metadata))


@dataclasses.dataclass(frozen=True)
class TextBuilder(Actors2DBuilder):
    """Addressable container for text parameters."""
    content: str = TEXT_DEFAULT_CONTENT
    font: str = TEXT_DEFAULT_FONT
    font_size_multiplier: int = TEXT_DEFAULT_FONT_SIZE_MULTIPLIER
    font_color: Any = TEXT_DEFAULT_FONT_COLOR
    font_outline_color: Any = TEXT_DEFAULT_FONT_OUTLINE_COLOR
    font_outline_multiplier: float = TEXT_DEFAULT_FONT_OUTLINE_MULTIPLIER
    bounding_box_width: Optional[int] = TEXT_DEFAULT_BOUNDING_BOX_WIDTH
    bounding_box_height: Optional[int] = TEXT_DEFAULT_BOUNDING_BOX_HEIGHT
    angle: int = TEXT_DEFAULT_ANGLE

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, TextBuilder):
            return False
        return all(
            self[key] == other[key]
            for key in dataclasses.asdict(self)
        )

    def __hash__(self) -> int:
        return hash(
            tuple(dataclasses.asdict(self).values())
        )

    def __call__(self):
        return build_text_box(**dataclasses.asdict(self))

    @property
    def canvas_height(self) -> int:
        return self.bounding_box_height

    @property
    def canvas_width(self) -> int:
        return self.bounding_box_width

    def set_canvas_size(self, height, width) -> 'TextBuilder':
        return dataclasses.replace(
            self,
            bounding_box_height=height,
            bounding_box_width=width,
        )

    def eval_spec(self, metadata: Mapping[str, str]) -> 'TextBuilder':
        return dataclasses.replace(
            self,
            content=self.content.format(**metadata)
        )


def build_text_box(
    *,
    content: str,
    font: str,
    font_size_multiplier: int,
    font_color: Any,
    font_outline_color: Any,
    font_outline_multiplier: float,
    bounding_box_width: int,
    bounding_box_height: int,
    angle: int,
):
    figsize = (
        bounding_box_width / TYPICAL_DPI,
        bounding_box_height / TYPICAL_DPI,
    )
    f, ax = plt.subplots(figsize=figsize)
    font_size = 1
    text = ax.text(
        0.5,
        0.5,
        content,
        ha='center',
        va='center',
        fontname=font,
        fontsize=font_size,
        color=font_color,
        rotation=angle,
    )
    # Rescale the text to fit the bounding box
    text_size = text.get_window_extent(f.canvas.get_renderer())
    text_width = text_size.width
    text_height = text_size.height
    scale = min(
        bounding_box_width / text_width,
        bounding_box_height / text_height,
    )
    font_size *= (scale * font_size_multiplier)
    text.set_fontsize(font_size)
    if font_outline_color is not None:
        text.set_path_effects(
            [
                patheffects.withStroke(
                    linewidth=font_size * font_outline_multiplier,
                    foreground=font_outline_color,
                ),
                patheffects.Normal(),
            ]
        )
    ax.axis('off')
    f.subplots_adjust(0, 0, 1, 1)
    return f


def build_scalar_bar(
    *,
    mapper: cm.ScalarMappable,
    name: Optional[str] = SCALAR_BAR_DEFAULT_NAME,
    below_color: Optional[str] = SCALAR_BAR_DEFAULT_BELOW_COLOR,
    length: int = SCALAR_BAR_DEFAULT_LENGTH,
    width: int = SCALAR_BAR_DEFAULT_WIDTH,
    orientation: Literal['h', 'v'] = SCALAR_BAR_DEFAULT_ORIENTATION,
    num_sig_figs: int = SCALAR_BAR_DEFAULT_NUM_SIG_FIGS,
    font: str = SCALAR_BAR_DEFAULT_FONT,
    name_fontsize_multiplier: float = (
        SCALAR_BAR_DEFAULT_NAME_FONTSIZE_MULTIPLIER
    ),
    lim_fontsize_multiplier: float = (
        SCALAR_BAR_DEFAULT_LIM_FONTSIZE_MULTIPLIER
    ),
    font_color: Any = SCALAR_BAR_DEFAULT_FONT_COLOR,
    font_outline_color: Any = (
        SCALAR_BAR_DEFAULT_FONT_OUTLINE_COLOR
    ),
    font_outline_multiplier: float = (
        SCALAR_BAR_DEFAULT_FONT_OUTLINE_MULTIPLIER
    ),
) -> Figure:
    if name is not None:
        name = name.upper() # TODO: change this! work into style
    vmin, vmax = mapper.get_clim()

    # TODO: Drop this ridiculous hack after we switch to programmatically
    #       creating SVG files instead of using matplotlib
    aspect = length / width
    width = 128
    length = int(width * aspect)

    static_length = num_sig_figs * width // 2
    dynamic_length = length - 2 * static_length
    dynamic = mapper.to_rgba(np.linspace(vmin, vmax, dynamic_length))
    above = np.tile(mapper.to_rgba(vmax), (static_length, 1))
    if below_color is not None:
        if len(below_color) == 4 and below_color[-1] == 0:
            # Not ideal, but looks better than transparent and too many
            # color bars actually end in black
            below_color = '#444444'
        below = np.tile(colors.to_rgba(below_color), (static_length, 1))
    else:
        below = np.tile(mapper.to_rgba(vmin), (static_length, 1))
    rgba = np.stack(width * [np.concatenate([below, dynamic, above])])

    match orientation:
        case 'h':
            figsize = (length / TYPICAL_DPI, width / TYPICAL_DPI)
            vmin_params = {
                'xy': (0, 0),
                'xytext': (0.02, 0.5),
                'rotation': 0,
                'ha': 'left',
                'va': 'center',
            }
            vmax_params = {
                'xy': (1, 0),
                'xytext': (0.98, 0.5),
                'rotation': 0,
                'ha': 'right',
                'va': 'center',
            }
            name_params = {
                'xy': (0.5, 0),
                'xytext': (0.5, 0.5),
                'rotation': 0,
                'ha': 'center',
                'va': 'center',
            }
        case 'v':
            figsize = (width / TYPICAL_DPI, length / TYPICAL_DPI)
            rgba = rgba.swapaxes(0, 1)[::-1]
            vmin_params = {
                'xy': (0, 0),
                'xytext': (0.5, 0.02),
                'rotation': 90,
                'ha': 'center',
                'va': 'bottom',
            }
            vmax_params = {
                'xy': (0, 1),
                'xytext': (0.5, 0.98),
                'rotation': 90,
                'ha': 'center',
                'va': 'top',
            }
            name_params = {
                'xy': (0, .5),
                'xytext': (0.5, 0.5),
                'rotation': 90,
                'ha': 'center',
                'va': 'center',
            }
        case _:
            raise ValueError(f'Invalid orientation: {orientation}')

    f, ax = plt.subplots(figsize=figsize)
    ax.imshow(rgba)
    # TODO: Drop this hack after we switch to programmatically creating
    #       SVG files instead of using matplotlib
    # mul = TYPICAL_DPI / f.dpi
    for vlim, vlim_params in zip(
        (vmin, vmax),
        (vmin_params, vmax_params),
    ):
        fontsize = width * lim_fontsize_multiplier # * mul
        ax.annotate(
            f'{vlim:.{num_sig_figs}g}',
            xycoords='axes fraction',
            fontsize=fontsize,
            fontfamily=font,
            color=font_color,
            path_effects=[
                patheffects.withStroke(
                    linewidth=font_outline_multiplier * fontsize,
                    foreground=font_outline_color,
                )
            ],
            **vlim_params,
        )
    if name is not None:
        fontsize = width * name_fontsize_multiplier # * mul
        ax.annotate(
            name,
            xycoords='axes fraction',
            fontsize=fontsize,
            fontfamily=font,
            color=font_color,
            path_effects=[
                patheffects.withStroke(
                    linewidth=font_outline_multiplier * fontsize,
                    foreground=font_outline_color,
                )
            ],
            **name_params,
        )
    ax.axis('off')
    f.subplots_adjust(0, 0, 1, 1)
    return f


def _uniquify_names(builders: Sequence[ScalarBarBuilder]):
    unique_builders = set()
    retained_builders = []
    for builder in builders:
        if builder not in unique_builders:
            unique_builders.add(builder)
            retained_builders.append(builder)
    return retained_builders


def tile_actors2d(
    builders: Sequence[ScalarBarBuilder],
    spacing: float = SCALAR_BAR_DEFAULT_SPACING,
    max_dimension: Optional[Tuple[int, int]] = None,
    require_unique_names: bool = True,
    background_color: Any = (0, 0, 0, 255),
) -> Optional[Tensor]:
    builders = [b for b in builders if b is not None]
    if len(builders) == 0:
        return None
    # Algorithm from https://stackoverflow.com/a/28268965
    if require_unique_names:
        builders = _uniquify_names(builders)
    count = len(builders)
    max_width, max_height = max_dimension
    if spacing < 1:
        spacing = spacing * min(max_width, max_height)
    spacing = int(spacing)
    width = max([
        builder.canvas_width
        for builder in builders
    ])
    height = max([
        builder.canvas_height
        for builder in builders
    ])
    candidates = np.concatenate((
        (max_width - np.arange(count) * spacing) /
        (np.arange(1, count + 1) * width),
        (max_height - np.arange(count) * spacing) /
        (np.arange(1, count + 1) * height),
    ))
    candidates = np.sort(candidates[candidates > 0])
    # Vectorising is probably faster than a binary search, so that's what
    # we're doing here
    n_col = np.floor((max_width + spacing) / (width * candidates + spacing))
    n_row = np.floor((max_height + spacing) / (height * candidates + spacing))
    feasible = (n_row * n_col) >= count
    layout_idx = np.max(np.where(feasible))
    layout = np.array([n_row[layout_idx], n_col[layout_idx]], dtype=int)
    scalar = candidates[layout_idx]
    width = int(scalar * width)
    height = int(scalar * height)
    # End algorithm from https://stackoverflow.com/a/28268965

    images = []
    buffers = []
    figs = []
    for builder in builders:
        builder = builder.set_canvas_size(height=height, width=width)
        fig = builder()
        # TODO: The below should work, but in typical matplotlib fashion, it
        # doesn't. One more compelling reason to switch to programmatically
        # creating SVG files instead of using matplotlib, because now we
        # instead have to do an expensive alpha compositing operation to get
        # the background colour right.
        # ax = fig.axes[0]
        # ax.set_facecolor(background_color)
        # From https://stackoverflow.com/a/8598881
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', transparent=True)
        buffer.seek(0)
        images += [Image.open(buffer)]
        buffers += [buffer]
        figs += [fig]

    #tight = {0: 'row', 1: 'col'}[np.argmin(layout)]
    argtight = np.argmin(layout)
    argslack = 1 - argtight
    counttight = layout[argtight]
    countslack = np.ceil(count / counttight).astype(int)
    layout = [None, None]
    layout[argtight] = counttight
    layout[argslack] = countslack

    # Things are about to get confusing because we're switching back and forth
    # between array convention and Cartesian convention. Sorry.
    canvas = Image.new('RGBA', (max_width, max_height), background_color)
    active_canvas_size = (
        layout[1] * width + (layout[1] - 1) * spacing,
        layout[0] * height + (layout[0] - 1) * spacing,
    )
    offset = [0, 0]
    offset[1 - argslack] = (
        canvas.size[1 - argslack] - active_canvas_size[1 - argslack]
    ) // 2
    for i, image in enumerate(images):
        # Unfortunately matplotlib invariably adds some padding, so an aspect
        # preserving resize is still necessary
        image = image.resize((width, height), Image.Resampling.LANCZOS)
        canvas.paste(image, tuple(int(o) for o in offset))
        # Filling this in row-major order
        if i % layout[1] == layout[1] - 1:
            offset[0] = 0
            offset[1] += height + spacing
        else:
            offset[0] += width + spacing
    #canvas.show()

    for buffer in buffers:
        buffer.close()
    return np.array(canvas)
