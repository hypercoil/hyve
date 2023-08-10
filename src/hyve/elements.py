# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
2-dimensional plot elements
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Operations for creating and tiling 2-dimensional plots and plot elements.
"""
import base64
import dataclasses
import io
from abc import abstractmethod
from collections.abc import Mapping as MappingABC
from statistics import median
from typing import Any, Literal, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import svg
from matplotlib import cm, colors, patheffects
from matplotlib.figure import Figure
from PIL import Image

from .const import (
    RASTER_DEFAULT_BOUNDING_BOX_HEIGHT,
    RASTER_DEFAULT_BOUNDING_BOX_WIDTH,
    RASTER_DEFAULT_FORMAT,
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
class ElementBuilder(MappingABC):
    """Addressable container for 2D plot elements."""

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
    def __call__(self, backend: Optional[str] = None) -> Any:
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
    def set_canvas_size(self, height: int, width: int) -> 'ElementBuilder':
        pass

    def eval_spec(self, metadata: Mapping[str, str]) -> 'ElementBuilder':
        return self


@dataclasses.dataclass(frozen=True)
class ScalarBarBuilder(ElementBuilder):
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
    priority: int = 0

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

    def __call__(self, backend: Optional[str] = None) -> Any:
        if backend is None:
            backend = get_default_backend()
        return build_scalar_bar(
            **dataclasses.asdict(self),
            backend=backend,
        )

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
class TextBuilder(ElementBuilder):
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
    priority: int = 0

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

    def __call__(self, backend: Optional[str] = None):
        if backend is None:
            backend = get_default_backend()
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


@dataclasses.dataclass(frozen=True)
class RasterBuilder(ElementBuilder):
    """Addressable container for raster parameters."""
    content: Tensor
    bounding_box_height: int = RASTER_DEFAULT_BOUNDING_BOX_HEIGHT
    bounding_box_width: int = RASTER_DEFAULT_BOUNDING_BOX_WIDTH
    fmt: str = RASTER_DEFAULT_FORMAT
    priority: int = 0

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, RasterBuilder):
            return False
        return np.all(self.content == other.content) and (
            all(
                self[key] == other[key]
                for key in dataclasses.asdict(self)
                if key != 'content'
            )
        )

    def __hash__(self) -> int:
        return hash(
            (self.content.data.tobytes(),) + tuple(
                v for k, v in dataclasses.asdict(self).items()
                if k != 'content'
            )
        )

    def __call__(self, backend: Optional[str] = None):
        if backend is None:
            backend = get_default_backend()
        return build_raster(**dataclasses.asdict(self))

    @property
    def canvas_height(self) -> int:
        return self.height

    @property
    def canvas_width(self) -> int:
        return self.width

    def set_canvas_size(self, height, width) -> 'RasterBuilder':
        return dataclasses.replace(
            self,
            height=height,
            width=width,
        )

    def eval_spec(self, metadata: Mapping[str, str]) -> 'RasterBuilder':
        return self


def get_default_backend() -> str:
    return 'svg'


def build_text_box(
    *,
    content: str = TEXT_DEFAULT_CONTENT,
    font: str = TEXT_DEFAULT_FONT,
    font_size_multiplier: int = TEXT_DEFAULT_FONT_SIZE_MULTIPLIER,
    font_color: Any = TEXT_DEFAULT_FONT_COLOR,
    font_outline_color: Any = TEXT_DEFAULT_FONT_OUTLINE_COLOR,
    font_outline_multiplier: float = TEXT_DEFAULT_FONT_OUTLINE_MULTIPLIER,
    bounding_box_width: int = TEXT_DEFAULT_BOUNDING_BOX_WIDTH,
    bounding_box_height: int = TEXT_DEFAULT_BOUNDING_BOX_HEIGHT,
    angle: int = TEXT_DEFAULT_ANGLE,
    backend: Optional[str] = None,
    **params: Any,
):
    if backend is None:
        backend = get_default_backend()
    params = {
        'content': content,
        'font': font,
        'font_size_multiplier': font_size_multiplier,
        'font_color': font_color,
        'font_outline_color': font_outline_color,
        'font_outline_multiplier': font_outline_multiplier,
        'bounding_box_width': bounding_box_width,
        'bounding_box_height': bounding_box_height,
        'angle': angle,
    }
    if backend == 'matplotlib':
        return build_text_box_matplotlib(**params)
    elif backend == 'svg':
        return build_text_box_svg(**params)
    raise ValueError(f'Unknown backend {backend}')


def build_text_box_matplotlib(
    *,
    content: str = TEXT_DEFAULT_CONTENT,
    font: str = TEXT_DEFAULT_FONT,
    font_size_multiplier: int = TEXT_DEFAULT_FONT_SIZE_MULTIPLIER,
    font_color: Any = TEXT_DEFAULT_FONT_COLOR,
    font_outline_color: Any = TEXT_DEFAULT_FONT_OUTLINE_COLOR,
    font_outline_multiplier: float = TEXT_DEFAULT_FONT_OUTLINE_MULTIPLIER,
    bounding_box_width: int = TEXT_DEFAULT_BOUNDING_BOX_WIDTH,
    bounding_box_height: int = TEXT_DEFAULT_BOUNDING_BOX_HEIGHT,
    angle: int = TEXT_DEFAULT_ANGLE,
):
    """
    Build a text box using matplotlib.

    It is strongly advised to use the SVG backend instead.
    """
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


def build_text_box_svg(
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
    font_size = bounding_box_height * font_size_multiplier
    transform = []
    if angle != 0:
        transform = [
            svg.Rotate(angle, bounding_box_width / 2, bounding_box_height / 2)
        ]
    text_params = dict(
        x=bounding_box_width / 2,
        y=bounding_box_height / 2,
        text=content,
        font_family=font,
        font_size=font_size,
        fill=font_color,
        text_anchor='middle',
        dominant_baseline='middle',
    )
    text_stroke = None
    if font_outline_color is not None:
        text_stroke = svg.Text(
            **text_params,
            stroke=font_outline_color,
            stroke_width=font_size * font_outline_multiplier,
            stroke_linejoin='round',
        )
    text = svg.Text(**text_params)
    elements = [text_stroke, text]
    textgroup = svg.G(
        elements=[e for e in elements if e is not None],
        transform=transform,
    )
    graphics = svg.SVG(
        width=bounding_box_width,
        height=bounding_box_height,
        elements=[textgroup],
    )
    # Rescale the text to fit the bounding box
    # TODO: Well, we tried. This doesn't work. In particular, svgelements is
    #       useless for this purpose. So instead, our prescription for adding
    #       text is to just use a font size that's big enough to fill the
    #       specified bounding box. The figure builder will take care of then
    #       scaling the bounding box and text to the desired size.
    # buffer = io.StringIO()
    # buffer.write(img.__str__())
    # buffer.seek(0)
    # start_w = font_size_multiplier * bounding_box_width / 2
    # start_h = font_size_multiplier * bounding_box_height / 2
    # max_bbox = (
    #     start_w,
    #     start_h,
    #     bounding_box_width - start_w,
    #     bounding_box_height - start_h,
    # )
    # bbox = (float('inf'), float('inf'), float('-inf'), float('-inf'))
    # for e in svgelements.SVG.parse(buffer).elements():
    #     if isinstance(e, svgelements.SVGText):
    #         try:
    #             e_bbox = e.bbox()
    #             bbox = (
    #                 min(bbox[0], e_bbox[0]),
    #                 min(bbox[1], e_bbox[1]),
    #                 max(bbox[2], e_bbox[2]),
    #                 max(bbox[3], e_bbox[3]),
    #             )
    #         except AttributeError:
    #             continue
    # scale = min(
    #     (max_bbox[2] - max_bbox[0]) / (bbox[2] - bbox[0]),
    #     (max_bbox[3] - max_bbox[1]) / (bbox[3] - bbox[1]),
    # )
    # transform += [
    #     svg.Scale(scale, scale),
    # ]
    # textgroup = svg.G(
    #     elements=[e for e in elements if e is not None],
    #     transform=transform,
    # )
    # graphics = svg.SVG(
    #     width=bounding_box_width,
    #     height=bounding_box_height,
    #     elements=[textgroup],
    #     viewBox=f'0 0 {bounding_box_width} {bounding_box_height}',
    # )
    return graphics


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
    backend: Optional[str] = None,
    **params: Any,
) -> Figure:
    """
    Build a scalar bar.

    Parameters
    ----------
    mapper : cm.ScalarMappable
        The mapper to use for the scalar bar.
    name : str, optional
        The name of the scalar bar.
    below_color : str, optional
        The color to use for values below the range.
    length : int, optional
        The length of the scalar bar.
    width : int, optional
        The width of the scalar bar.
    orientation : {'h', 'v'}, optional
        The orientation of the scalar bar.
    num_sig_figs : int, optional
        The number of significant figures to use for the scalar bar.
    font : str, optional
        The font to use for the scalar bar.
    name_fontsize_multiplier : float, optional
        The font size multiplier to use for the name.
    lim_fontsize_multiplier : float, optional
        The font size multiplier to use for the limits.
    font_color : Any, optional
        The font color to use.
    font_outline_color : Any, optional
        The font outline color to use.
    font_outline_multiplier : float, optional
        The font outline multiplier to use.
    backend : str, optional
        The backend to use for the scalar bar.

    Returns
    -------
    Figure
        The figure containing the scalar bar.
    """
    if backend is None:
        backend = get_default_backend()
    params = {
        'mapper': mapper,
        'name': name,
        'below_color': below_color,
        'length': length,
        'width': width,
        'orientation': orientation,
        'num_sig_figs': num_sig_figs,
        'font': font,
        'name_fontsize_multiplier': name_fontsize_multiplier,
        'lim_fontsize_multiplier': lim_fontsize_multiplier,
        'font_color': font_color,
        'font_outline_color': font_outline_color,
        'font_outline_multiplier': font_outline_multiplier,
    }
    if backend == 'matplotlib':
        return build_scalar_bar_matplotlib(**params)
    elif backend == 'svg':
        return build_scalar_bar_svg(**params)
    raise ValueError(f'Unknown backend: {backend}')


def build_scalar_bar_matplotlib(
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


def build_scalar_bar_svg(
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

    static_length = num_sig_figs * width // 2
    dynamic_length = length - 2 * static_length
    dynamic = mapper.to_rgba(np.linspace(vmin, vmax, dynamic_length))

    dynamic_grad = svg.LinearGradient(
        id=f'{name}-gradient',
        elements=[
            svg.Stop(
                offset=i / len(dynamic),
                stop_color=colors.to_hex(dynamic[i]),
            )
            for i in range(len(dynamic))
        ]
    )
    dynamic = svg.Rect(
        x=static_length,
        y=0,
        width=dynamic_length,
        height=width,
        fill=f'url(#{name}-gradient)',
    )
    above = svg.Rect(
        x=length / 2,
        y=0,
        width=length / 2,
        height=width,
        fill=colors.to_hex(mapper.to_rgba(vmax)),
    )
    if below_color is not None:
        if len(below_color) == 4 and below_color[-1] == 0:
            # Not ideal, but looks better than transparent and too many
            # color bars actually end in black
            below_color = '#444444'
        else:
            below_color = colors.to_hex(below_color)
    else:
        below_color = colors.to_hex(mapper.to_rgba(vmin))
    below = svg.Rect(
        x=0,
        y=0,
        width=length / 2,
        height=width,
        fill=below_color,
    )

    font_color = colors.to_hex(font_color)
    font_outline_color = colors.to_hex(font_outline_color)

    common_params = {
        'y': 0.5 * width,
        'font_family': font,
        'fill': font_color,
        'dominant_baseline': 'middle',
    }
    vlim_params = {
        **common_params,
        'font_size': width * lim_fontsize_multiplier,
    }
    vmin_params = {
        'x': 0.02 * length,
        'text': f'{vmin:.{num_sig_figs}g}',
        'text_anchor': 'start',
        **vlim_params,
    }
    vmax_params = {
        'x': 0.98 * length,
        'text': f'{vmax:.{num_sig_figs}g}',
        'text_anchor': 'end',
        **vlim_params,
    }
    name_params = {
        'x': 0.5 * length,
        'text': name,
        'font_size': width * name_fontsize_multiplier,
        'text_anchor': 'middle',
        **common_params,
    }

    vmin_label_stroke = svg.Text(
        **vmin_params,
        stroke=font_outline_color,
        stroke_width=(
            font_outline_multiplier * width * lim_fontsize_multiplier
        ),
        stroke_linejoin='round',
    )
    vmin_label = svg.Text(**vmin_params)
    vmax_label_stroke = svg.Text(
        **vmax_params,
        stroke=font_outline_color,
        stroke_width=(
            font_outline_multiplier * width * lim_fontsize_multiplier
        ),
        stroke_linejoin='round',
    )
    vmax_label = svg.Text(**vmax_params)
    if name is not None:
        name_label_stroke = svg.Text(
            **name_params,
            stroke=font_outline_color,
            stroke_width=(
                font_outline_multiplier * width * name_fontsize_multiplier
            ),
            stroke_linejoin='round',
        )
        name_label = svg.Text(**name_params)
    else:
        name_label_stroke = None
        name_label = None
    elements = [
        dynamic_grad,
        above,
        below,
        dynamic,
        vmin_label_stroke,
        vmax_label_stroke,
        name_label_stroke,
        vmin_label,
        vmax_label,
        name_label,
    ]
    elements = [e for e in elements if e is not None]
    match orientation:
        case 'h':
            transform =[]
        case 'v':
            transform = [svg.Rotate(-90, 0, 0), svg.Translate(-length, 0)]
            # We nest the elements doubly so that future transforms applied
            # over these will use the canvas coordinate system instead of the
            # rotated coordinate system
            elements = [
                svg.G(
                    elements=elements,
                    transform=transform,
                )
            ]
    group = svg.G(
        id=f'scalarbar-{name}',
        elements=elements,
        transform=[],
    )
    return svg.SVG(
        width=length if orientation == 'h' else width,
        height=width if orientation == 'h' else length,
        elements=[group],
        viewBox=f'0 0 {length} {width}',
    )


def build_raster(
    *,
    content: Tensor,
    bounding_box_height: int = RASTER_DEFAULT_BOUNDING_BOX_HEIGHT,
    bounding_box_width: int = RASTER_DEFAULT_BOUNDING_BOX_WIDTH,
    fmt: str = RASTER_DEFAULT_FORMAT,
    backend: Optional[str] = None,
    **params: Any,
) -> Any:
    if backend is None:
        backend = get_default_backend()
    params = {
        'content': content,
        'bounding_box_height': bounding_box_height,
        'bounding_box_width': bounding_box_width,
        'fmt': fmt,
    }
    if backend == 'svg':
        return build_raster_svg(**params)
    raise ValueError(f'Unknown backend {backend}')


def build_raster_svg(
    *,
    content: Tensor,
    bounding_box_height: int = RASTER_DEFAULT_BOUNDING_BOX_HEIGHT,
    bounding_box_width: int = RASTER_DEFAULT_BOUNDING_BOX_WIDTH,
    fmt: str = RASTER_DEFAULT_FORMAT,
) -> svg.SVG:
    content = Image.fromarray(content)
    if fmt == 'png':
        content = content.convert('RGBA')
    else:
        raise ValueError(
            f'Unsupported format {fmt}. Only png is supported at the '
            'moment.'
        )
    buffer = io.BytesIO()
    content.save(buffer, format=fmt)
    buffer.seek(0)

    image_width, image_height = content.size
    scale = min(
        bounding_box_width / image_width, bounding_box_height / image_height
    )
    image_width = int(image_width * scale)
    image_height = int(image_height * scale)
    shift_x = (bounding_box_width - image_width) // 2
    shift_y = (bounding_box_height - image_height) // 2
    transform = [svg.Translate(shift_x, shift_y)]

    # By convention, we wrap it in a singleton group before returning
    img = svg.Image(
        href='data:image/png;base64,' + base64.b64encode(
            buffer.read()
        ).decode('ascii'),
        width=image_width,
        height=image_height,
    )
    return svg.SVG(
        width=bounding_box_width,
        height=bounding_box_height,
        elements=[svg.G(elements=[img], transform=transform)],
        viewBox=f'0 0 {bounding_box_width} {bounding_box_height}',
    )


def _uniquify_names(builders: Sequence[ElementBuilder]):
    unique_builders = set()
    retained_builders = []
    for builder in builders:
        if builder not in unique_builders:
            unique_builders.add(builder)
            retained_builders.append(builder)
    return retained_builders


def tile_plot_elements(
    builders: Sequence[ElementBuilder],
    spacing: float = SCALAR_BAR_DEFAULT_SPACING,
    max_dimension: Optional[Tuple[int, int]] = None,
    require_unique_names: bool = True,
) -> Optional[Tensor]:
    builders = [b for b in builders if b is not None]
    if len(builders) == 0:
        return None
    # Algorithm from https://stackoverflow.com/a/28268965
    if require_unique_names:
        builders = _uniquify_names(builders)
    builders = sorted(builders, key=lambda b: b.priority)
    count = len(builders)
    max_width, max_height = max_dimension
    if spacing < 1:
        spacing = spacing * min(max_width, max_height)
    spacing = int(spacing)
    width = median([
        builder.canvas_width
        for builder in builders
    ])
    height = median([
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

    builders_scaled = []
    for builder in builders:
        builder_width = builder.canvas_width
        builder_height = builder.canvas_height
        scale = min(width / builder_width, height / builder_height)
        builder = builder.set_canvas_size(
            height=int(scale * builder_height),
            width=int(scale * builder_width),
        )
        builders_scaled.append(builder)

    # layout is in array convention, i.e. (nrow, ncol)
    # max_dimension, active_canvas_size, and offset are in Cartesian
    # convention, i.e. (width, height)
    # Things are about to get confusing because we're switching back and forth
    # between these conventions. Sorry.
    #tight = {0: 'row', 1: 'col'}[np.argmin(layout)]
    active_canvas_size_guess = (
        layout[1] * width + (layout[1] - 1) * spacing,
        layout[0] * height + (layout[0] - 1) * spacing,
    )
    argtight = 1 - np.argmin([
        i - j
        for i, j in zip(max_dimension, active_canvas_size_guess)
    ])
    argslack = 1 - argtight
    counttight = layout[argtight]
    countslack = np.ceil(count / counttight).astype(int)
    layout = [None, None]
    layout[argtight] = counttight
    layout[argslack] = countslack

    active_canvas_size = (
        layout[1] * width + (layout[1] - 1) * spacing,
        layout[0] * height + (layout[0] - 1) * spacing,
    )
    offset = [0, 0]
    offset[1 - argslack] = (
        max_dimension[1 - argslack] - active_canvas_size[1 - argslack]
    ) // 2
    base_offset = [o for o in offset]
    canvas_elements = []
    for i, builder in enumerate(builders_scaled):
        element = builder().elements[0]
        internal_displacement = [
            (width - builder.canvas_width) // 2,
            (height - builder.canvas_height) // 2,
        ]
        transform = element.transform + [
            svg.Translate(
                offset[0] + internal_displacement[0],
                offset[1] + internal_displacement[1],
            )
        ]
        element.transform = transform
        canvas_elements.append(element)
        # Filling this in row-major order
        if i % layout[1] == layout[1] - 1:
            offset[0] = base_offset[0]
            offset[1] += height + spacing
        else:
            offset[0] += width + spacing

    collection = svg.G(
        id='elements2d-collection',
        elements=canvas_elements,
    )
    canvas = svg.SVG(
        width=max_width,
        height=max_height,
        viewBox=f'0 0 {max_width} {max_height}',
        elements=[collection],
    )

    return canvas
