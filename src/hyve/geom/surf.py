# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Brain surfaces
~~~~~~~~~~~~~~
Brain surface geometry data containers and geometric primitives.
"""
import dataclasses
import pathlib
import warnings
from typing import (
    Any,
    Iterable,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import nibabel as nb
import numpy as np
import pyvista as pv
import templateflow.api as tflow
from matplotlib import colors
from scipy.spatial.distance import cdist

from ..const import (
    DEFAULT_CMAP,
    DEFAULT_COLOR,
    LAYER_ALIM_DEFAULT_VALUE,
    LAYER_ALIM_NEGATIVE_DEFAULT_VALUE,
    LAYER_ALIM_PERCENTILE_DEFAULT_VALUE,
    LAYER_ALPHA_DEFAULT_VALUE,
    LAYER_AMAP_DEFAULT_VALUE,
    LAYER_AMAP_NEGATIVE_DEFAULT_VALUE,
    LAYER_CLIM_NEGATIVE_DEFAULT_VALUE,
    LAYER_CLIM_PERCENTILE_DEFAULT_VALUE,
    LAYER_CMAP_NEGATIVE_DEFAULT_VALUE,
    LAYER_COLOR_DEFAULT_VALUE,
    LAYER_NAN_OVERRIDE_DEFAULT_VALUE,
    SURF_DEFAULT_STYLE,
    SURF_SCALARS_BELOW_COLOR_DEFAULT_VALUE,
    SURF_SCALARS_CLIM_DEFAULT_VALUE,
    SURF_SCALARS_CMAP_DEFAULT_VALUE,
    SURF_SCALARS_DEFAULT_VALUE,
    SURF_SCALARS_LAYERS_DEFAULT_VALUE,
    CIfTIStructures,
    Tensor,
    neuromaps_fetch_fn,
    template_dict,
)
from ..elements import ScalarBarBuilder
from ..util import relabel_parcels, scalar_percentile
from .base import (
    Layer,
    SubgeometryParameters,
    compose_layers,
    layer_rgba,
)


def is_path_like(obj: Any) -> bool:
    """
    Check if an object could represent a path on the filesystem.

    Parameters
    ----------
    obj : Any
        Object to check.

    Returns
    -------
    bool
        True if the object is a string or a ``pathlib.Path`` object.
    """
    return isinstance(obj, (str, pathlib.Path))


def extend_to_max(
    *pparams: Sequence[Tensor],
    axis: int = -1,
) -> Iterable[Tensor]:
    """
    Extend multiple arrays along an axis to the maximum length among them.

    Parameters
    ----------
    *pparams : Tensor
        Arrays to extend.
    axis : int (default -1)
        Axis along which to extend.

    Yields
    ------
    Tensor
        Extended arrays.
    """
    max_len = max(p.shape[axis] for p in pparams)
    pad = [(0, 0)] * pparams[0].ndim
    for p in pparams:
        pad[axis] = (0, max_len - p.shape[axis])
        yield np.pad(p, pad, 'constant')


def get_cifti_brain_model_axis(
    cifti: nb.Cifti2Image,
) -> Tuple[Sequence[str], Sequence[str]]:
    return tuple(
        (i, a)
        for (i, a) in tuple(
            (i, cifti.header.get_axis(i)) for i in range(cifti.ndim)
        )
        if isinstance(a, nb.cifti2.cifti2_axes.BrainModelAxis)
    )[0]


# class ProjectionKey:
#     """
#     Currently, PyVista does not support non-string keys for point data. This
#     class is unused, but is a placeholder in case PyVista supports this in
#     the future.
#     """
#     def __init__(self, projection: str):
#         self.projection = projection

#     def __hash__(self):
#         return hash(f'_projection_{self.projection}')

#     def __eq__(self, other):
#         return self.projection == other.projection

#     def __repr__(self):
#         return f'Projection({self.projection})'


@dataclasses.dataclass
class ProjectedPolyData(pv.PolyData):
    """
    PyVista ``PolyData`` object with multiple projections.

    We can use this, for example, to associate the same vertex-wise or face-
    wise data with multiple projections of a surface. For example, we can
    associate parcellation or scalar-valued data with both the inflated and
    the white matter surfaces of a cortical hemisphere and switch between
    them.

    Projections are accessible via the ``projections`` property, which is a
    dictionary mapping projection names to the associated point data. The
    ``points`` property is always the current projection.

    Parameters
    ----------
    *pparams : Sequence[Any]
        Positional parameters to pass to ``PolyData``.
    projection : str
        Name of the projection to use as the initial projection.
    **params : Mapping[str, Any]
        Keyword parameters to pass to ``PolyData``.
    """

    def __init__(
        self,
        *pparams: Sequence[Any],
        projection: str = None,
        **params: Mapping[str, Any],
    ):
        super().__init__(*pparams, **params)
        self.point_data[f'_projection_{projection}'] = self.points.copy()
        self.active_projection = projection

    def __repr__(self):
        return super().__repr__()

    @property
    def projections(self) -> Mapping[str, Tensor]:
        """
        Dictionary mapping projection names to point data.
        """
        return {
            k[12:]: v
            for k, v in self.point_data.items()
            if k[:12] == '_projection_'
        }

    @classmethod
    def from_polydata(
        cls,
        *,
        projection: str,
        ignore_errors: bool = False,
        **params: Mapping[str, Any],
    ) -> 'ProjectedPolyData':
        """
        Create a ``ProjectedPolyData`` object by combining multiple
        ``PolyData`` objects that each represent a projection.

        Parameters
        ----------
        projection : str
            Name of the projection to use as the initial projection.
        ignore_errors : bool (default False)
            If True, ignore errors when adding a projection.
        **params : Mapping[str, Any]
            Any additional parameters should be keyword arguments of the form
            ``<projection>=<PolyData>``.
        """
        obj = cls(params[projection], projection=projection)
        for key, value in params.items():
            if key != projection:
                if ignore_errors:
                    try:
                        obj.add_projection(key, value.points)
                    except Exception:
                        continue
                else:
                    obj.add_projection(key, value.points)
        return obj

    def get_projection(self, projection: str) -> Tensor:
        """
        Get the point coordinates for a projection.

        Parameters
        ----------
        projection : str
            Name of the projection.

        Returns
        -------
        Tensor
            Point coordinates for the projection.
        """
        projections = self.projections
        if projection not in projections:
            raise KeyError(
                f'Projection "{projection}" not found. '
                f'Available projections: {set(projections.keys())}'
            )
        return projections[projection]

    def add_projection(
        self,
        projection: str,
        points: Tensor,
    ) -> None:
        """
        Add a projection.

        Parameters
        ----------
        projection : str
            Name of the projection.
        points : Tensor
            Coordinates of each vertex in the projection.
        """
        if len(points) != self.n_points:
            raise ValueError(
                f'Number of points {len(points)} must match {self.n_points}.'
            )
        self.point_data[f'_projection_{projection}'] = points

    def remove_projection(
        self,
        projection: str,
    ) -> None:
        """
        Remove a projection.

        Parameters
        ----------
        projection : str
            Name of the projection to remove.
        """
        projections = self.projections
        if projection not in projections:
            raise KeyError(
                f'Projection "{projection}" not found. '
                f'Available projections: {set(projections.keys())}'
            )
        del self.point_data[f'_projection_{projection}']

    def project(
        self,
        projection: str,
    ) -> None:
        """
        Switch to a different projection.

        Parameters
        ----------
        projection : str
            Name of the projection to switch to.
        """
        if projection == self.active_projection:
            return
        projections = self.projections
        if projection not in projections:
            raise KeyError(
                f'Projection "{projection}" not found. '
                f'Available projections: {set(projections.keys())}'
            )
        self.points = projections[projection]
        self.active_projection = projection

    # TODO: Doing this "the right way" (the commented-out code below) is
    #       prone to segfaults when PyVista makes deep copies. I'm not sure
    #       why. For now, we just do it the wrong way (the uncommented code
    #       below) and work around it in `hemisphere_slack_transform_surf`.
    def translate(
        self,
        xyz: Tensor,
        # transform_all_input_vectors: bool = False,
    ) -> 'ProjectedPolyData':
        points = self.points + xyz
        translated_proj = f'{self.active_projection}_translated'
        self.add_projection(translated_proj, points)
        self.project(translated_proj)
        return self
        # translated = super().translate(xyz, transform_all_input_vectors)
        # translated.active_projection = self.active_projection
        # proj_coor = self.get_projection(self.active_projection)
        # translated.add_projection(self.active_projection, proj_coor + xyz)
        # return translated

    def init_composition(
        self,
        layers: Sequence[Layer],
        *,
        data_domain: Literal['point_data', 'cell_data'] = 'cell_data',
        **params,
    ) -> None:
        dst = layers[0]
        try:
            cmap = dst.cmap or DEFAULT_CMAP
            color = None
            try:
                color_array = self.cell_data[dst.color]
            except KeyError:
                color_array = self.point_data[dst.color]
                data_domain = 'point_data'
        except Exception:
            cmap = None
            color = dst.color or DEFAULT_COLOR
            try:
                self.cell_data[layers[1].name]
                color_array = np.zeros(self.n_cells)
            except (KeyError, IndexError):
                data_domain = 'point_data'
                color_array = np.zeros(self.n_points)
        if isinstance(dst.alpha, str):
            alpha_array = self.__getattribute__(
                data_domain
            )[dst.alpha]
        else:
            alpha_array = None
        dst = Layer(
            name=dst.name,
            cmap=cmap,
            cmap_negative=dst.cmap_negative,
            clim=dst.clim,
            clim_negative=dst.clim_negative,
            clim_percentile=dst.clim_percentile,
            color=color,
            color_negative=dst.color_negative,
            alpha=dst.alpha,
            alpha_negative=dst.alpha_negative,
            alim=dst.alim,
            alim_negative=dst.alim_negative,
            amap=dst.amap,
            amap_negative=dst.amap_negative,
            below_color=dst.below_color,
            nan_override=dst.nan_override,
            hide_subthreshold=dst.hide_subthreshold,
            scalar_bar_style=dst.scalar_bar_style,
        )
        return dst, (color_array, alpha_array), {'data_domain': data_domain}

    def layer_to_tensors(
        self,
        layer: Layer,
        data_domain: Literal['point_data', 'cell_data'],
    ) -> Tuple[Tensor, Sequence[ScalarBarBuilder]]:
        """
        Convert a layer to RGBA colors.
        """
        try:
            try:
                color_array = self.__getattribute__(
                    data_domain
                )[layer.color]
                color = None
            except KeyError:
                colors.to_rgba(layer.color) # Check if it's a valid color
                color_array = np.zeros(
                    self.n_points
                    if data_domain == 'point_data'
                    else self.n_cells
                )
                if layer.name in self.__getattribute__(data_domain):
                    color_array = np.where(
                        np.isnan(self.__getattribute__(data_domain)[layer.name]),
                        np.nan,
                        color_array,
                    )
                color = layer.color
            if isinstance(layer.alpha, str):
                alpha_array = self.__getattribute__(
                    data_domain
                )[layer.alpha]
                alpha_repl = None
            else:
                alpha_array = None
                alpha_repl = layer.alpha
            layer = dataclasses.replace(
                layer,
                color=color,
                alpha=alpha_repl,
            )
        except (KeyError, ValueError):
            raise ValueError(
                'All layers must be defined over the same data '
                'domain. The base layer is defined over '
                f'{data_domain}, but either the layer {layer.color} or '
                f'the alpha scalars {layer.alpha} (if string) was not '
                f'found in {data_domain}, or {layer.color} is an invalid '
                'matplotlib colour specification. In particular, ensure '
                'that, if you have mapped any layer to faces, all '
                'other layers are also mapped to faces.'
            )
        rgba, scalar_bar_builders = layer_rgba(
            layer=layer,
            color_scalar_array=color_array,
            alpha_scalar_array=alpha_array,
        )

        return rgba, scalar_bar_builders

    def paint(
        self,
        plotter: pv.Plotter,
        layers: Sequence[Layer],
        surf_alpha: Optional[float] = None,
        style: Union[Mapping, Literal['__default__']] = '__default__',
        copy_actors: bool = False,
    ) -> Tuple[pv.Plotter, Sequence[ScalarBarBuilder]]:
        layers = [
            layer
            if layer.color is not None
            else dataclasses.replace(
                layer,
                color=layer.name,
            )
            for layer in layers
        ]
        rgba, scalar_bar_builders, extra_params = compose_layers(self, layers)
        data_domain = extra_params.get('data_domain', 'cell_data')
        # We shouldn't have to do this, but for some reason exporting to
        # HTML adds transparency even if we set alpha to 1 when we use
        # explicit RGBA to colour the mesh. Dropping the alpha channel
        # fixes this.
        if (surf_alpha is None):
            rgba = rgba[:, :3]

        name = '-'.join([str(layer.name) for layer in layers])
        name = f'{name}_{data_domain}_rgba'
        if data_domain == 'cell_data':
            self.cell_data[name] = rgba
        elif data_domain == 'point_data':
            self.point_data[name] = rgba
        if style == '__default__':
            style = SURF_DEFAULT_STYLE
        else:
            style = {**SURF_DEFAULT_STYLE, **style}
        # fallback, because setting the scalars using the arg in the add_mesh
        # call below doesn't always work
        self.set_active_scalars(name)
        # TODO: copying the mesh seems like it could create memory issues.
        #       A better solution would be delayed execution.
        plotter.add_mesh(
            self,
            scalars=rgba,
            rgb=True,
            **style,
            copy_mesh=copy_actors,
        )
        return plotter, scalar_bar_builders


@dataclasses.dataclass
class CortexTriSurface:
    """
    A pair of ``ProjectedPolyData`` objects representing the left and right
    cortical hemispheres.

    In general, it is not recommended to instantiate this class directly.
    Instead, use one of the class method constructors according to the
    input data format.

    Parameters
    ----------
    left : ProjectedPolyData
        Left hemisphere.
    right : ProjectedPolyData
        Right hemisphere.
    mask : str, optional
        Name of the vertex dataset containing the mask for both hemispheres.
    """

    left: ProjectedPolyData
    right: ProjectedPolyData
    mask: Optional[str] = None

    @classmethod
    def from_darrays(
        cls,
        left: Mapping[str, Tuple[Tensor, Tensor]],
        right: Mapping[str, Tuple[Tensor, Tensor]],
        left_mask: Optional[Tensor] = None,
        right_mask: Optional[Tensor] = None,
        projection: Optional[str] = None,
    ) -> 'CortexTriSurface':
        """
        Create a ``CortexTriSurface`` from dictionaries of data arrays.

        Parameters
        ----------
        left : Mapping[str, Tuple[Tensor, Tensor]]
            Dictionary mapping projection dataset names to
            ``(vertex coordinates, triangles)`` tuples for the left
            hemisphere.
        right : Mapping[str, Tuple[Tensor, Tensor]]
            Dictionary mapping projection dataset names to
            ``(vertex coordinates, triangles)`` tuples for the right
            hemisphere.
        left_mask : Tensor, optional
            Boolean scalar array indicating whether each vertex is included in
            the left hemisphere when adding scalar vertex-wise datasets. If
            None, all vertices are included.
        right_mask : Tensor, optional
            Boolean scalar array indicating whether each vertex is included in
            the right hemisphere when adding scalar vertex-wise datasets. If
            None, all vertices are included.
        projection : str, optional
            Name of the projection to use as the initial projection. If None,
            the first projection in the ``left`` dictionary is used.

        Returns
        -------
        CortexTriSurface
            A ``CortexTriSurface`` object.
        """
        if projection is None:
            projection = list(left.keys())[0]
        left, mask_str = cls._hemisphere_darray_impl(
            left,
            left_mask,
            projection,
        )
        right, _ = cls._hemisphere_darray_impl(
            right,
            right_mask,
            projection,
        )
        return cls(left, right, mask_str)

    @classmethod
    def from_gifti(
        cls,
        left: Mapping[str, Union[str, nb.gifti.gifti.GiftiImage]],
        right: Mapping[str, Union[str, nb.gifti.gifti.GiftiImage]],
        left_mask: Optional[Union[str, nb.gifti.gifti.GiftiImage]] = None,
        right_mask: Optional[Union[str, nb.gifti.gifti.GiftiImage]] = None,
        projection: Optional[str] = None,
    ) -> 'CortexTriSurface':
        """
        Create a ``CortexTriSurface`` from GIFTI files.

        Parameters
        ----------
        left : Mapping[str, Union[str, nb.gifti.gifti.GiftiImage]]
            Dictionary mapping projection dataset names to GIFTI files or
            ``nibabel`` ``GiftiImage`` objects for the left hemisphere. Each
            file or ``GiftiImage`` object must contain data arrays
            corresponding to vertex coordinates and triangles.
        right : Mapping[str, Union[str, nb.gifti.gifti.GiftiImage]]
            Dictionary mapping projection dataset names to GIFTI files or
            ``nibabel`` ``GiftiImage`` objects for the right hemisphere. Each
            file or ``GiftiImage`` object must contain data arrays
            corresponding to vertex coordinates and triangles.
        left_mask : Union[str, nb.gifti.gifti.GiftiImage], optional
            GIFTI file or ``nibabel`` ``GiftiImage`` object containing a
            boolean scalar array indicating whether each vertex is included in
            the left hemisphere when adding scalar vertex-wise datasets. If
            None, all vertices are included.
        right_mask : Union[str, nb.gifti.gifti.GiftiImage], optional
            GIFTI file or ``nibabel`` ``GiftiImage`` object containing a
            boolean scalar array indicating whether each vertex is included in
            the right hemisphere when adding scalar vertex-wise datasets. If
            None, all vertices are included.
        projection : str, optional
            Name of the projection to use as the initial projection. If None,
            the first projection in the ``left`` dictionary is used.
        """
        left, left_mask = cls._hemisphere_gifti_impl(left, left_mask)
        right, right_mask = cls._hemisphere_gifti_impl(right, right_mask)
        return cls.from_darrays(
            left={
                k: tuple(d.data for d in v.darrays) for k, v in left.items()
            },
            right={
                k: tuple(d.data for d in v.darrays) for k, v in right.items()
            },
            left_mask=left_mask,
            right_mask=right_mask,
            projection=projection,
        )

    @classmethod
    def _from_archive(
        cls,
        fetch_fn: callable,
        coor_query: dict,
        mask_query: dict,
        projections: Union[str, Sequence[str]],
        load_mask: bool,
    ) -> 'CortexTriSurface':
        """
        Helper method to create a ``CortexTriSurface`` from a cloud-based data
        archive. Used by the class method constructors ``from_tflow`` and
        ``from_nmaps``.
        """
        if isinstance(projections, str):
            projections = (projections,)
        lh, rh = {}, {}
        for projection in projections:
            coor_query.update(suffix=projection)
            lh_path, rh_path = (
                fetch_fn(**coor_query, hemi='L'),
                fetch_fn(**coor_query, hemi='R'),
            )
            lh[projection] = lh_path
            rh[projection] = rh_path
        lh_mask, rh_mask = None, None
        if load_mask:
            lh_mask, rh_mask = (
                fetch_fn(**mask_query, hemi='L'),
                fetch_fn(**mask_query, hemi='R'),
            )
        return cls.from_gifti(
            lh, rh, lh_mask, rh_mask, projection=projections[0]
        )

    @classmethod
    def from_tflow(
        cls,
        template: str = 'fsLR',
        projections: Union[str, Sequence[str]] = 'veryinflated',
        load_mask: bool = False,
    ) -> 'CortexTriSurface':
        """
        Create a ``CortexTriSurface`` from a reference in the TemplateFlow
        data archive.

        Parameters
        ----------
        template : str (default='fsLR')
            TemplateFlow template name. Must be a surface template like
            ``fsLR`` or ``fsaverage``.
        projections : str or sequence of str (default='veryinflated')
            Name(s) of the projection to load into the ``CortexTriSurface``.
            The first projection in the sequence is used as the initial
            projection.
        load_mask : bool (default=False)
            Whether to load the surface mask from the TemplateFlow data
            archive. If True, the mask is loaded as a boolean scalar vertex-
            wise dataset named ``__mask__``.
        """
        template = template_dict()[template]
        return cls._from_archive(
            fetch_fn=tflow.get,
            coor_query=template.TFLOW_COOR_QUERY,
            mask_query=template.TFLOW_MASK_QUERY,
            projections=projections,
            load_mask=load_mask,
        )

    @classmethod
    def from_nmaps(
        cls,
        template: str = 'fsaverage',
        projections: Union[str, Sequence[str]] = 'pial',
        load_mask: bool = False,
    ) -> 'CortexTriSurface':
        """
        Create a ``CortexTriSurface`` from a reference in the Neuromaps data
        archive.

        Parameters
        ----------
        template : str (default='fsaverage')
            Neuromaps template name. Must be a surface template like
            ``fsLR`` or ``fsaverage``.
        projections : str or sequence of str (default='pial')
            Name(s) of the projection to load into the ``CortexTriSurface``.
            The first projection in the sequence is used as the initial
            projection.
        load_mask : bool (default=False)
            Whether to load the surface mask from the Neuromaps data archive.
            If True, the mask is loaded as a boolean scalar vertex-wise
            dataset named ``__mask__``.
        """
        template = template_dict()[template]
        return cls._from_archive(
            fetch_fn=neuromaps_fetch_fn,
            coor_query=template.NMAPS_COOR_QUERY,
            mask_query=template.NMAPS_MASK_QUERY,
            projections=projections,
            load_mask=load_mask,
        )

    @property
    def n_points(self) -> Mapping[str, int]:
        """Number of vertices in each hemisphere."""
        return {
            'left': self.left.n_points,
            'right': self.right.n_points,
        }

    @property
    def point_data(self) -> Mapping[str, Mapping]:
        """Dictionary of point data for each hemisphere."""
        return {
            'left': self.left.point_data,
            'right': self.right.point_data,
        }

    @property
    def masks(self) -> Mapping[str, Tensor]:
        """Dictionary of masks for each hemisphere."""
        return {
            'left': self.left.point_data[self.mask],
            'right': self.right.point_data[self.mask],
        }

    @property
    def mask_size(self) -> Mapping[str, int]:
        """
        Number of vertices that are True (included) in each hemisphere mask.
        """
        return {
            'left': self.masks['left'].sum().item(),
            'right': self.masks['right'].sum().item(),
        }

    def add_cifti_dataset(
        self,
        name: str,
        cifti: Union[str, nb.cifti2.cifti2.Cifti2Image],
        is_masked: bool = False,
        apply_mask: bool = True,
        null_value: Optional[float] = 0.0,
        select: Optional[Sequence[int]] = None,
        exclude: Optional[Sequence[int]] = None,
        allow_multihemisphere: bool = True,
        coerce_to_scalar: bool = True,
    ):
        """
        Add a CIFTI dataset to the ``CortexTriSurface``.

        Parameters
        ----------
        name : str
            Name of the dataset.
        cifti : str or ``Cifti2Image``
            Path to a CIFTI file or a ``nibabel`` ``Cifti2Image`` object.
        is_masked : bool (default=False)
            Indicates whether the CIfTI dataset includes values for all
            vertices (``is_masked=False``) or only for vertices included in
            the surface mask (``is_masked=True``).
        apply_mask : bool (default=True)
            Whether to apply the surface mask to the CIFTI dataset before
            adding it to the ``CortexTriSurface``.
        null_value : float or None (default=0.)
            Value to use for vertices excluded by the surface mask.
        """
        names_dict = {
            CIfTIStructures.LEFT: 'left',
            CIfTIStructures.RIGHT: 'right',
        }
        slices = {}

        if is_path_like(cifti):
            cifti = nb.load(cifti)
        model_ax_idx, model_axis = get_cifti_brain_model_axis(cifti)

        for struc, slc, _ in model_axis.iter_structures():
            hemi = names_dict.get(struc, None)
            if hemi is not None:
                start, stop = slc.start, slc.stop
                start = start if start is not None else 0
                stop = (
                    stop if stop is not None else cifti.shape[model_ax_idx]
                )
                slices[hemi] = slice(start, stop)

        # Hmm, why are we casting to float32 here? No harm, but it's not
        # obvious why we're doing it instead of something more principled.
        data = cifti.get_fdata(dtype='float32')
        while data.shape[0] == 1:
            data = data[0]
        return self.add_vertex_dataset(
            name=name,
            data=data,
            left_slice=slices['left'],
            right_slice=slices['right'],
            default_slices=False,
            is_masked=is_masked,
            apply_mask=apply_mask,
            null_value=null_value,
            select=select,
            exclude=exclude,
            allow_multihemisphere=allow_multihemisphere,
            coerce_to_scalar=coerce_to_scalar,
        )

    def add_gifti_dataset(
        self,
        name: str,
        left_gifti: Optional[Union[str, nb.gifti.gifti.GiftiImage]] = None,
        right_gifti: Optional[Union[str, nb.gifti.gifti.GiftiImage]] = None,
        is_masked: bool = False,
        apply_mask: bool = True,
        null_value: Optional[float] = 0.0,
        select: Optional[Sequence[int]] = None,
        exclude: Optional[Sequence[int]] = None,
        allow_multihemisphere: bool = True,
        coerce_to_scalar: bool = True,
    ):
        """
        Add a GIFTI dataset (or datasets) to the ``CortexTriSurface``.

        Parameters
        ----------
        name : str
            Name of the dataset.
        left_gifti : str or ``GiftiImage`` or None (default=None)
            Path to a GIFTI file or a ``nibabel`` ``GiftiImage`` object for
            the left hemisphere. If None, the dataset is assumed to be
            right-only.
        right_gifti : str or ``GiftiImage`` or None (default=None)
            Path to a GIFTI file or a ``nibabel`` ``GiftiImage`` object for
            the right hemisphere. If None, the dataset is assumed to be
            left-only.
        is_masked : bool (default=False)
            Indicates whether the GIFTI dataset includes values for all
            vertices (``is_masked=False``) or only for vertices included in
            the surface mask (``is_masked=True``).
        apply_mask : bool (default=True)
            Whether to apply the surface mask to the GIFTI dataset before
            adding it to the ``CortexTriSurface``.
        null_value : float or None (default=0.)
            Value to use for vertices excluded by the surface mask.
        map_all : bool (default=True)
            If the GIFTI datasets include multiple data arrays, whether to map
            all of them to the ``CortexTriSurface``. If False, only the array
            indicated by ``arr_idx`` will be mapped. If all arrays are mapped,
            the name of each array will be appended with ``_{i}``, where ``i``
            is the index of the array.
        arr_idx : int (default=0)
            Index of the data array to map to the ``CortexTriSurface`` if
            ``map_all=False``.
        select : sequence of int or None (default=None)
            Indices of the data arrays to map to the ``CortexTriSurface`` if
            ``map_all=True`` and ``exclude=None``.
        exclude : sequence of int or None (default=None)
            Indices of the data arrays to exclude from mapping to the
            ``CortexTriSurface`` if ``map_all=True``. If ``exclude=None``, all
            arrays will be mapped unless ``select`` is specified.
        """
        if isinstance(left_gifti, str):
            left_gifti = nb.load(left_gifti)
        if isinstance(right_gifti, str):
            right_gifti = nb.load(right_gifti)
        left_data = left_gifti.darrays if left_gifti else []
        right_data = right_gifti.darrays if right_gifti else []

        left_array = right_array = None
        if left_gifti and len(left_data) > 0:
            left_data = [
                d.data[None, ...] if d.data.ndim == 1 else d.data
                for d in left_data
            ]
            left_array = np.concatenate(left_data, axis=0)
        if right_gifti and len(right_data) > 0:
            right_data = [
                d.data[None, ...] if d.data.ndim == 1 else d.data
                for d in right_data
            ]
            right_array = np.concatenate(right_data, axis=0)

        return self.add_vertex_dataset(
            name=name,
            left_data=left_array,
            right_data=right_array,
            is_masked=is_masked,
            apply_mask=apply_mask,
            null_value=null_value,
            select=select,
            exclude=exclude,
            allow_multihemisphere=allow_multihemisphere,
            coerce_to_scalar=coerce_to_scalar,
        )

    def add_vertex_dataset(
        self,
        name: str,
        data: Optional[Tensor] = None,
        left_data: Optional[Tensor] = None,
        right_data: Optional[Tensor] = None,
        left_slice: Optional[slice] = None,
        right_slice: Optional[slice] = None,
        default_slices: bool = True,
        is_masked: bool = False,
        apply_mask: bool = True,
        null_value: Optional[float] = 0.0,
        oob_value: Optional[float] = np.nan,
        select: Optional[Sequence[int]] = None,
        exclude: Optional[Sequence[int]] = None,
        allow_multihemisphere: bool = True,
        coerce_to_scalar: bool = True,
    ):
        """
        Add a vertex-wise dataset to the ``CortexTriSurface``.

        Examples of vertex-wise datasets include vertex-wise curvature,
        vertex- wise thickness, vertex-wise functional activation maps,
        parcellations, etc.

        Either the parameters ``data`` or ``left_data`` and ``right_data``
        must be provided. If ``data`` is provided, it will be split into
        ``left_data`` and ``right_data`` based on the ``left_slice`` and
        ``right_slice`` parameters.

        If ``left_slice`` and ``right_slice`` are not provided, they will
        be inferred automatically from either the number of vertices in each
        hemisphere's ``ProjectedPolyData`` or the number of True values in
        each hemisphere's ``__mask__`` tensor; this decision is made based on
        whether the provided data are already masked or not, as indicated by
        the ``is_masked`` parameter.

        If the provided data are not already masked but should be prior to
        adding the dataset, the ``apply_mask`` parameter can be toggled on.

        Parameters
        ----------
        name : str
            Name of the dataset.
        data : ``torch.Tensor`` or ``numpy.ndarray``
            Tensor of vertex-wise data for the entire cortex. If provided,
            ``left_data`` and ``right_data`` must be ``None``.
        left_data : ``torch.Tensor`` or ``numpy.ndarray``
            Tensor of vertex-wise data for the left hemisphere. If provided,
            ``data`` must be ``None``.
        right_data : ``torch.Tensor`` or ``numpy.ndarray``
            Tensor of vertex-wise data for the right hemisphere. If provided,
            ``data`` must be ``None``.
        left_slice : ``slice`` (default: ``None``)
            If ``data`` is provided, this slice will be used to extract the
            left hemisphere data from the ``data`` tensor. If no slice is
            provided, the left hemisphere data will be extracted from the
            first half of the ``data`` tensor according to the value of the
            ``is_masked`` parameter.
        right_slice : ``slice`` (default: ``None``)
            If ``data`` is provided, this slice will be used to extract the
            right hemisphere data from the ``data`` tensor. If no slice is
            provided, the right hemisphere data will be extracted from the
            second half of the ``data`` tensor according to the value of the
            ``is_masked`` parameter.
        default_slices : bool (default: ``True``)
            If ``True``, default slices will be used to extract the left and
            right hemisphere data from the ``data`` tensor if no slices are
            provided. If ``False``, an error will be raised if no slices are
            provided.
        is_masked : bool (default: ``False``)
            If ``True``, the provided data are assumed to already be masked.
            If needed, the ``data`` tensor will then be sliced into left and
            right hemisphere data based on the size of the left hemisphere's
            mask. If ``False``, the provided data are assumed to be unmasked.
            If needed, the ``data`` tensor will then be sliced into left and
            right hemisphere data based on the number of vertices in each
            hemisphere's ``ProjectedPolyData``.
        apply_mask : bool (default: ``True``)
            If ``True``, the provided data will be masked prior to adding the
            dataset (unless they are already masked). If ``False``, the
            provided data will be added as-is.
        null_value : float (default: ``0.``)
            Null value to use when masking the provided data. Any values not
            in the mask will be replaced with or set to this value.
        """
        if data is not None:
            if left_slice is None and right_slice is None:
                if default_slices:
                    if is_masked:
                        left_slice = slice(0, self.mask_size['left'])
                        right_slice = slice(self.mask_size['left'], None)
                    else:
                        left_slice = slice(0, self.n_points['left'])
                        right_slice = slice(self.n_points['left'], None)
                else:
                    warnings.warn(
                        'No slices were provided for vertex data, and '
                        'default slicing was toggled off. The `data` tensor '
                        'will be ignored. Attempting fallback to `left_data` '
                        'and `right_data`.\n\n'
                        'To silence this warning, provide slices for the '
                        'data or toggle on default slicing if a `data` tensor'
                        ' is provided. Alternatively, provide `left_data` '
                        'and `right_data` directly and exclusively.'
                    )
            if left_slice is not None:
                if left_data is not None:
                    warnings.warn(
                        'Both `left_data` and `left_slice` were provided. '
                        'The `left_data` tensor will be ignored. '
                        'To silence this warning, provide `left_data` or '
                        '`left_slice` exclusively.'
                    )
                left_data = data[..., left_slice]
            if right_slice is not None:
                if right_data is not None:
                    warnings.warn(
                        'Both `right_data` and `right_slice` were provided. '
                        'The `right_data` tensor will be ignored. '
                        'To silence this warning, provide `right_data` or '
                        '`right_slice` exclusively.'
                    )
                right_data = data[..., right_slice]
        if left_data is None and right_data is None:
            raise ValueError(
                'Either no data was provided, or insufficient information '
                'was provided to slice the data into left and right '
                'hemispheres.'
            )
        offset = 0
        if coerce_to_scalar:
            left_data, right_data = left_data.squeeze(), right_data.squeeze()
        if not allow_multihemisphere:
            left_data, right_data = relabel_parcels(
                left_data,
                right_data,
                null_value=null_value,
            )

        if left_data is None:
            count_left = 0
        elif left_data.ndim == 1:
            count_left = 1
        else:
            count_left = left_data.shape[0]
        if right_data is None:
            count_right = 0
        elif right_data.ndim == 1:
            count_right = 1
        else:
            count_right = right_data.shape[0]
        if allow_multihemisphere:
            count = max(count_left, count_right)
        else:
            count = count_left + count_right
        zero_prefix = int(np.ceil(np.log10(count)))

        if left_data is not None:
            scalars_names_L = self._assign_hemisphere_vertex_data(
                name=name,
                data=left_data,
                is_masked=is_masked,
                apply_mask=apply_mask,
                null_value=oob_value,
                hemisphere='left',
                select=select,
                exclude=exclude,
                offset=offset,
                coerce_to_scalar=coerce_to_scalar,
                zero_prefix=zero_prefix,
            )
        if not allow_multihemisphere:
            offset = len(scalars_names_L)
        if right_data is not None:
            scalars_names_R = self._assign_hemisphere_vertex_data(
                name=name,
                data=right_data,
                is_masked=is_masked,
                apply_mask=apply_mask,
                null_value=oob_value,
                hemisphere='right',
                select=select,
                exclude=exclude,
                offset=offset,
                coerce_to_scalar=coerce_to_scalar,
                zero_prefix=zero_prefix,
            )
        return tuple(set(scalars_names_L + scalars_names_R))

    def parcellate_vertex_dataset(
        self,
        name: str,
        parcellation: str,
    ) -> Tensor:
        """
        Map a vertex-wise dataset to a parcellated dataset. Parcellation is
        performed by computing the mean of the vertex-wise data for each
        parcel.

        Parameters
        ----------
        name : str
            Name of the dataset.
        parcellation : str
            Name of the parcellation dataset to use for parcellation.

        Returns
        -------
        parcellated_data : Tensor
            Parcellated data.
        """
        l_num, l_denom = self._hemisphere_parcellate_impl(
            name, parcellation, 'left'
        )
        r_num, r_denom = self._hemisphere_parcellate_impl(
            name, parcellation, 'right'
        )
        l_num, r_num = extend_to_max(l_num, r_num, axis=0)
        l_denom, r_denom = extend_to_max(l_denom, r_denom, axis=0)
        if r_num.ndim == 1:
            r_num = r_num[:, None]
        if l_num.ndim == 1:
            l_num = l_num[:, None]
        return (l_num + r_num) / (l_denom + r_denom)

    def rename_array(self, src: str, dst: str):
        """
        Rename a dataset array.

        Parameters
        ----------
        src : str
            Name of the dataset to rename.
        dst : str
            New name of the dataset.
        """
        self.left.rename_array(src, dst)
        self.right.rename_array(src, dst)

    def scatter_into_parcels(
        self,
        data: Tensor,
        parcellation: str,
        sink: Optional[str] = None,
    ) -> Tensor:
        """
        Scatter a parcellated dataset into a vertex-wise dataset. Parcellation
        is performed by assigning the value of each parcel to all vertices in
        that parcel.

        Parameters
        ----------
        data : Tensor
            Data to scatter into parcels.
        parcellation : str
            Name of the parcellation dataset to use for parcellation.
        sink : str (default: ``None``)
            Name of the dataset to add the scattered data to. If ``None``, the
            scattered data will not be added to the ``ProjectedPolyData``.

        Returns
        -------
        scattered_data : Tensor
            Scattered data.
        """
        scattered_left = self._hemisphere_into_parcels_impl(
            data, parcellation, 'left'
        )
        scattered_right = self._hemisphere_into_parcels_impl(
            data, parcellation, 'right'
        )
        if sink is not None:
            return self.add_vertex_dataset(
                name=sink,
                left_data=scattered_left.T,
                right_data=scattered_right.T,
            )
        return (scattered_left, scattered_right)

    def vertex_to_face(
        self,
        name: str,
        interpolation: Literal['mode', 'mean'] = 'mode',
    ):
        """
        Resample a vertex-wise scalar dataset onto mesh faces. The face-valued
        scalar dataset will be added to the ``ProjectedPolyData`` as a face
        dataset.

        Parameters
        ----------
        name : str
            Name of the dataset.
        interpolation : str (default: ``'mode'``)
            Interpolation method to use. Must be one of ``'mode'`` or
            ``'mean'``. ``'mode'`` will assign the most common value of the
            neighbouring vertices to the face. ``'mean'`` will assign the mean
            value of the neighbouring vertices to the face.
        """
        if interpolation not in ('mode', 'mean'):
            raise ValueError(
                'Interpolation method must be one of "mode" or "mean".'
            )
        points_name = f'{name}:points'
        self.left.cell_data[name] = self._hemisphere_resample_v2f_impl(
            name, interpolation, 'left'
        )
        self.left.point_data[points_name] = self.left.point_data[name]
        self.left.point_data.remove(name)
        self.right.cell_data[name] = self._hemisphere_resample_v2f_impl(
            name, interpolation, 'right'
        )
        self.right.point_data[points_name] = self.right.point_data[name]
        self.right.point_data.remove(name)

    def select_active_parcels(
        self,
        parcellation_name: str,
        activation_name: str,
        activation_threshold: float = 2.58,
        parcel_coverage_threshold: float = 0.5,
        use_abs: bool = False,
    ):
        if (
            (parcellation_name in self.left.point_data) or
            (parcellation_name in self.right.point_data)
        ):
            assert (
                activation_name in self.left.point_data or
                activation_name in self.right.point_data
            ), (
                f'Activation dataset {activation_name} must be defined over '
                'vertices.'
            )
            domain = 'vertex'
            parcellation_left = self.left.point_data[parcellation_name]
            parcellation_right = self.right.point_data[parcellation_name]
            activation_left = self.left.point_data[activation_name]
            activation_right = self.right.point_data[activation_name]
        elif (
            (parcellation_name in self.left.cell_data) or
            (parcellation_name in self.right.cell_data)
        ):
            assert (
                activation_name in self.left.cell_data or
                activation_name in self.right.cell_data
            ), (
                f'Activation dataset {activation_name} must be defined over '
                'faces.'
            )
            domain = 'face'
            parcellation_left = self.left.cell_data[parcellation_name]
            parcellation_right = self.right.cell_data[parcellation_name]
            activation_left = self.left.cell_data[activation_name]
            activation_right = self.right.cell_data[activation_name]
        self._hemisphere_select_active_parcels_impl(
            'left',
            parcellation_name,
            parcellation=np.nan_to_num(parcellation_left).astype(int),
            activation=activation_left,
            activation_threshold=activation_threshold,
            parcel_coverage_threshold=parcel_coverage_threshold,
            use_abs=use_abs,
            domain=domain,
        )
        self._hemisphere_select_active_parcels_impl(
            'right',
            parcellation_name,
            parcellation=np.nan_to_num(parcellation_right).astype(int),
            activation=activation_right,
            activation_threshold=activation_threshold,
            parcel_coverage_threshold=parcel_coverage_threshold,
            use_abs=use_abs,
            domain=domain,
        )

    def draw_boundaries(
        self,
        scalars: str,
        boundary_name: str,
        threshold: float = 0.5,
        num_steps: int = 1,
        target_domain: Optional[Literal['vertex', 'face']] = None,
        v2f_interpolation: str = 'mode',
        copy_values_to_boundary: bool = True,
        boundary_fill: float = 1.0,
        nonboundary_fill: float = 0.0,
    ):
        overwrite = False
        if scalars == boundary_name:
            overwrite = True
            boundary_name = f'{boundary_name}:boundary'
        if (
            (scalars in self.left.point_data) or
            (scalars in self.right.point_data)
        ):
            source_domain = domain = 'vertex'
            scalars_left = self.left.point_data[scalars]
            scalars_right = self.right.point_data[scalars]
        elif (
            (scalars in self.left.cell_data) or
            (scalars in self.right.cell_data)
        ):
            assert NotImplementedError(
                'Drawing boundaries for face datasets is not yet '
                'implemented. It is possible to obtain face-valued '
                'boundaries by first drawing boundaries (before '
                'resampling vertex datasets to faces), and setting ``face`` '
                'as the target domain in the draw boundaries call.'
            )
            source_domain = domain = 'face'
            scalars_left = self.left.cell_data[scalars]
            scalars_right = self.right.cell_data[scalars]
        else:
            raise ValueError(
                f'Unable to compute boundary for {scalars}: not found '
                'in surface data.'
            )
        if target_domain is None:
            target_domain = domain
        self._hemisphere_draw_boundaries_impl(
            'left',
            scalars=scalars,
            scalar_array=scalars_left,
            boundary_name=boundary_name,
            threshold=threshold,
            num_steps=num_steps,
            source_domain=source_domain,
            target_domain=target_domain,
            domain=domain,
            overwrite=overwrite,
            copy_values_to_boundary=copy_values_to_boundary,
            boundary_fill=boundary_fill,
            nonboundary_fill=nonboundary_fill,
        )
        self._hemisphere_draw_boundaries_impl(
            'right',
            scalars=scalars,
            scalar_array=scalars_right,
            boundary_name=boundary_name,
            threshold=threshold,
            num_steps=num_steps,
            source_domain=source_domain,
            target_domain=target_domain,
            domain=domain,
            v2f_interpolation=v2f_interpolation,
            overwrite=overwrite,
            copy_values_to_boundary=copy_values_to_boundary,
            boundary_fill=boundary_fill,
            nonboundary_fill=nonboundary_fill,
        )

    def parcel_centres_of_mass(
        self,
        parcellation: str,
        projection: str,
    ):
        """
        Compute the centre of mass of each parcel in a parcellation.

        Parameters
        ----------
        parcellation : str
            Name of the parcellation dataset to use for parcellation.
        projection : str
            Name of the projection to use for vertex coordinates.

        Returns
        -------
        centres_of_mass : Tensor
            Centre of mass coordinates of each parcel.
        """
        projection_name = f'_projection_{projection}'
        return self.parcellate_vertex_dataset(
            projection_name,
            parcellation=parcellation,
        )

    def present_in_hemisphere(self, hemi: str, key: str) -> bool:
        """
        Check if a dataset is present in a hemisphere.

        Parameters
        ----------
        hemi : str
            Hemisphere to check. Must be ``'left'`` or ``'right'``.
        key : str
            Name of the dataset to check for.

        Returns
        -------
        bool
            Whether the dataset is present in the hemisphere.
        """
        return (
            key in self.__getattribute__(hemi).point_data
        ) or (
            key in self.__getattribute__(hemi).cell_data
        )

    def scalars_centre_of_mass(
        self,
        hemisphere: str,
        scalars: str,
        projection: str,
    ) -> Tensor:
        """
        Compute the centre of mass of a scalar dataset.

        Parameters
        ----------
        hemisphere : str
            Hemisphere to compute the centre of mass for. Must be ``'left'``
            or ``'right'``.
        scalars : str
            Name of the scalar dataset to compute the centre of mass for.
        projection : str
            Name of the projection to use for vertex coordinates.

        Returns
        -------
        Tensor
            Coordinates of the centre of mass.
        """
        projection_name = f'_projection_{projection}'
        if hemisphere == 'left':
            proj_data = self.left.point_data[projection_name]
            scalars_data = self.left.point_data[scalars].reshape(-1, 1)
        elif hemisphere == 'right':
            proj_data = self.right.point_data[projection_name]
            scalars_data = self.right.point_data[scalars].reshape(-1, 1)
        else:
            raise ValueError(
                f'Invalid hemisphere: {hemisphere}. '
                'Must be "left" or "right".'
            )
        num = np.nansum(proj_data * scalars_data, axis=0)
        den = np.nansum(scalars_data, axis=0)
        return num / den

    def scalars_peak(
        self,
        hemisphere: str,
        scalars: str,
        projection: str,
    ) -> Tensor:
        """
        Find the coordinates of the peak value in a scalar dataset.

        Parameters
        ----------
        hemisphere : str
            Hemisphere to compute the peak for. Must be ``'left'`` or
            ``'right'``.
        scalars : str
            Name of the scalar dataset to compute the peak for.
        projection : str
            Name of the projection to use for vertex coordinates.

        Returns
        -------
        Tensor
            Coordinates of the peak.
        """
        projection_name = f'_projection_{projection}'
        if hemisphere == 'left':
            proj_data = self.left.point_data[projection_name]
            scalars_data = self.left.point_data[scalars].reshape(-1, 1)
        elif hemisphere == 'right':
            proj_data = self.right.point_data[projection_name]
            scalars_data = self.right.point_data[scalars].reshape(-1, 1)
        else:
            raise ValueError(
                f'Invalid hemisphere: {hemisphere}. '
                'Must be "left" or "right".'
            )
        return proj_data[np.argmax(
            np.where(
                np.isnan(scalars_data),
                -np.inf,
                scalars_data,
            )
        )]

    def poles(self, hemisphere: str) -> Tensor:
        """
        Find the poles (extremal coordinates along a canonical axis) of a
        hemisphere.

        Parameters
        ----------
        hemisphere : str
            Hemisphere to compute the poles for. Must be ``'left'`` or
            ``'right'``.

        Returns
        -------
        Tensor
            Coordinates of the poles.
        """
        if hemisphere == 'left':
            pole_names = (
                'lateral',
                'posterior',
                'ventral',
                'medial',
                'anterior',
                'dorsal',
            )
        elif hemisphere == 'right':
            pole_names = (
                'medial',
                'posterior',
                'ventral',
                'lateral',
                'anterior',
                'dorsal',
            )
        coors = self.__getattribute__(hemisphere).points
        pole_index = np.concatenate(
            (
                coors.argmin(axis=0),
                coors.argmax(axis=0),
            )
        )
        poles = coors[pole_index]
        return dict(zip(pole_names, poles))

    def closest_poles(
        self,
        hemisphere: str,
        coors: Tensor,
        metric: str = 'euclidean',
        n_poles: int = 1,
    ) -> Tensor:
        """
        Find the closest pole(s) to a set of coordinates.

        Poles are defined as the extremal coordinates along a canonical axis.

        Parameters
        ----------
        hemisphere : str
            Hemisphere to compute the poles for. Must be ``'left'`` or
            ``'right'``.
        coors : Tensor
            Coordinates to find the closest pole(s) to.
        metric : str (default: Euclidean)
            Metric to use for computing distances. Must be ``'euclidean'`` or
            ``'spherical'``.
        n_poles : int (default: 1)
            Number of closest poles to return.

        Returns
        -------
        Tensor
            Names of the closest pole(s).
        """
        poles = self.poles(hemisphere)
        poles_coors = np.array(list(poles.values()))
        if metric == 'euclidean':
            pass
        elif metric == 'spherical':
            # TODO: spherical distance is implemented downstream in
            #       hypercoil but relies on JAX, which we don't want as a
            #       dependency here. We should implement it here.
            warnings.warn(
                'Spherical distance is not currently implemented. '
                'Fallback to Euclidean distance.'
            )
            # proj = self.projection
            # self.__getattribute__(hemisphere).project("sphere")
            # dists = spherical_geodesic(coors, poles_coors)
            # self.__getattribute__(hemisphere).project(proj)
        # TODO: add case for the geodesic on the manifold as implemented in
        #       PyVista/VTK.
        else:
            raise ValueError(
                f'Metric must currently be either euclidean or spherical, '
                f'but {metric} was given.'
            )
        dists = cdist(np.atleast_2d(coors), np.atleast_2d(poles_coors))
        pole_index = np.argsort(dists, axis=1)[:, :n_poles]
        pole_names = np.array(list(poles.keys()))
        # TODO: need to use tuple indexing here
        return pole_names[pole_index]

    @staticmethod
    def _hemisphere_darray_impl(data, mask, projection):
        """
        Helper function used when creating a ``CortexTriSurface`` from a
        dictionary of data arrays corresponding to projections of each
        cortical hemisphere.
        """
        surf = pv.make_tri_mesh(*data[projection])
        surf = ProjectedPolyData(surf, projection=projection)
        for key, (points, _) in data.items():
            if key != projection:
                surf.add_projection(key, points)
        mask_str = None
        if mask is not None:
            surf.point_data['__mask__'] = mask
            mask_str = '__mask__'
        return surf, mask_str

    @staticmethod
    def _hemisphere_gifti_impl(data, mask):
        """
        Helper function used when creating a ``CortexTriSurface`` from a
        dictionary of GIFTI files.
        """
        data = {
            k: (nb.load(v) if is_path_like(v) else v) for k, v in data.items()
        }
        if mask is not None:
            if is_path_like(mask):
                mask = nb.load(mask)
            mask = mask.darrays[0].data.astype(bool)
        return data, mask

    def _assign_hemisphere_vertex_data(
        self,
        name: str,
        data: Tensor,
        is_masked: bool,
        apply_mask: bool,
        null_value: Optional[float],
        hemisphere: str,
        select: Optional[Sequence[int]] = None,
        exclude: Optional[Sequence[int]] = None,
        offset: int = 0,
        coerce_to_scalar: bool = True,
        zero_prefix: int = 0,
    ) -> None:
        data = self._hemisphere_vertex_data_impl(
            data,
            is_masked,
            apply_mask,
            null_value,
            hemisphere,
        )
        if data.ndim == 2 and coerce_to_scalar:
            n_scalars = data.shape[-1]

            exclude = exclude or []
            names = []
            # TODO: Do we want to support select and exclude when we don't
            #       coerce to scalar? i.e., dropping rows/columns of a 2D
            #       array that is then loaded as a single matrix-valued
            #       point dataset.
            if select is not None and exclude is None:
                exclude = [i for i in range(n_scalars) if i not in select]

            names = [
                f'{name}{i + offset:0{zero_prefix}d}'
                for i in range(n_scalars)
                if i not in exclude
            ]
            for n, d in zip(names, data.T):
                try:
                    self.__getattribute__(
                        hemisphere
                    ).point_data.set_array(d, n)
                except ValueError as e:
                    if self.mask:
                        mask_size = self.__getattribute__(
                            hemisphere
                        ).point_data[self.mask].sum()
                        if len(d) == mask_size:
                            raise ValueError(
                                'The number of vertices in the provided data '
                                'matches the number of vertices in the mask, '
                                'but the a dimension mismatch was raised '
                                'when trying to add the data to the '
                                'hemisphere. This is likely a consequence of '
                                'failing to specify that the data are '
                                'masked. Try setting `is_masked=True` when '
                                'adding the data. Original error message:\n\n'
                                + str(e)
                            )
                    raise
        else:
            names = [name]
            try:
                self.__getattribute__(
                    hemisphere
                ).point_data.set_array(data, name)
            except ValueError as e:
                if self.mask:
                    #hemi_size = self.__getattribute__(hemisphere).n_points
                    mask_size = self.__getattribute__(
                        hemisphere
                    ).point_data[self.mask].sum()
                    if len(data) == mask_size:
                        raise ValueError(
                            'The number of vertices in the provided data '
                            'matches the number of vertices in the mask, but '
                            'the a dimension mismatch was raised when trying '
                            'to add the data to the hemisphere. This is '
                            'likely a consequence of failing to specify that '
                            'the data are masked. Try setting '
                            '`is_masked=True` when adding the data. Original '
                            'error message:\n\n' + str(e)
                        )
                raise
        return names

    def _hemisphere_vertex_data_impl(
        self,
        data: Tensor,
        is_masked: bool,
        apply_mask: bool,
        null_value: Optional[float],
        hemisphere: str,
    ) -> Tensor:
        """
        Helper function used when adding a vertex dataset to a cortical
        hemisphere.
        """
        if null_value is None:
            null_value = np.nan
        if is_masked:
            if data.ndim == 2:
                data = data.T
                init = np.full(
                    (self.n_points[hemisphere], data.shape[-1]),
                    null_value,
                )
            else:
                init = np.full(self.n_points[hemisphere], null_value)
            init[self.masks[hemisphere]] = data
            return init
        elif apply_mask:
            if data.ndim == 2:
                data = data.T
                mask = self.masks[hemisphere][..., None]
            else:
                mask = self.masks[hemisphere]
            return np.where(
                mask,
                data,
                null_value,
            )
        else:
            return data

    @staticmethod
    def _hemisphere_parcellation_impl(
        point_data: Mapping,
        parcellation: str,
        null_value: Optional[float] = 0.0,
        transpose: bool = False,
    ) -> Tensor:
        """
        Helper function used to create a parcellation matrix when parcellating
        or scattering data onto a cortical hemisphere.
        """
        parcellation = point_data[parcellation]
        if null_value is None:
            null_value = parcellation.min() - 1
        parcellation[np.isnan(parcellation)] = null_value
        parcellation = parcellation.astype(int)
        if parcellation.ndim == 1:
            parcellation = parcellation - parcellation.min()
            null_index = int(null_value - parcellation.min())
            n_parcels = parcellation.max() + 1
            parcellation = np.eye(n_parcels)[parcellation]
            parcellation = np.delete(parcellation, null_index, axis=1)
        if transpose:
            denom = parcellation.sum(axis=-1, keepdims=True)
        else:
            denom = parcellation.sum(axis=0, keepdims=True)
        denom = np.where(denom == 0, 1, denom)
        return parcellation, denom

    def _hemisphere_parcellate_impl(
        self,
        name: str,
        parcellation: str,
        hemisphere: str,
    ) -> Tensor:
        """
        Helper function used when parcellating data onto a cortical
        hemisphere.
        """
        try:
            point_data = self.point_data[hemisphere]
        except KeyError:
            return None
        parcellation, denom = self._hemisphere_parcellation_impl(
            point_data,
            parcellation=parcellation,
        )
        data = point_data[name]
        data[np.isnan(data)] = 0
        return parcellation.T @ data, denom.T

    def _hemisphere_into_parcels_impl(
        self,
        data: Tensor,
        parcellation: str,
        hemisphere: str,
    ) -> Tensor:
        """
        Helper function used when scattering data onto a cortical hemisphere.
        """
        try:
            point_data = self.point_data[hemisphere]
        except KeyError:
            return None
        parcellation, denom = self._hemisphere_parcellation_impl(
            point_data,
            parcellation=parcellation,
            transpose=True,
        )
        data[np.isnan(data)] = 0
        return (parcellation / denom) @ data[: parcellation.shape[-1]]

    def _hemisphere_resample_v2f_impl(
        self,
        name: str,
        interpolation: str,
        hemisphere: str,
    ) -> Tensor:
        faces = self.__getattribute__(
            hemisphere
        ).faces.reshape(-1, 4)[..., 1:]
        scalars = self.__getattribute__(hemisphere).point_data[name]
        scalars = scalars[faces]
        if interpolation == 'mode':
            scalars = np.where(
                scalars[..., 1] == scalars[..., 2],
                scalars[..., 1],
                scalars[..., 0],
            )
        elif interpolation == 'mean':
            scalars = scalars.mean(axis=-1)
        else:
            raise ValueError(
                f'Interpolation must be either "mode" or "mean", but '
                f'{interpolation} was given.'
            )
        return scalars

    def _hemisphere_select_active_parcels_impl(
        self,
        hemisphere: str,
        parcellation_name: str,
        parcellation: Tensor,
        activation: Tensor,
        activation_threshold: float = 2.58,
        parcel_coverage_threshold: float = 0.5,
        use_abs: bool = False,
        domain: str = 'vertex',
        null_value: float = 0.0,
    ):
        if use_abs:
            activation = np.abs(activation)
        activation = (activation > activation_threshold)
        parcellation_arg = parcellation
        parcellation = parcellation - parcellation.min()
        null_index = int(null_value - parcellation.min())
        n_parcels = parcellation.max() + 1
        parcellation = np.eye(n_parcels)[parcellation]
        parcellation = np.delete(parcellation, null_index, axis=1)
        selection = (
            (activation @ parcellation) /
            parcellation.sum(0) >
            parcel_coverage_threshold
        )
        parcellation = np.where(
            parcellation.T[selection].sum(0),
            parcellation_arg,
            np.nan,
        )
        if domain == 'vertex':
            self.__getattribute__(hemisphere).point_data[
                parcellation_name
            ] = parcellation
        else:
            self.__getattribute__(hemisphere).cell_data[
                parcellation_name
            ] = parcellation

    def _hemisphere_draw_boundaries_impl(
        self,
        hemisphere: str,
        scalars: str,
        scalar_array: Tensor,
        boundary_name: str,
        threshold: float = 0.5,
        num_steps: int = 1,
        source_domain: str = 'vertex',
        target_domain: str = 'vertex',
        domain: str = 'vertex',
        v2f_interpolation: str = 'mode',
        overwrite: bool = False,
        copy_values_to_boundary: bool = True,
        boundary_fill: float = 1.0,
        nonboundary_fill: float = 0.0,
    ):
        step = 0
        hemi_surf = self.__getattribute__(hemisphere)
        faces = hemi_surf.faces.reshape(-1, 4)[..., 1:]
        #TODO: This would probably be much more efficient if we could use an
        #      adjacency matrix to find the neighbours of each face, rather
        #      than iteratively resampling between vertex and face data.
        while (step < (num_steps * 2)) or (domain != target_domain):
            if domain == 'vertex':
                scalar_array = scalar_array[faces].astype(float)
                scalar_array = ~(
                    (
                        np.abs(scalar_array[..., 0] - scalar_array[..., 1])
                        < threshold
                    ) & (
                        np.abs(scalar_array[..., 0] - scalar_array[..., 2])
                        < threshold
                    ) & (
                        np.abs(scalar_array[..., 1] - scalar_array[..., 2])
                        < threshold
                    )
                )
                hemi_surf.cell_data[boundary_name] = scalar_array
                domain = 'face'
            else:
                scalar_array = (
                    #TODO: We're doing a lot of unnecessary resampling here.
                    #      Can we at least just resample the boundary faces?
                    pv.DataSetFilters.cell_data_to_point_data(
                        hemi_surf
                    ).point_data[boundary_name] != 0
                )
                hemi_surf.point_data[boundary_name] = scalar_array
                domain = 'vertex'
            step += 1
        dataset = (
            hemi_surf.point_data
            if domain == 'vertex'
            else hemi_surf.cell_data
        )
        if copy_values_to_boundary:
            if source_domain == target_domain:
                scalar_array = dataset[scalars]
            elif target_domain == 'vertex':
                scalar_array = pv.DataSetFilters.cell_data_to_point_data(
                    hemi_surf
                ).point_data[scalars]
            else:
                scalar_array = self._hemisphere_resample_v2f_impl(
                    name=scalars,
                    interpolation=v2f_interpolation,
                    hemisphere=hemisphere,
                )
            dataset[boundary_name] = np.where(
                dataset[boundary_name], scalar_array, nonboundary_fill
            )
        else:
            dataset[boundary_name] = np.where(
                dataset[boundary_name], boundary_fill, nonboundary_fill
            )
        if domain == 'face':
            if boundary_name in hemi_surf.point_data:
                hemi_surf.point_data.remove(boundary_name)
            if overwrite:
                hemi_surf.cell_data[scalars] = np.array(
                    dataset[boundary_name]
                )
                if boundary_name in hemi_surf.cell_data:
                    hemi_surf.cell_data.remove(boundary_name)
        elif domain == 'vertex':
            if boundary_name in hemi_surf.cell_data:
                hemi_surf.cell_data.remove(boundary_name)
            if overwrite:
                hemi_surf.point_data[scalars] = np.array(
                    dataset[boundary_name]
                )
                if boundary_name in hemi_surf.point_data:
                    hemi_surf.point_data.remove(boundary_name)

    def scalar_percentile(
        self,
        layers: Sequence[Layer],
        hemispheres: Sequence[str],
    ) -> Sequence[Layer]:
        surfs = {
            'left': getattr(self, 'left', None),
            'right': getattr(self, 'right', None),
        }
        layers_eval = []
        layers = [
            dataclasses.replace(layer, clim_percentile=True)
            if layer.clim is None
            else layer
            for layer in layers
        ]
        # layers = [
        #     dataclasses.replace(layer, alim_percentile=True)
        #     if layer.alim is None
        #     else layer
        #     for layer in layers
        # ]
        for layer in layers:
            clim = layer.clim
            alim = layer.alim
            arrays = []
            alpha_arrays = []
            eval_colour = False
            eval_alpha = False
            if layer.clim_percentile:
                eval_colour = True
            if layer.alim_percentile and isinstance(layer.alpha, str):
                eval_alpha = True
            if layer.clim_percentile or layer.alim_percentile:
                for hemisphere in hemispheres:
                    hemi_surf = surfs[hemisphere]
                    scalar_array = None
                    alpha_array = None
                    if hemi_surf is None:
                        continue
                    try:
                        if eval_colour:
                            scalar_array = hemi_surf.cell_data[layer.name]
                        if eval_alpha:
                            alpha_array = (
                                hemi_surf.cell_data[layer.alpha]
                            )
                    except (KeyError, TypeError):
                        try:
                            if eval_colour:
                                scalar_array = (
                                    hemi_surf.point_data[layer.name]
                                )
                            if eval_alpha:
                                alpha_array = (
                                    hemi_surf.point_data[layer.alpha]
                                )
                        except (KeyError, TypeError):
                            pass
                    if scalar_array is not None:
                        arrays.append(scalar_array)
                    if alpha_array is not None:
                        alpha_arrays.append(alpha_array)
                if len(arrays) > 0 and eval_colour:
                    clim = scalar_percentile(np.concatenate(arrays))
                elif not eval_colour:
                    clim = layer.clim
                else:
                    clim = (0, 1)
                if len(alpha_arrays) > 0 and eval_alpha:
                    alim = scalar_percentile(np.concatenate(alpha_arrays))
                elif not eval_alpha:
                    alim = layer.alim
                else:
                    alim = (0, 1)
            layer = dataclasses.replace(
                layer,
                clim=clim,
                alim=alim,
                clim_percentile=False,
                alim_percentile=False,
            )
            layers_eval.append(layer)
        return layers_eval


def plot_surf_f(
    plotter: pv.Plotter,
    scalar_bar_builders: Sequence[ScalarBarBuilder],
    hemispheres: Sequence[str],
    hemisphere_parameters: SubgeometryParameters,
    copy_actors: bool = False,
    *,
    surf: Optional['CortexTriSurface'] = None,
    surf_projection: str = 'pial',
    surf_color: Optional[str] = 'white',
    surf_alpha: Optional[float] = None,
    surf_style: Union[Mapping, Literal['__default__']] = '__default__',
    surf_scalars: Optional[str] = SURF_SCALARS_DEFAULT_VALUE,
    surf_scalars_boundary_color: str = 'black',
    surf_scalars_boundary_width: int = 0,
    surf_scalars_color: Any = LAYER_COLOR_DEFAULT_VALUE,
    surf_scalars_cmap: Any = SURF_SCALARS_CMAP_DEFAULT_VALUE,
    surf_scalars_cmap_negative: Any = LAYER_CMAP_NEGATIVE_DEFAULT_VALUE,
    surf_scalars_clim: Any = SURF_SCALARS_CLIM_DEFAULT_VALUE,
    surf_scalars_clim_negative: Any = LAYER_CLIM_NEGATIVE_DEFAULT_VALUE,
    surf_scalars_clim_percentile: bool = LAYER_CLIM_PERCENTILE_DEFAULT_VALUE,
    surf_scalars_alpha: Union[float, str] = LAYER_ALPHA_DEFAULT_VALUE,
    surf_scalars_amap: Union[
        callable, Tuple[float, float]
    ] = LAYER_AMAP_DEFAULT_VALUE,
    surf_scalars_amap_negative: Union[
        callable, Tuple[float, float]
    ] = LAYER_AMAP_NEGATIVE_DEFAULT_VALUE,
    surf_scalars_alim: Optional[
        Tuple[float, float]
    ] = LAYER_ALIM_DEFAULT_VALUE,
    surf_scalars_alim_negative: Optional[
        Tuple[float, float]
    ] = LAYER_ALIM_NEGATIVE_DEFAULT_VALUE,
    surf_scalars_alim_percentile: bool = LAYER_ALIM_PERCENTILE_DEFAULT_VALUE,
    surf_scalars_nan_override: Any = LAYER_NAN_OVERRIDE_DEFAULT_VALUE,
    surf_scalars_below_color: str = SURF_SCALARS_BELOW_COLOR_DEFAULT_VALUE,
    surf_scalars_layers: Union[
        Optional[Sequence[Layer]],
        Tuple[Optional[Sequence[Layer]]]
    ] = SURF_SCALARS_LAYERS_DEFAULT_VALUE,
    # Consumed but not used
    template: Optional[str] = None,
) -> Tuple[pv.Plotter, Sequence[ScalarBarBuilder]]:
    # Sometimes by construction surf_scalars is an empty tuple or list, which
    # causes problems later on. So we convert it to None if it's empty.
    if surf_scalars is not None:
        surf_scalars = None if len(surf_scalars) == 0 else surf_scalars

    if surf is not None:
        for hemisphere in hemispheres:
            hemi_surf = surf.__getattribute__(hemisphere)
            hemi_surf.project(surf_projection)
            hemi_clim = hemisphere_parameters.get(
                hemisphere,
                'surf_scalars_clim',
            )
            hemi_cmap = hemisphere_parameters.get(
                hemisphere,
                'surf_scalars_cmap',
            )
            hemi_layers = hemisphere_parameters.get(
                hemisphere,
                'surf_scalars_layers',
            )
            if hemi_layers is None:
                hemi_layers = []

            base_layers = []
            if surf_color is not None:
                base_layers += [
                    Layer(
                        name=None,
                        cmap=None,
                        clim=None,
                        cmap_negative=None,
                        color=surf_color,
                        alpha=surf_alpha,
                        below_color=None,
                    )
                ]
            if surf_scalars is not None:
                base_layers += [
                    Layer(
                        name=surf_scalars,
                        color=surf_scalars_color or surf_scalars,
                        cmap=hemi_cmap,
                        cmap_negative=surf_scalars_cmap_negative,
                        clim=hemi_clim,
                        clim_negative=surf_scalars_clim_negative,
                        clim_percentile=surf_scalars_clim_percentile,
                        alpha=surf_scalars_alpha,
                        amap=surf_scalars_amap,
                        amap_negative=surf_scalars_amap_negative,
                        alim=surf_scalars_alim,
                        alim_negative=surf_scalars_alim_negative,
                        alim_percentile=surf_scalars_alim_percentile,
                        below_color=surf_scalars_below_color,
                        nan_override=surf_scalars_nan_override,
                    )
                ]
            hemi_layers = list(base_layers) + list(hemi_layers)
            hemi_layers = surf.scalar_percentile(
                layers=hemi_layers,
                # We're hard-coding the limits to use both hemispheres here
                # so that the scalar bar range is consistent for figures that
                # simultaneously include views of single and both hemispheres.
                hemispheres=(('left', 'right')),
            )
            plotter, new_builders = hemi_surf.paint(
                plotter=plotter,
                layers=hemi_layers,
                surf_alpha=surf_alpha,
                style=surf_style,
                copy_actors=copy_actors,
            )
            if (
                surf_scalars_boundary_width > 0
                and surf_scalars is not None
                and surf_scalars not in hemi_surf.cell_data
            ):
                plotter.add_mesh(
                    hemi_surf.contour(
                        isosurfaces=range(
                            int(max(hemi_surf.point_data[surf_scalars]))
                        ),
                        scalars=surf_scalars,
                    ),
                    color=surf_scalars_boundary_color,
                    line_width=surf_scalars_boundary_width,
                )
            scalar_bar_builders = scalar_bar_builders + new_builders
    return plotter, scalar_bar_builders


def plot_surf_aux_f(
    metadata: Mapping[str, Sequence[str]],
    *,
    surf: Optional['CortexTriSurface'] = None,
    surf_projection: str = 'pial',
    surf_color: Optional[str] = 'white',
    surf_alpha: Optional[float] = None,
    surf_style: Union[Mapping, Literal['__default__']] = '__default__',
    surf_scalars: Optional[str] = SURF_SCALARS_DEFAULT_VALUE,
    surf_scalars_boundary_color: str = 'black',
    surf_scalars_boundary_width: int = 0,
    surf_scalars_color: Any = LAYER_COLOR_DEFAULT_VALUE,
    surf_scalars_cmap: Any = SURF_SCALARS_CMAP_DEFAULT_VALUE,
    surf_scalars_cmap_negative: Any = LAYER_CMAP_NEGATIVE_DEFAULT_VALUE,
    surf_scalars_clim: Any = SURF_SCALARS_CLIM_DEFAULT_VALUE,
    surf_scalars_clim_negative: Any = LAYER_CLIM_NEGATIVE_DEFAULT_VALUE,
    surf_scalars_clim_percentile: bool = LAYER_CLIM_PERCENTILE_DEFAULT_VALUE,
    surf_scalars_alpha: Union[float, str] = LAYER_ALPHA_DEFAULT_VALUE,
    surf_scalars_amap: Union[
        callable, Tuple[float, float]
    ] = LAYER_AMAP_DEFAULT_VALUE,
    surf_scalars_amap_negative: Union[
        callable, Tuple[float, float]
    ] = LAYER_AMAP_NEGATIVE_DEFAULT_VALUE,
    surf_scalars_alim: Optional[
        Tuple[float, float]
    ] = LAYER_ALIM_DEFAULT_VALUE,
    surf_scalars_alim_negative: Optional[
        Tuple[float, float]
    ] = LAYER_ALIM_NEGATIVE_DEFAULT_VALUE,
    surf_scalars_alim_percentile: bool = LAYER_ALIM_PERCENTILE_DEFAULT_VALUE,
    surf_scalars_nan_override: Any = LAYER_NAN_OVERRIDE_DEFAULT_VALUE,
    surf_scalars_below_color: str = SURF_SCALARS_BELOW_COLOR_DEFAULT_VALUE,
    surf_scalars_layers: Union[
        Optional[Sequence[Layer]],
        Tuple[Optional[Sequence[Layer]]]
    ] = SURF_SCALARS_LAYERS_DEFAULT_VALUE,
    # Consumed but not used
    template: Optional[str] = None,
) -> Mapping[str, Sequence[str]]:
    metadata['surfscalars'] = [surf_scalars or None]
    metadata['projection'] = [surf_projection or None]
    if surf_scalars_layers is not None:
        layers = surf_scalars_layers
        if len(layers) == 2 and isinstance(layers[0], (list, tuple)):
            layers = (
                layers[0]
                if metadata['hemisphere'][0] != 'right'
                else layers[1]
            )
        layers = '+'.join(layer.name for layer in layers)
        metadata['surfscalars'] = (
            [f'{metadata["surfscalars"][0]}+{layers}']
            if metadata['surfscalars'][0] is not None
            else [layers]
        )
    return metadata
