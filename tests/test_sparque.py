# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests using the parcellations in the OHBM ``sparque`` poster
"""
import pytest

from hyve.flows import plotdef
from hyve.transforms import (
    surf_from_archive,
    surf_scalars_from_cifti,
    surf_scalars_from_gifti,
    surf_scalars_from_array,
    surf_scalars_from_nifti,
    parcellate_colormap,
    vertex_to_face,
    plot_to_image,
    save_snapshots,
)

@pytest.mark.parametrize('cmap', ['network', 'modal'])
@pytest.mark.parametrize('parcellation_name, parcellation_path', [
    ('Schaefer400', '/Users/rastkociric/Downloads/desc-schaefer_res-0400_atlas.nii'),
    ('MyConnectomeWard400', '/Users/rastkociric/Downloads/myconnectome_ward400_parcellation.nii.gz'),
    ('MSCWard400', '/Users/rastkociric/Downloads/MSC_ward400_parcellation.nii.gz'),
    ('Glasser360', (
        '/Users/rastkociric/Downloads/Glasser_2016.32k.L.label.gii',
        '/Users/rastkociric/Downloads/Glasser_2016.32k.R.label.gii',
    ),),
    ('Gordon333', (
        '/Users/rastkociric/Downloads/Gordon.32k.L.label.gii',
        '/Users/rastkociric/Downloads/Gordon.32k.R.label.gii',
    )),
])
def test_sparque(parcellation_name, parcellation_path, cmap):

    # import lytemaps
    # from nilearn import datasets, surface
    # fsaverage = lytemaps.datasets.fetch_fsaverage()
    # print(fsaverage['pial'].L)
    # surf_data_L = surface.vol_to_surf(
    #     parcellation_path,
    #     surf_mesh=str(fsaverage["pial"].L),
    #     #inner_mesh=fsaverage["white_left"],
    #     interpolation="nearest",
    # )
    # surf_data_R = surface.vol_to_surf(
    #     parcellation_path,
    #     surf_mesh=str(fsaverage["pial"].R),
    #     #inner_mesh=fsaverage["white_left"],
    #     interpolation="nearest",
    # )
    # print(surf_data_L.shape, surf_data_R.shape)

    if isinstance(parcellation_path, tuple):
        parcellation_path_L, parcellation_path_R = parcellation_path
        filearg = {
            f'{parcellation_name}_gifti_left': parcellation_path_L,
            f'{parcellation_name}_gifti_right': parcellation_path_R,
        }
        transform = surf_scalars_from_gifti(
            parcellation_name,
            is_masked=False,
            apply_mask=True,
            allow_multihemisphere=False,
        )
    elif parcellation_path.endswith('.nii.gz'):
        filearg = {f'{parcellation_name}_nifti': parcellation_path}
        transform = surf_scalars_from_nifti(
            parcellation_name,
            template='fsLR',
            method='nearest',
            plot=True,
            threshold=0,
            allow_multihemisphere=False,
        )
    elif parcellation_path.endswith('.nii'):
        # Not always, but here yes
        filearg = {f'{parcellation_name}_cifti': parcellation_path}
        transform = surf_scalars_from_cifti(
            parcellation_name,
            plot=True,
            allow_multihemisphere=False,
        )
        

    plot_f = plotdef(
        surf_from_archive(),
        transform,
        parcellate_colormap(cmap, parcellation_name),
        #vertex_to_face(parcellation_name),
        plot_to_image(),
        save_snapshots(
            fname_spec=(
                'project-sparque_scalars-{scalars}_hemisphere-{hemisphere}_view-{view}_'
                f'cmap-{cmap}'
            ),
        ),
    )
    plot_f(
        template='fsLR',
        load_mask=True,
        surf_projection=('veryinflated',),
        hemisphere=['left', 'right'],
        views=[
            'lateral', 'medial', 'ventral', 'dorsal', 'anterior', 'posterior',
        ],
        output_dir='/tmp',
        surf_scalars_boundary_color='black',
        surf_scalars_below_color='#555555',
        surf_scalars_boundary_width=3,
        **filearg,
        # We need a test for array inputs
        # **{f'{parcellation_name}_array_left': surf_data_L},
        # **{f'{parcellation_name}_array_right': surf_data_R},
    )
