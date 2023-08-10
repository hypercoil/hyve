# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests using the parcellations in the OHBM ``sparque`` poster
"""
import pytest

from hyve_examples import (
    get_schaefer400_cifti,
    get_myconnectomeWard400_nifti,
    get_mscWard400_nifti,
    get_glasser360_gifti,
    get_gordon333_gifti,
)
from hyve.flows import plotdef
from hyve.transforms import (
    surf_from_archive,
    surf_scalars_from_cifti,
    surf_scalars_from_gifti,
    surf_scalars_from_array,
    surf_scalars_from_nifti,
    parcellate_colormap,
    plot_to_image,
    save_snapshots,
)
from hyve.util import sanitise

@pytest.mark.parametrize('cmap', ['network', 'modal'])
@pytest.mark.parametrize('parcellation_name, parcellation_path', [
    ('Schaefer400', get_schaefer400_cifti()),
    ('MyConnectomeWard400', get_myconnectomeWard400_nifti()),
    ('MSCWard400', get_mscWard400_nifti()),
    ('Glasser360', get_glasser360_gifti()),
    ('Gordon333', get_gordon333_gifti()),
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

    paramstr = sanitise(parcellation_name)
    if isinstance(parcellation_path, dict):
        parcellation_path_L, parcellation_path_R = (
            parcellation_path['left'], parcellation_path['right']
        )
        filearg = {
            f'{paramstr}_gifti_left': parcellation_path_L,
            f'{paramstr}_gifti_right': parcellation_path_R,
        }
        transform = surf_scalars_from_gifti(
            parcellation_name,
            is_masked=False,
            apply_mask=True,
            allow_multihemisphere=False,
        )
    elif parcellation_path.endswith('.nii.gz'):
        filearg = {f'{paramstr}_nifti': parcellation_path}
        transform = surf_scalars_from_nifti(
            parcellation_name,
            template='fsLR',
            method='nearest',
            plot=True,
            threshold=0,
            allow_multihemisphere=False,
        )
    elif parcellation_path.endswith('.nii'):  # Not always, but here yes
        filearg = {f'{paramstr}_cifti': parcellation_path}
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
                'project-sparque_scalars-{surfscalars}_hemisphere-{hemisphere}_view-{view}_'
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
        # can't do this because it's autofilled by parcellate_colormap
        # surf_scalars_below_color='#555555',
        surf_scalars_boundary_width=3,
        **filearg,
        # We need a test for array inputs
        # **{f'{parcellation_name}_array_left': surf_data_L},
        # **{f'{parcellation_name}_array_right': surf_data_R},
    )
