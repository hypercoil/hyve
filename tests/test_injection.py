# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for all views on a brain surface
"""
import pytest
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import seaborn as sns
import templateflow.api as tflow

from hyve_examples import get_pain_thresh_nifti, get_svg_cuboid
from hyve.flows import plotdef
from hyve.layout import Cell
from hyve.transforms import (
    surf_from_archive,
    surf_scalars_from_nifti,
    add_surface_overlay,
    save_figure,
    plot_to_image,
    pyplot_element,
    svg_element,
    text_element,
)

COMMON_PARAMS = dict(
    load_mask=True,
    hemisphere=['left', 'right', None],
    views={
        'left': ('medial', 'lateral'),
        'right': ('medial', 'lateral'),
        'both': ('dorsal', 'ventral', 'anterior', 'posterior'),
    },
    output_dir='/tmp',
    window_size=(600, 500),
)


def get_annotations():
    return {
        0: dict(
            hemisphere='left',
            view='lateral',
        ),
        1: dict(
            hemisphere='left',
            view='medial',
        ),
        2: dict(view='dorsal'),
        3: dict(view='anterior'),
        4: dict(
            hemisphere='right',
            view='lateral',
        ),
        5: dict(
            hemisphere='right',
            view='medial',
        ),
        6: dict(view='ventral'),
        7: dict(view='posterior'),
        8: {},
        9: {},
    }


@pytest.mark.parametrize('projection', ['pial', 'veryinflated'])
def test_svg_injection(projection):
    layout = Cell() | Cell() << (1 / 2)
    layout = layout * (Cell() / Cell() / Cell() / Cell() << (1 / 4))
    layout = layout | (Cell() / Cell() << (1 / 2)) << (1 / 2)
    annotations = get_annotations()
    annotations[8]['elements'] = ['title', 'scalar_bar']
    annotations[9]['elements'] = ['cuboid']
    layout = layout.annotate(annotations)

    if projection == 'pial':
        template = 'fsaverage'
    else:
        template = 'fsLR'
    pain = get_pain_thresh_nifti()
    plot_f = plotdef(
        surf_from_archive(),
        add_surface_overlay(
            'pain',
            surf_scalars_from_nifti(
                'pain', template=template, plot=True
            ),
        ),
        svg_element(
            name='cuboid',
            src_file=get_svg_cuboid(),
            height=262,
            width=223,
        ),
        text_element(
            name='title',
            content='pain',
            bounding_box_height=192,
            font_size_multiplier=0.25,
            font_color='#cccccc',
            priority=-1,
        ),
        plot_to_image(),
        save_figure(
            layout_kernel=layout,
            padding=10,
            canvas_size=(1800, 1500),
            canvas_color=(0, 0, 0),
            fname_spec=f'scalars-pain_projection-{projection}_desc-cuboid_page-{{page}}',
            scalar_bar_action='collect',
        ),
    )
    plot_f(
        template=template,
        pain_nifti=pain,
        pain_clim=(2, 8),
        pain_below_color=(0, 0, 0, 0),
        pain_scalar_bar_style={
            'name': 'z-stat',
            'orientation': 'h',
            'lim_fontsize_multiplier': 0.6,
            'name_fontsize_multiplier': 0.5,
        },
        pain_cmap='inferno',
        surf_projection=(projection,),
        **COMMON_PARAMS,
    )


def test_pyplot_injection():
    layout = Cell() | Cell() << (1 / 2)
    layout = layout * (Cell() / Cell() / Cell() / Cell() << (1 / 4))
    layout = layout | (Cell() / Cell() << (1 / 2)) << (1 / 2)
    layout = layout % ((
        Cell() | Cell() << (1 / 2)) / (
        Cell() | Cell() << (1 / 2)) << (1 / 2)) << 9
    annotations = get_annotations()
    annotations[8]['elements'] = ['title', 'scalar_bar']
    annotations[9]['elements'] = ['pyplot0']
    annotations[10] = {'elements': ['pyplot1']}
    annotations[11] = {'elements': ['pyplot2']}
    annotations[12] = {'elements': ['pyplot3']}
    layout = layout.annotate(annotations)

    # https://gist.github.com/mwaskom/7be0963cc57f6c89f7b2
    plt.style.use("dark_background")
    sns.set(style="ticks", context="talk")
    def pyplot0_f(figsize):
        fig, ax = plt.subplots(figsize=figsize)
        # https://seaborn.pydata.org/examples/layered_bivariate_plot.html
        n = 10000
        mean = [0, 0]
        cov = [(2, .4), (.4, .2)]
        rng = np.random.RandomState(0)
        x, y = rng.multivariate_normal(mean, cov, n).T
        sns.scatterplot(x=x, y=y, s=5, color=".15")
        sns.histplot(x=x, y=y, bins=50, pthresh=.1, cmap="mako")
        sns.kdeplot(x=x, y=y, levels=5, color="w", linewidths=1)
        return fig
    def pyplot1_f(figsize):
        height = min(figsize)
        aspect = figsize[0] / figsize[1]
        # https://seaborn.pydata.org/examples/multiple_conditional_kde.html
        sns.set_theme(style="white")
        diamonds = sns.load_dataset("diamonds")
        fig = sns.displot(
            data=diamonds,
            x="carat", hue="cut", kind="kde",
            multiple="fill", clip=(0, None),
            palette="ch:rot=-.25,hue=1,light=.75",
            height=height, aspect=aspect,
        )
        return fig
    def pyplot2_f(figsize):
        height = min(figsize)
        # https://seaborn.pydata.org/examples/smooth_bivariate_kde.html
        sns.set_theme()
        df = sns.load_dataset("penguins")
        g = sns.JointGrid(
            data=df, x="body_mass_g", y="bill_depth_mm",
            space=0, height=height,
        )
        g.plot_joint(sns.kdeplot,
                    fill=True, clip=((2200, 6800), (10, 25)),
                    thresh=0, levels=100, cmap="rocket")
        g.plot_marginals(sns.histplot, color="#03051A", alpha=1, bins=25)
        return g
    def pyplot3_f(figsize):
        height = figsize[1]
        aspect = figsize[0] / figsize[1]
        # https://seaborn.pydata.org/examples/strip_regplot.html
        sns.set_theme()
        mpg = sns.load_dataset("mpg")
        fig = sns.catplot(
            data=mpg, x="cylinders", y="acceleration", hue="weight",
            native_scale=True, zorder=1, height=height, aspect=aspect,
        )
        sns.regplot(
            data=mpg, x="cylinders", y="acceleration",
            scatter=False, truncate=False, order=2, color=".2",
        )
        fig.tight_layout()
        return fig

    pain = get_pain_thresh_nifti()
    plot_f = plotdef(
        surf_from_archive(),
        add_surface_overlay(
            'pain',
            surf_scalars_from_nifti(
                'pain', template='fsLR', plot=True
            ),
        ),
        pyplot_element(
            name='pyplot0',
            plotter=pyplot0_f,
            figsize=(6, 5),
        ),
        pyplot_element(
            name='pyplot1',
            plotter=pyplot1_f,
            figsize=(6, 5),
        ),
        pyplot_element(
            name='pyplot2',
            plotter=pyplot2_f,
            figsize=(6, 5),
        ),
        pyplot_element(
            name='pyplot3',
            plotter=pyplot3_f,
            figsize=(6, 5),
        ),
        text_element(
            name='title',
            content='pain',
            bounding_box_height=192,
            font_size_multiplier=0.25,
            font_color='#cccccc',
            priority=-1,
        ),
        plot_to_image(),
        save_figure(
            layout_kernel=layout,
            padding=10,
            canvas_size=(1800, 1500),
            canvas_color=(0, 0, 0),
            fname_spec=f'scalars-pain_desc-pyplot_page-{{page}}',
            scalar_bar_action='collect',
        ),
    )
    plot_f(
        template='fsLR',
        pain_nifti=pain,
        pain_clim=(2, 8),
        pain_below_color=(0, 0, 0, 0),
        pain_scalar_bar_style={
            'name': 'z-stat',
            'orientation': 'h',
            'lim_fontsize_multiplier': 0.6,
            'name_fontsize_multiplier': 0.5,
        },
        pain_cmap='inferno',
        surf_projection=('veryinflated',),
        theme=pv.themes.DarkTheme(),
        **COMMON_PARAMS,
    )
