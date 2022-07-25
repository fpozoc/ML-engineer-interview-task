#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" src/visualization/figures.py

This file contains the code for the figures.

Classes and functions:
"""

from __future__ import absolute_import, division, print_function

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

__author__ = "Fernando Pozo"
__copyright__ = "Copyright 2022"
__license__ = "GNU General Public License"
__version__ = "0.0.1"
__maintainer__ = "Fernando Pozo"
__email__ = "fpozoc@gmx.com"
__status__ = "Development"


def correlation_heatmap(df:pd.DataFrame, title:str=None, savefig_path:bool=False, annotation:str=None) -> None:
    """Plots a correlation heatmap.

    Args:
        df (pd.DataFrame): Dataframe to be plotted.
        title (str, optional): Title of the figure. Defaults to None.
        savefig_path (bool, optional): If True, the figure is saved. Defaults to False.
        annotation (str, optional): Annotation to be added to the figure. Defaults to None.
    
    Returns:
        None
    """
    plt.figure(figsize=(20,15))
    plt.title(title)
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    data = df.T.sort_index(ascending=True).T.corr(method ='pearson')
    sns.heatmap(
            data=data.round(1),  # our correlation matrix
            linewidths=0.3,  # the width of lines separating the matrix squares
            square=True,   # enforce 1:1 ratios among correlation cells
            cmap=cmap,  # use the color map we defined above
            vmax=1,  # define the max of our correlation scale
            vmin=-1, # define the min of our correlation scale
            center=0,  # The value at which the color map is centered about (white)
            cbar_kws={"shrink": .75},  # shrink the scale a bit
            annot=True
        )
    plt.ylim(0, data.shape[0])
    plt.yticks(rotation=0)  
    if savefig_path == True:
       plt.savefig(savefig_path, bbox_inches='tight')
    return plt.show()

def plot_histogram(values:list, bins:int) -> None:
    """Plot an histogram using Matplotlib.

    Args:
        values (list): List of values to be plotted.
        bins (int): Number of bins.

    Returns:
        None
    """
    plt.figure(figsize=(10,5))
    plt.hist(values, density=True, bins=bins)  # density=False would make counts
    plt.ylabel('Density')
    plt.xlabel('Data');