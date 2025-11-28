"""
Visualization functions for medulloblastoma gene expression analysis.

This module provides interactive plotting functions using Plotly for visualizing
high-dimensional gene expression data through dimensionality reduction techniques
like UMAP. Supports both discrete categorical coloring and continuous spectrum
coloring for different analysis needs.

Key functionalities:
- UMAP visualization with discrete group coloring
- UMAP visualization with continuous score/probability coloring
- Interactive plots with hover information and customizable styling
- Support for both 2D and 3D visualizations
- Export capabilities (HTML, PNG, PDF, SVG formats)
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Union, Optional, Dict, Any

import numpy as np
import pandas as pd
import umap
import plotly.express as px
from loguru import logger

def plot_umap_binary(
    data,
    clinical,
    colors_dict,
    shapes_dict=None,
    n_components=2,
    save_fig=False,
    save_as=None,
    seed=None,
    title='UMAP',
    show=True,
    marker_size=8,
):
    """
    Plot UMAP of the data with Plotly using different colors for the different groups.

    Parameters
    ----------
    data : pandas.DataFrame
        Features as rows and samples as columns (same as in plot_umap).
    clinical : pandas.Series
        Category per sample (index must match data.columns).
    colors_dict : dict
        Mapping {group_name: color_hex_or_name}.
    shapes_dict: dict
        Mapping {group_name: shape}.
    n_components : int, optional
        2 or 3, by default 2.
    save_fig : bool, optional
        If True, save HTML/PNG/PDF/SVG, by default False.
    save_as : str or None, optional
        Base path (without extension) for saving, by default None.
    seed : int or None, optional
        Random state for UMAP, by default None.
    title : str, optional
        Plot title, by default 'UMAP'.
    show : bool, optional
        If True, display the plot, by default True.
    """

    # Check number of samples is the first dimension of data:
    if data.shape[0] != clinical.shape[0]:
        data = data.T
        if data.shape[0] != clinical.shape[0]:
            raise ValueError("Data and clinical metadata must have the same number of samples")


    if n_components not in (2, 3):
        raise ValueError("n_components must be 2 or 3 for plot_umap_plotly")

    today = datetime.now().strftime("%Y%m%d")
    if save_as is None:
        suffix = "UMAP" if n_components == 2 else "3D_UMAP"
        save_as = f"{today}_{suffix}"

    if seed is not None:
        umap_ = umap.UMAP(n_components=n_components, random_state=seed)
    else:
        umap_ = umap.UMAP(n_components=n_components)

    # data: samples x features
    X_umap = umap_.fit_transform(data)
    print("X_umap.shape", X_umap.shape)

    # Determine color and shape series from clinical
    if isinstance(clinical, pd.DataFrame):
        color_col = clinical.columns[0]
        color_series = clinical[color_col]
        # use second column for shapes if provided and shapes_dict is given
        if shapes_dict is not None and clinical.shape[1] >= 2:
            shape_col = clinical.columns[1]
            shape_series = clinical[shape_col]
        else:
            shape_series = None
    elif isinstance(clinical, pd.Series):
        color_series = clinical
        shape_series = None
    else:
        raise ValueError("clinical must be a pandas Series or DataFrame")
    print("color_series.shape", color_series.shape)

    # Build plotting DataFrame
    all_patients = data.index.tolist()
    print("len(all_patients)", len(all_patients))
    print("color_series.loc[all_patients].values.shape", color_series.loc[all_patients].values.shape)
    df_plot = pd.DataFrame(
        {
            "sample": all_patients,
            "group": color_series.loc[all_patients].values,
            "UMAP_1": X_umap[:, 0],
            "UMAP_2": X_umap[:, 1],
        }
    )
    if n_components == 3:
        df_plot["UMAP_3"] = X_umap[:, 2]

    # Attach shape column if available
    if shape_series is not None:
        df_plot["shape"] = shape_series.loc[all_patients].values

    # Build color sequence in the order of unique groups
    unique_groups = df_plot["group"].unique()
    color_sequence = [colors_dict[g] for g in unique_groups]

    # Prepare symbol mapping if shapes are used
    symbol_map = None
    if "shape" in df_plot.columns and shapes_dict is not None:
        # convert common Matplotlib markers to Plotly symbols if needed
        matplot_to_plotly = {
            'o': 'circle', 's': 'square', '^': 'triangle-up', 'v': 'triangle-down',
            'D': 'diamond', 'd': 'diamond-wide', 'X': 'x', 'x': 'x', '*': 'star',
            '+': 'cross', 'p': 'pentagon', 'h': 'hexagon', 'H': 'hexagon2'
        }
        unique_shapes = df_plot["shape"].unique()
        symbol_map = {}
        for sh in unique_shapes:
            # get marker definition from shapes_dict; fallback to the value itself
            marker = shapes_dict.get(sh, shapes_dict.get(str(sh), sh))
            # translate matplotlib marker codes to plotly symbol names when possible
            symbol = matplot_to_plotly.get(marker, marker)
            symbol_map[sh] = symbol

    # Create plotly figure with optional symbols
    if n_components == 2:
        fig = px.scatter(
            df_plot,
            x="UMAP_1",
            y="UMAP_2",
            color="group",
            color_discrete_sequence=color_sequence,
            hover_name="sample",
            template="simple_white",
            width=800,
            height=800,
            symbol="shape" if "shape" in df_plot.columns and symbol_map is not None else None,
            symbol_map=symbol_map if symbol_map is not None else None,
        )
        fig.update_layout(
            title=title,
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
        )
        fig.update_traces(marker=dict(size=marker_size))
    else:
        fig = px.scatter_3d(
            df_plot,
            x="UMAP_1",
            y="UMAP_2",
            z="UMAP_3",
            color="group",
            color_discrete_sequence=color_sequence,
            hover_name="sample",
            template="simple_white",
            width=800,
            height=800,
            symbol="shape" if "shape" in df_plot.columns and symbol_map is not None else None,
            symbol_map=symbol_map if symbol_map is not None else None,
        )
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="UMAP 1",
                yaxis_title="UMAP 2",
                zaxis_title="UMAP 3",
            ),
        )
        fig.update_traces(marker=dict(size=marker_size))
    # Optional saving
    if save_fig:
        base_dir = os.path.dirname(save_as)
        if base_dir and not os.path.exists(base_dir):
            os.makedirs(base_dir, exist_ok=True)
        # Save as HTML
        fig.write_html(f"{save_as}.html")
        # Save static images
        for extension in ['png', 'pdf', 'svg']:
            print(f"Saved UMAP plotly figure to: {save_as}.{extension}")
            fig.write_image(f"{save_as}.{extension}", scale=2)
    if show:
        fig.show()

def plot_umap_spectrum(
    data,
    clinical,
    colormap='viridis',
    shapes_dict=None,
    n_components=2,
    save_fig=False,
    save_as=None,
    seed=None,
    title='UMAP',
    show=True,
    marker_size=8,
    color_range=None,
    colorbar_title=None,
):
    """
    Plot UMAP of the data with Plotly using continuous colormap for numeric values.

    Parameters
    ----------
    data : pandas.DataFrame
        Features as rows and samples as columns (same as in plot_umap).
    clinical : pandas.Series
        Numeric values (0-1) per sample (index must match data.columns).
    colormap : str, optional
        Plotly colorscale name (e.g., 'viridis', 'plasma', 'inferno', 'magma',
        'turbo', 'rainbow', 'jet', 'hot', 'cool', 'blues', 'reds'), by default 'viridis'.
    shapes_dict: dict
        Mapping {group_name: shape}. Only used if clinical is DataFrame with 2+ columns.
    n_components : int, optional
        2 or 3, by default 2.
    save_fig : bool, optional
        If True, save HTML/PNG/PDF/SVG, by default False.
    save_as : str or None, optional
        Base path (without extension) for saving, by default None.
    seed : int or None, optional
        Random state for UMAP, by default None.
    title : str, optional
        Plot title, by default 'UMAP'.
    show : bool, optional
        If True, display the plot, by default True.
    marker_size : int, optional
        Marker size, by default 8.
    color_range : tuple or None, optional
        Range for colorscale (min, max). If None, uses data range.
    colorbar_title : str or None, optional
        Title for colorbar. If None, uses column name or 'Value'.
    """

    # Check number of samples is the first dimension of data:
    if data.shape[0] != clinical.shape[0]:
        data = data.T
        if data.shape[0] != clinical.shape[0]:
            raise ValueError("Data and clinical metadata must have the same number of samples")

    if n_components not in (2, 3):
        raise ValueError("n_components must be 2 or 3 for plot_umap_spectrum")

    today = datetime.now().strftime("%Y%m%d")
    if save_as is None:
        suffix = "UMAP_spectrum" if n_components == 2 else "3D_UMAP_spectrum"
        save_as = f"{today}_{suffix}"

    if seed is not None:
        umap_ = umap.UMAP(n_components=n_components, random_state=seed)
    else:
        umap_ = umap.UMAP(n_components=n_components)

    # data: samples x features
    X_umap = umap_.fit_transform(data)
    print("X_umap.shape", X_umap.shape)

    # Determine color and shape series from clinical
    if isinstance(clinical, pd.DataFrame):
        color_col = clinical.columns[0]
        color_series = clinical[color_col]
        # use second column for shapes if provided and shapes_dict is given
        if shapes_dict is not None and clinical.shape[1] >= 2:
            shape_col = clinical.columns[1]
            shape_series = clinical[shape_col]
        else:
            shape_series = None
    elif isinstance(clinical, pd.Series):
        color_series = clinical
        shape_series = None
    else:
        raise ValueError("clinical must be a pandas Series or DataFrame")

    print("color_series.shape", color_series.shape)
    print(f"color_series range: [{color_series.min():.3f}, {color_series.max():.3f}]")

    # Validate that color_series contains numeric values
    if not pd.api.types.is_numeric_dtype(color_series):
        raise ValueError("For continuous coloring, clinical data must contain numeric values")

    # Build plotting DataFrame
    all_patients = data.index.tolist()
    print("len(all_patients)", len(all_patients))

    df_plot = pd.DataFrame(
        {
            "sample": all_patients,
            "color_value": color_series.loc[all_patients].values,
            "UMAP_1": X_umap[:, 0],
            "UMAP_2": X_umap[:, 1],
        }
    )
    if n_components == 3:
        df_plot["UMAP_3"] = X_umap[:, 2]

    # Attach shape column if available
    if shape_series is not None:
        df_plot["shape"] = shape_series.loc[all_patients].values

    # Set up color range
    if color_range is None:
        color_range = [df_plot["color_value"].min(), df_plot["color_value"].max()]

    # Set colorbar title
    if colorbar_title is None:
        if isinstance(clinical, pd.Series):
            colorbar_title = clinical.name if clinical.name else "Value"
        else:
            colorbar_title = color_series.name if color_series.name else "Value"

    # Prepare symbol mapping if shapes are used
    symbol_map = None
    if "shape" in df_plot.columns and shapes_dict is not None:
        # convert common Matplotlib markers to Plotly symbols if needed
        matplot_to_plotly = {
            'o': 'circle', 's': 'square', '^': 'triangle-up', 'v': 'triangle-down',
            'D': 'diamond', 'd': 'diamond-wide', 'X': 'x', 'x': 'x', '*': 'star',
            '+': 'cross', 'p': 'pentagon', 'h': 'hexagon', 'H': 'hexagon2'
        }
        unique_shapes = df_plot["shape"].unique()
        symbol_map = {}
        for sh in unique_shapes:
            # get marker definition from shapes_dict; fallback to the value itself
            marker = shapes_dict.get(sh, shapes_dict.get(str(sh), sh))
            # translate matplotlib marker codes to plotly symbol names when possible
            symbol = matplot_to_plotly.get(marker, marker)
            symbol_map[sh] = symbol

    # Create plotly figure with continuous color scale
    if n_components == 2:
        fig = px.scatter(
            df_plot,
            x="UMAP_1",
            y="UMAP_2",
            color="color_value",
            color_continuous_scale=colormap,
            hover_name="sample",
            hover_data={"color_value": ":.3f"},
            template="simple_white",
            width=800,
            height=800,
            range_color=color_range,
            symbol="shape" if "shape" in df_plot.columns and symbol_map is not None else None,
            symbol_map=symbol_map if symbol_map is not None else None,
        )
        fig.update_layout(
            title=title,
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
            coloraxis_colorbar=dict(title=colorbar_title)
        )
        fig.update_traces(marker=dict(size=marker_size))
    else:
        fig = px.scatter_3d(
            df_plot,
            x="UMAP_1",
            y="UMAP_2",
            z="UMAP_3",
            color="color_value",
            color_continuous_scale=colormap,
            hover_name="sample",
            hover_data={"color_value": ":.3f"},
            template="simple_white",
            width=800,
            height=800,
            range_color=color_range,
            symbol="shape" if "shape" in df_plot.columns and symbol_map is not None else None,
            symbol_map=symbol_map if symbol_map is not None else None,
        )
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="UMAP 1",
                yaxis_title="UMAP 2",
                zaxis_title="UMAP 3",
            ),
            coloraxis_colorbar=dict(title=colorbar_title)
        )
        fig.update_traces(marker=dict(size=marker_size))

    # Optional saving
    if save_fig:
        base_dir = os.path.dirname(save_as)
        if base_dir and not os.path.exists(base_dir):
            os.makedirs(base_dir, exist_ok=True)
        # Save as HTML
        fig.write_html(f"{save_as}.html")
        # Save static images
        for extension in ['png', 'pdf', 'svg']:
            print(f"Saved UMAP spectrum figure to: {save_as}.{extension}")
            fig.write_image(f"{save_as}.{extension}", scale=2)
    if show:
        fig.show()
