"""
q2_plots.py
---

Plots for q2_formal_features
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from tueplots import bundles

plt.rcParams.update(bundles.beamer_moml())
plt.rcParams.update(
    {
        "figure.dpi": 200,
        # "figure.figsize": (5.5324802778 / 1.5, 8.6938975 / 3*1.5),
        "font.size": 10,  # base font size
        "axes.titlesize": 14,  # title size
        "axes.labelsize": 12,  # x and y label size
        "xtick.labelsize": 10,  # x tick labels
        "ytick.labelsize": 10,  # y tick labels
        "legend.fontsize": 10,  # legend text
    }
)
# Color palette
years = [1940, 1941, 1942, 1943, 1944, 1945]
palette = dict(zip(years, sns.color_palette("tab20", len(years))))
palette[1945] = "gold"  # Override 1945


def plot_depiction_ratio_per_episode(
    df: pd.DataFrame, spearman_p: float, spearman_r: float, median: float, mean: float, std: float
):
    """Creates plot for soldier ratio per episode"""

    description = "Entwicklung der Soldatendarstellungen"

    # Aggregate per episode
    df_episode = (
        df.groupby(["episode", "year"], as_index=False)
        .agg(
            {
                "ratio_individual_depictions": "mean",
                "ratio_group_depictions": "mean",
                "ratio_collective_depictions": "mean",
            }
        )
        .sort_values("episode")
    )

    # Smooth with rolling mean
    window = 5
    for col in ["ratio_individual_depictions", "ratio_group_depictions", "ratio_collective_depictions"]:
        df_episode[col] = df_episode[col].rolling(window=window, center=True, min_periods=1).mean()

    # Figure and axes
    fig, ax = plt.subplots(figsize=(6, 4))

    text = (
        f"Spearman ρ = {spearman_r:.2f} (p = {spearman_p:.3g})\n"
        f"Mean: {mean:.2f}\n"
        f"Standard deviation: {std:.2f}"
    )

    # Stackplot
    ax.stackplot(
        df_episode["episode"],
        df_episode["ratio_individual_depictions"],
        df_episode["ratio_group_depictions"],
        df_episode["ratio_collective_depictions"],
        labels=["Individell", "Mehrere", "Kollektiv"],
        colors=["#d7191c", "white", "#2b83ba"],
        alpha=0.6,
    )

    # Regression lines
    sns.regplot(
        data=df,
        x="episode",
        y="ratio_individual_depictions",
        scatter=False,
        label="Individuelle Darstellung Trend",
        color="#d7191c",
        ax=ax,
    )
    sns.regplot(
        data=df,
        x="episode",
        y="ratio_group_depictions",
        scatter=False,
        label="Multiple  Darstellung Trend",
        color="lightgrey",
        ax=ax,
    )
    sns.regplot(
        data=df,
        x="episode",
        y="ratio_collective_depictions",
        scatter=False,
        label="Kollektive Darstellung Trend",
        color="#2b83ba",
        ax=ax,
    )

    # Vertical lines for years
    year_positions = df.groupby("year")["episode"].min()
    for ep in year_positions:
        ax.axvline(x=ep, color="lightgrey", linestyle="-", alpha=0.5)

    # Secondary x-axis for years
    secax = ax.secondary_xaxis("top")
    secax.set_xticks(year_positions.values)  # positions at first episode of each year
    secax.set_xticklabels(year_positions.index)  # show year labels
    secax.set_xlabel("Year")

    ax.text(
        0.05,
        0.95,
        text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", color="lightgrey", alpha=0.4),
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Anteil der Soldatendarstellungen")
    ax.set_title(f"{description} (n={len(df_episode)})")
    ax.set_ylim(0, 1)
    ax.set_xlim(df["episode"].min(), df["episode"].max())

    leg = ax.legend(loc="upper right", frameon=True, fontsize=9)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_alpha(0.7)

    return plt

def plot_narrative_trends_per_episode(df: pd.DataFrame, columns: list[str], labels: dict[str, str], window: int = 3):
    """
    Creates a stacked plot with trend lines for selected narrative columns.
    
    Args:
        df: DataFrame containing 'episode', 'year', and numeric dummy columns.
        columns: list of column names to plot.
        labels: dictionary mapping columns to display labels.
        window: rolling mean window for smoothing.
    """

     # Ensure we only use numeric columns
    df_numeric = df[["episode", "year"] + columns].copy()
    
    # Aggregate per episode
    df_episode = df_numeric.groupby(["episode", "year"], as_index=False).mean().sort_values("episode")
    
    # Rolling mean smoothing
    window = 3
    for col in columns:
        df_episode[col] = df_episode[col].rolling(window=window, center=True, min_periods=1).mean()
     
    # Figure
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = sns.color_palette("tab10", len(columns))
    
    ax.stackplot(
        df_episode["episode"],
        [df_episode[col] for col in columns],
        labels=[labels[col] for col in columns],
        colors=colors,
        alpha=0.6
    )
    
    # Regression trend lines
    for col, color in zip(columns, colors):
        sns.regplot(
            data=df_episode,
            x="episode",
            y=col,
            scatter=False,
            color=color,
            ax=ax
        )
    
    # Vertical lines for years
    year_positions = df.groupby("year")["episode"].min()
    for ep in year_positions:
        ax.axvline(x=ep, color="lightgrey", linestyle="-", alpha=0.5)
    
    # Secondary x-axis for years
    secax = ax.secondary_xaxis("top")
    secax.set_xticks(year_positions.values)
    secax.set_xticklabels(year_positions.index)
    secax.set_xlabel("Jahr")
    
    ax.set_xlabel("Episode")
    ax.set_ylabel("Anteil der Kapitel")
    ax.set_title(f"Narrative Framing Trends (n={len(df_episode)})")
    ax.set_xlim(df["episode"].min(), df["episode"].max())
    ax.set_ylim(0, 1)
    
    leg = ax.legend(loc="upper right", frameon=True, fontsize=9)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_alpha(0.7)
    
    plt.tight_layout()
    return plt

def plot_category_trends_per_episode(
    df: pd.DataFrame,
    columns: list[str],
    labels: dict[str, str],
    title: str,
    window: int = 3
):
    """
    General stacked area plot for any category/dummy columns per episode.
    
    Args:
        df: DataFrame containing 'episode', 'year', and numeric dummy columns.
        columns: List of column names to plot.
        labels: Dictionary mapping column names to display labels.
        title: Plot title.
        window: Rolling mean window for smoothing.
    """
    # Ensure we only use numeric columns
    df_numeric = df[["episode", "year"] + columns].copy()

    # Aggregate per episode
    df_episode = df_numeric.groupby(["episode", "year"], as_index=False).mean().sort_values("episode")

    # Apply rolling mean smoothing
    for col in columns:
        df_episode[col] = df_episode[col].rolling(window=window, center=True, min_periods=1).mean()

    # Figure and colors
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = sns.color_palette("tab10", len(columns))

    # Stackplot
    ax.stackplot(
        df_episode["episode"],
        [df_episode[col] for col in columns],
        labels=[labels[col] for col in columns],
        colors=colors,
        alpha=0.6
    )

    # Regression trend lines
    for col, color in zip(columns, colors):
        sns.regplot(
            data=df_episode,
            x="episode",
            y=col,
            scatter=False,
            color=color,
            ax=ax
        )

    # Vertical lines for years
    year_positions = df.groupby("year")["episode"].min()
    for ep in year_positions:
        ax.axvline(x=ep, color="lightgrey", linestyle="-", alpha=0.5)

    # Secondary x-axis for years
    secax = ax.secondary_xaxis("top")
    secax.set_xticks(year_positions.values)
    secax.set_xticklabels(year_positions.index)
    secax.set_xlabel("Jahr")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Anteil der Kapitel")
    ax.set_title(f"{title} (n={len(df_episode)})")
    ax.set_xlim(df["episode"].min(), df["episode"].max())
    ax.set_ylim(0, 1)

    leg = ax.legend(loc="upper right", frameon=True, fontsize=9)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_alpha(0.7)

    plt.tight_layout()
    return plt

def plot_heatmap(
    corr_df: pd.DataFrame,
    title: str,
    row_prefix: str | None = None,
    col_prefix: str | None = None,
    figsize: tuple | None = None
):
    """
    Plots a heatmap with fixed size and optionally strips prefixes from labels.

    Args:
        corr_df: DataFrame of correlations.
        title: Plot title.
        row_prefix: Optional prefix to remove from row labels.
        col_prefix: Optional prefix to remove from column labels.
        figsize: Optional figure size (width, height).
    """
    # Prepare labels
    row_labels = corr_df.index
    col_labels = corr_df.columns

    if row_prefix:
        row_labels = [r.replace(row_prefix, "") for r in row_labels]
    if col_prefix:
        col_labels = [c.replace(col_prefix, "") for c in col_labels]

    # Dynamic figure size if not provided
    if figsize is None:
        fig_width = max(8, len(col_labels) * 0.5)
        fig_height = max(6, len(row_labels) * 0.5)
        figsize = (fig_width, fig_height)

    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    sns.heatmap(
        corr_df,
        cmap="coolwarm",
        center=0,
        xticklabels=col_labels,
        yticklabels=row_labels,
        ax=ax
    )
    ax.set_title(title)
    return plt

def plot_keyword_trends(yearly_keywords, top_n=10):

    important_words = yearly_keywords.mean().nlargest(top_n).index

    fig, ax = plt.subplots(figsize=(12, 6))

    for word in important_words:
        ax.plot(yearly_keywords.index, yearly_keywords[word], label=word)

    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")

    ax.set_title("Keyword Trends in den Zusammenfassungen (1940–1945)")
    ax.set_ylabel("TF-IDF Relevanz")
    ax.set_xlabel("Jahr")

    fig.tight_layout()

    return plt

def plot_topic_profile(topic_profile: pd.DataFrame, title="Narratives Framing nach Topic"):
    """
    Plots a heatmap of dummy variable averages per topic.
    """
    fig, ax = plt.subplots(figsize=(12,6), layout="constrained")
    sns.heatmap(topic_profile, cmap="viridis", ax=ax)
    
    ax.set_title(title)
    ax.set_ylabel("Topic")
    ax.set_xlabel("Narrative")

    return plt