"""
q2_plots.py
---

Plots for q2_formal_features
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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
