#!/usr/bin/env python3
"""
Visualization script for optimization results.
Allows plotting latency and energy against any parameter in the result CSV.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from pathlib import Path
import numpy as np


def load_results(csv_path):
    """Load results from CSV file."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} data points from {csv_path}")
    print(f"Available columns: {list(df.columns)}")
    return df


def plot_metrics_vs_parameter(df, parameter, output_dir=None, show=True):
    """
    Create plots of latency and energy vs a given parameter.

    Args:
        df: DataFrame with results
        parameter: Column name to plot against
        output_dir: Directory to save plots (optional)
        show: Whether to display plots
    """
    if parameter not in df.columns:
        print(f"Error: Parameter '{parameter}' not found in data.")
        print(f"Available parameters: {list(df.columns)}")
        return

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Color by pareto rank if available
    if "pareto_rank" in df.columns:
        colors = df["pareto_rank"]
        cmap = "viridis"
    else:
        colors = "blue"
        cmap = None

    # Plot latency vs parameter
    scatter1 = axes[0].scatter(
        df[parameter], df["latency"], c=colors, cmap=cmap, alpha=0.6, s=50
    )
    axes[0].set_xlabel(parameter, fontsize=12)
    axes[0].set_ylabel("Latency", fontsize=12)
    axes[0].set_title(f"Latency vs {parameter}", fontsize=14, fontweight="bold")
    axes[0].grid(True, alpha=0.3)

    # Plot energy vs parameter
    scatter2 = axes[1].scatter(
        df[parameter], df["energy"], c=colors, cmap=cmap, alpha=0.6, s=50
    )
    axes[1].set_xlabel(parameter, fontsize=12)
    axes[1].set_ylabel("Energy", fontsize=12)
    axes[1].set_title(f"Energy vs {parameter}", fontsize=14, fontweight="bold")
    axes[1].grid(True, alpha=0.3)

    # Add colorbars if using pareto rank
    if "pareto_rank" in df.columns:
        cbar1 = plt.colorbar(scatter1, ax=axes[0])
        cbar1.set_label("Pareto Rank", fontsize=10)
        cbar2 = plt.colorbar(scatter2, ax=axes[1])
        cbar2.set_label("Pareto Rank", fontsize=10)

    plt.tight_layout()

    # Save if output directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"metrics_vs_{parameter}.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_pareto_front(df, output_dir=None, show=True):
    """Plot the Pareto front in latency-energy space."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Separate Pareto optimal points (rank 0) from others
    if "pareto_rank" in df.columns:
        pareto_optimal = df[df["pareto_rank"] == 0]
        other_points = df[df["pareto_rank"] > 0]

        # Plot non-Pareto points
        ax.scatter(
            other_points["latency"],
            other_points["energy"],
            c="lightgray",
            alpha=0.5,
            s=30,
            label="Non-Pareto",
        )

        # Plot Pareto front
        ax.scatter(
            pareto_optimal["latency"],
            pareto_optimal["energy"],
            c="red",
            s=100,
            marker="*",
            label="Pareto Front",
            edgecolors="darkred",
            linewidths=1.5,
            zorder=5,
        )

        # Connect Pareto points
        pareto_sorted = pareto_optimal.sort_values("latency")
        ax.plot(
            pareto_sorted["latency"],
            pareto_sorted["energy"],
            "r--",
            alpha=0.5,
            linewidth=1.5,
            zorder=4,
        )
    else:
        ax.scatter(df["latency"], df["energy"], c="blue", alpha=0.6, s=50)

    ax.set_xlabel("Latency", fontsize=12)
    ax.set_ylabel("Energy", fontsize=12)
    ax.set_title("Pareto Front: Energy vs Latency", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "pareto_front.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved Pareto front to {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_correlation_heatmap(df, output_dir=None, show=True):
    """Plot correlation heatmap of all numeric parameters."""
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()

    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        ax=ax,
        cbar_kws={"label": "Correlation"},
    )
    ax.set_title("Parameter Correlation Heatmap", fontsize=14, fontweight="bold")

    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "correlation_heatmap.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved correlation heatmap to {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_all_parameters(df, output_dir=None, show=False):
    """Create plots for latency and energy against all parameters."""
    # Exclude non-parameter columns
    exclude_cols = [
        "latency",
        "energy",
        "pareto_rank",
        "feasible",
        "g_compute",
        "g_memory",
        "g_area",
        "cv",
    ]

    parameters = [col for col in df.columns if col not in exclude_cols]

    print(f"\nGenerating plots for {len(parameters)} parameters...")
    for param in parameters:
        print(f"  - Plotting {param}")
        plot_metrics_vs_parameter(df, param, output_dir, show=False)

    print(f"\nAll plots saved to {output_dir}")


def interactive_mode(df, output_dir=None):
    """Interactive mode to select parameters to plot."""
    exclude_cols = ["latency", "energy"]
    available_params = [col for col in df.columns if col not in exclude_cols]

    print("\n" + "=" * 60)
    print("INTERACTIVE VISUALIZATION MODE")
    print("=" * 60)
    print("\nAvailable parameters:")
    for i, param in enumerate(available_params, 1):
        print(f"  {i:2d}. {param}")

    print("\nCommands:")
    print("  - Enter a number to plot that parameter")
    print("  - Enter 'all' to plot all parameters")
    print("  - Enter 'pareto' to show Pareto front")
    print("  - Enter 'heatmap' to show correlation heatmap")
    print("  - Enter 'quit' or 'q' to exit")
    print("=" * 60)

    while True:
        try:
            choice = input("\nYour choice: ").strip().lower()

            if choice in ["quit", "q", "exit"]:
                print("Exiting...")
                break

            elif choice == "all":
                plot_all_parameters(df, output_dir, show=False)
                print("All plots generated!")

            elif choice == "pareto":
                plot_pareto_front(df, output_dir, show=True)

            elif choice == "heatmap":
                plot_correlation_heatmap(df, output_dir, show=True)

            elif choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(available_params):
                    param = available_params[idx]
                    plot_metrics_vs_parameter(df, param, output_dir, show=True)
                else:
                    print(f"Invalid number. Please enter 1-{len(available_params)}")

            else:
                print("Invalid command. Try again.")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize optimization results: latency and energy vs parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python visualize_results.py results.csv
  
  # Plot against specific parameter
  python visualize_results.py results.csv --parameter sram_size
  
  # Plot all parameters
  python visualize_results.py results.csv --all
  
  # Show Pareto front
  python visualize_results.py results.csv --pareto
  
  # Save outputs to directory
  python visualize_results.py results.csv --all --output plots/
        """,
    )

    parser.add_argument("csv_file", type=str, help="Path to CSV file with results")
    parser.add_argument(
        "--parameter",
        "-p",
        type=str,
        help="Parameter to plot against (e.g., sram_size)",
    )
    parser.add_argument(
        "--all", "-a", action="store_true", help="Generate plots for all parameters"
    )
    parser.add_argument("--pareto", action="store_true", help="Show Pareto front plot")
    parser.add_argument(
        "--heatmap", action="store_true", help="Show correlation heatmap"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output directory for saved plots",
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true", help="Run in interactive mode"
    )
    parser.add_argument(
        "--no-show", action="store_true", help="Do not display plots (only save them)"
    )

    args = parser.parse_args()

    # Load data
    if not os.path.exists(args.csv_file):
        print(f"Error: File '{args.csv_file}' not found.")
        return

    df = load_results(args.csv_file)

    # Set default output directory if saving plots
    if args.output is None and (args.all or args.no_show):
        csv_dir = os.path.dirname(args.csv_file)
        args.output = os.path.join(csv_dir, "plots")

    show_plots = not args.no_show

    # Execute based on arguments
    if args.interactive or (
        not args.parameter and not args.all and not args.pareto and not args.heatmap
    ):
        # Default to interactive mode if no specific action specified
        interactive_mode(df, args.output)

    else:
        if args.all:
            plot_all_parameters(df, args.output, show=show_plots)

        if args.parameter:
            plot_metrics_vs_parameter(df, args.parameter, args.output, show=show_plots)

        if args.pareto:
            plot_pareto_front(df, args.output, show=show_plots)

        if args.heatmap:
            plot_correlation_heatmap(df, args.output, show=show_plots)


if __name__ == "__main__":
    main()
