"""Plot and tabulate DE-KRR, KRR, PINN, and DNN repetition results."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import numpy as np


METHODS = ("DE-KRR", "KRR", "PINN", "DNN")
SIZES = (8000, 9000, 10000, 11000, 12000)
METRICS = ("point_coverage", "uniform_coverage", "mse", "radius")
PLOT_METHODS = ("DE-KRR", "DNN", "PINN", "KRR")
COLORS = ("#1f77b4", "#ff7f0e", "#d62728", "#2ca02c")
MARKERS = ("o", "^", "s", "x")
STYLES = ("-", "--", "-.", ":")


def load_results(data_dir):
    results = {}
    for method in METHODS:
        for n in SIZES:
            path = data_dir / method / f"n{n}_200_repetitions.csv"
            with path.open(encoding="utf-8-sig", newline="") as file:
                reader = csv.DictReader(file)
                headers = {name.strip().lower(): name for name in (reader.fieldnames or [])}
                missing = set(METRICS) - headers.keys()
                if missing:
                    raise ValueError(f"{path}: missing columns {', '.join(sorted(missing))}")
                rows = [[float(row[headers[key]]) for key in METRICS] for row in reader]
            values = np.asarray(rows, dtype=float)
            if values.shape != (200, len(METRICS)) or not np.isfinite(values).all():
                raise ValueError(f"{path}: expected 200 finite rows for the four metrics")
            if np.any(values < 0) or np.any(values[:, :2] > 1):
                raise ValueError(f"{path}: invalid coverage, MSE, or radius")
            results[method, n] = dict(zip(METRICS, values.T))
    return results


def plot_results(results, means, output_dir):
    def make_axes(ylabel):
        fig, ax = plt.subplots(figsize=(5.2, 2.7), layout="constrained")
        ax.set_xlabel("n", fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.grid(linestyle=":", linewidth=.6, color=".75")
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=9, length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        return fig, ax

    fig, ax = make_axes("Uniform coverage")
    for method, color, marker, style in zip(PLOT_METHODS, COLORS, MARKERS, STYLES):
        ax.plot(SIZES, [means[method, n]["uniform_coverage"] for n in SIZES],
                label=method, color=color, marker=marker, linestyle=style,
                linewidth=1, markersize=3)
    ax.axhline(.95, color="black", linewidth=.8)
    minimum = min(.95, *(row["uniform_coverage"] for row in means.values()))
    ax.set_ylim(max(0, minimum - .025), 1.012)
    ax.set_xticks(SIZES)
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.legend(loc="center left", bbox_to_anchor=(1.02, .5), fontsize=8)
    fig.savefig(output_dir / "uniform_coverage.png", dpi=300)
    plt.close(fig)

    fig, ax = make_axes("Bandwidth")
    centers = np.arange(len(SIZES))
    for i, (method, color) in enumerate(zip(PLOT_METHODS, COLORS)):
        ax.boxplot(
            [2 * results[method, n]["radius"] for n in SIZES],
            positions=centers + (i-1.5)*.18, widths=.15, manage_ticks=False,
            patch_artist=True,
            boxprops={"facecolor": color, "edgecolor": color, "linewidth": .6},
            medianprops={"color": "white", "linewidth": .9},
            whiskerprops={"color": color, "linewidth": .7},
            capprops={"color": color, "linewidth": .7},
            flierprops={"marker": ".", "markersize": 2, "alpha": .65,
                        "markeredgecolor": color},
        )
    ax.set_xticks(centers, SIZES)
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.legend(handles=[Patch(facecolor=color, label=method)
                       for method, color in zip(PLOT_METHODS, COLORS)],
              loc="center left", bbox_to_anchor=(1.02, .5), fontsize=8)
    fig.savefig(output_dir / "bandwidth.png", dpi=300)
    plt.close(fig)


def tex_mse(value):
    mantissa, exponent = f"{value:.2e}".split("e")
    return rf"\({mantissa}\times 10^{{{int(exponent)}}}\)" if value else "0"


def write_table(means, output_dir):
    lines = [
        r"% Requires \usepackage{booktabs}.",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Mean coverage, MSE, and band radius across 200 repetitions. "
        r"Radius denotes the reported half-width.}",
        r"\label{tab:method-comparison}",
        r"\begin{tabular}{rlrrrr}",
        r"\toprule",
        r" & & \multicolumn{2}{c}{Coverage (\%)} & & \\",
        r"\cmidrule(lr){3-4}",
        r"$n$ & Method & Point & Uniform & MSE & Radius \\",
        r"\midrule",
    ]
    for i, n in enumerate(SIZES):
        if i:
            lines.append(r"\addlinespace")
        for j, method in enumerate(METHODS):
            row = means[method, n]
            label_n = str(n) if j == 0 else ""
            lines.append(
                f"{label_n} & {method} & {100*row['point_coverage']:.2f} & "
                f"{100*row['uniform_coverage']:.2f} & {tex_mse(row['mse'])} & "
                f"{row['radius']:.4f}" + r" \\"
            )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (output_dir / "comparison.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent,
                        help="Parent folder of DE-KRR, KRR, PINN, and DNN")
    parser.add_argument("--output-dir", type=Path, help="Defaults to DATA_DIR/figures")
    args = parser.parse_args()
    output_dir = args.output_dir or args.data_dir / "figures"
    results = load_results(args.data_dir)
    means = {group: {key: float(values.mean()) for key, values in row.items()}
             for group, row in results.items()}
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_results(results, means, output_dir)
    write_table(means, output_dir)
    print(f"Saved comparison plots and LaTeX table in {output_dir.resolve()}")


if __name__ == "__main__":
    main()
