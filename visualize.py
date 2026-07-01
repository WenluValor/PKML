import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import MaxNLocator
from sklearn.linear_model import LinearRegression

def MSE_plot(case):
    files = {
        2000: "n2000.csv",
        2500: "n2500.csv",
        3000: "n3000.csv",
        3500: "n3500.csv",
        4000: "n4000.csv"
    }

    methods = ['ours', 'dnn', 'pinn', 'krr']
    label_dict = {
        "ours": "PDE-KRR",
        "dnn": "DNN",
        "krr": "KRR",
        "pinn": "PINN"
    }
    colors = ['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c']
    styles = ['-', '--', '-.', ':', '--']
    markers = ['o', '^', 's', 'x']

    results = {m: [] for m in methods}
    sample_sizes = []
    for n, file in files.items():
        df = pd.read_csv(case + '/' + file)
        sample_sizes.append(n)
        for method in methods:
            results[method].append(
                df[f"MSE_{method}"].mean()
            )
    fig = plt.figure(figsize=(5, 3))
    ax = fig.add_subplot(111)

    for i in range(len(methods)):
        method = methods[i]
        ax.plot(sample_sizes, results[method],
                label=label_dict[method], color=colors[i],
                linewidth=1, linestyle=styles[i], marker=markers[i],
                markersize=3)

    # ax.axhline(0.95, linestyle='-', color='black', linewidth=1)
    # ax.set_ylim(0.80, 1.00)
    ax.set_xticks(sample_sizes)
    ax.yaxis.set_major_formatter(
        FormatStrFormatter('%.3f')
    )
    ax.yaxis.set_major_locator(
        MaxNLocator(5)
    )

    ax.set_xlabel("n", fontsize=14, fontweight='bold')
    ax.set_ylabel("MSE", fontsize=14, fontweight='bold')
    ax.legend(
        loc='center left',
        bbox_to_anchor=(1, 0.5)
    )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.grid(linestyle='--', linewidth=0.5, color='gray', alpha=0.3)
    ax.tick_params(axis='x', width=0)
    ax.tick_params(axis='y', width=0)

    plt.tight_layout()
    plt.savefig(case + 'MSEplot.png', dpi=600, bbox_inches='tight')
    plt.show()

def line_plot(case, type):
    files = {
        2000: "n2000.csv",
        2500: "n2500.csv",
        3000: "n3000.csv",
        3500: "n3500.csv",
        4000: "n4000.csv"
    }

    methods = ["ours", "dnn", "pinn", "krr"]
    label_dict = {
        "ours": "PDE-KRR",
        "dnn": "DNN",
        "krr": "KRR",
        "pinn": "PINN"
    }
    colors = ['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c']
    styles = ['-', '--', '-.', ':', '--']
    markers = ['o', '^', 's', 'x']

    coverage_results = {m: [] for m in methods}
    sample_sizes = []
    for n, file in files.items():
        df = pd.read_csv(case + '/' + file)
        sample_sizes.append(n)
        for method in methods:
            coverage_results[method].append(
                df[f"{type}cov_{method}"].mean()
            )
    fig = plt.figure(figsize=(5, 3))
    ax = fig.add_subplot(111)

    for i in range(len(methods)):
        method = methods[i]
        ax.plot(sample_sizes, coverage_results[method],
                label=label_dict[method], color=colors[i],
                linewidth=1, linestyle=styles[i], marker=markers[i],
                markersize=3)

    ax.axhline(0.95, linestyle='-', color='black', linewidth=1)
    ax.set_ylim(0.20, 1.01)
    ax.set_xticks(sample_sizes)
    ax.yaxis.set_major_formatter(
        FormatStrFormatter('%.2f')
    )
    ax.yaxis.set_major_locator(
        MaxNLocator(5)
    )

    ax.set_xlabel("n", fontsize=14, fontweight='bold')
    ax.set_ylabel("Coverage", fontsize=14, fontweight='bold')
    ax.legend(
        loc='center left',
        bbox_to_anchor=(1, 0.5)
    )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.grid(linestyle='--', linewidth=0.5, color='gray', alpha=0.3)
    ax.tick_params(axis='x', width=0)
    ax.tick_params(axis='y', width=0)

    plt.tight_layout()
    plt.savefig(case + 'lineplot.png', dpi=600, bbox_inches='tight')
    plt.show()

def box_plot(case):
    files = {
        2000: "n2000.csv",
        2500: "n2500.csv",
        3000: "n3000.csv",
        3500: "n3500.csv",
        4000: "n4000.csv"
    }

    methods = ["ours", "dnn", "pinn", "krr"]
    colors = ['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c']
    conditions = ['2000', '2500', '3000', '3500', '4000']

    data = {'2000': {'ours': 0, 'dnn': 0, 'krr': 0, 'pinn': 0},
            '2500': {'ours': 0, 'dnn': 0, 'krr': 0, 'pinn': 0},
            '3000': {'ours': 0, 'dnn': 0, 'krr': 0, 'pinn': 0},
            '3500': {'ours': 0, 'dnn': 0, 'krr': 0, 'pinn': 0},
            '4000': {'ours': 0, 'dnn': 0, 'krr': 0, 'pinn': 0}}

    for method in methods:
        for n, file in files.items():
            df = pd.read_csv(case + '/' + file)
            data[str(n)][method] = np.array(df.loc[:, 'rad_' + method])

    all_data_list = []
    positions_list = []

    offset = [-0.3, -0.15, 0, 0.15, 0.3]
    group_center = [1, 2, 3, 4, 5]

    for j, method in enumerate(methods):
        all_data = []
        positions = []
        for i, condition in enumerate(conditions):
            all_data.append(data[condition][method])
            positions.append(group_center[i] + offset[j])
        all_data_list.append(all_data)
        positions_list.append(positions)

    fig = plt.figure(figsize=(5, 3))
    ax = fig.add_subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    for i, method in enumerate(methods):
        flierprops = dict(marker='o', markerfacecolor=colors[i],
                          markersize=2, markeredgecolor='none')
        meanlineprops = dict(linestyle='-', color='black', linewidth=0.0)
        medianlineprops = dict(linestyle='-', color='black', linewidth=0)
        bp = ax.boxplot(all_data_list[i], positions=positions_list[i], notch=True,
                        widths=0.15, patch_artist=True,
                        boxprops=dict(linewidth=0.0),
                        showmeans=True, meanline=meanlineprops,
                        flierprops=flierprops,
                        meanprops=meanlineprops, medianprops=medianlineprops,
                        capprops=dict(linewidth=0.5, color=colors[i]),
                        whiskerprops=dict(linewidth=0.5, color=colors[i]),)
        for patch in bp['boxes']:
            patch.set_facecolor(colors[i])
            # patch.set_edgecolor('none')

    x_ticks = [2000, 2500, 3000, 3500, 4000]

    ax.set_xticks(group_center, conditions)
    ax.set_xticklabels(x_ticks)
    ax.set_xlabel("n", fontsize=15, fontweight='bold')
    ax.set_ylabel('Bandwidth', fontsize=15, fontweight='bold', labelpad=10)
    ax.grid(linestyle='--', linewidth=0.5, color='gray', alpha=0.3)
    ax.tick_params(axis='x', width=0)
    ax.tick_params(axis='y', width=0)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f77b4', label='PDE-KRR'),
        Patch(facecolor='#ff7f0e', label='DNN'),
        Patch(facecolor='#d62728', label='PINN'),
        Patch(facecolor='#2ca02c', label='KRR')
    ]
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(case + 'boxplot.png', dpi=600, bbox_inches='tight')
    plt.show()

    return

def table():
    methods = ["ours", "dnn", "pinn", "krr"]
    row1 = {"n": 4000}
    row2 = {"n": 4000}
    for method in methods:
        df = pd.read_csv('bias/n4000.csv')
        row1[f"cov_{method}"] = df[f"unicov_{method}"].mean() * 100
        row1[f"rad_{method}"] = df[f"rad_{method}"].mean() * 100
        row1[f"MSE_{method}"] = df[f"MSE_{method}"].mean() * 100
        df = pd.read_csv('appr/n4000.csv')
        row2[f"cov_{method}"] = df[f"unicov_{method}"].mean() * 100
        row2[f"rad_{method}"] = df[f"rad_{method}"].mean() * 100
        row2[f"MSE_{method}"] = df[f"MSE_{method}"].mean() * 100

    for i in range(4):  # row
        str1 = ''
        str2 = ''
        method = methods[i]
        for j in range(3):  # column
            if j == 0:
                str1 += '& ' + "{:.2f}".format(row1['cov_' + method]) + ' '
                str2 += '& ' + "{:.2f}".format(row2['cov_' + method]) + ' '
            elif j == 1:
                str1 += '& ' + "{:.2f}".format(row1['rad_' + method]) + ' '
                str2 += '& ' + "{:.2f}".format(row2['rad_' + method]) + ' '
            else:
                str1 += '& ' + "{:.2f}".format(row1['MSE_' + method]) + ' '
                str2 += '& ' + "{:.2f}".format(row2['MSE_' + method]) + ' '
        if i == 0:
            print('\\textbf{PDE-KRR} ' + str1 + '& ' + str2 + '\\\\')
        elif i == 1:
            print('\\textbf{DNN} ' + str1 + '& ' + str2 + '\\\\')
        elif i == 2:
            print('\\textbf{PINN} ' + str1 + '& ' + str2 + '\\\\')
        elif i == 3:
            print('\\textbf{KRR} ' + str1 + '& ' + str2 + '\\\\ \\hline')


def verify_sqrt(case):
    files = {
        2000: "n2000.csv",
        2500: "n2500.csv",
        3000: "n3000.csv",
        3500: "n3500.csv",
        4000: "n4000.csv"
    }
    df = pd.read_csv(case + '/n2000.csv')
    B = df.shape[0]

    data = np.zeros([B, 5])
    i = 0
    for n, file in files.items():
        df = pd.read_csv(case + '/' + file)
        data[:, i] = np.array(df.loc[:, 'rad_ours'])
        i += 1

    coefs = np.zeros([B])
    for i in range(B):
        x = np.arange(2000, 4500, 500).reshape(-1, 1)
        X = -np.log(x)
        Y = np.log(data[i])
        reg = LinearRegression().fit(X, Y)
        coefs[i] = reg.coef_[0]

    print(np.average(coefs))


if __name__ == "__main__":
    # MSE_plot('bias')
    # line_plot('appr', type='uni')
    # box_plot('appr')
    table()
    # verify_sqrt(case='appr')
    exit(0)










