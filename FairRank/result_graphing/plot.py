import json
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
import csv
from ..data_analysis import combine as combine
import importlib
import os
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# from .. import MARC

with open('./FairRank/settings.json') as f:
    settings = json.load(f)
experiment_name = os.path.basename(settings["READ_FILE_SETTINGS"]["PATH"]).split(".")[0]

gender_labels = settings["GENDER_DATA_DEFINE"]
protected_keys = [k for k, v in gender_labels.items() if v == '1']
unprotected_keys = [k for k, v in gender_labels.items() if v == '0']
SMALL_GAMMA = settings["DELTR_OPTIONS"]["small_gamma"]
LARGE_GAMMA = settings["DELTR_OPTIONS"]["large_gamma"]

delimiters = "_", "/", "\\", "(", ")", "=", ",", '\"', " ", "."
regex_pattern = '|'.join(map(re.escape, delimiters))

delimiters_2 = "=", ","
regex_pattern_2 = '|'.join(map(re.escape, delimiters_2))

# if 'Female' in protected_keys:
#     protected_group = 'Females'
#     unprotected_group = 'Males'
#
# if 'Female' in unprotected_keys:
#     protected_group = 'Males'
#     unprotected_group = 'Females'

# flip_choice = settings["DELTR_OPTIONS"]["flip_choice"]

# graph_font = {'family': 'serif'}
#
# plt.rcParams["font.family"] = "serif"

protected_group_dict = {'(W)NBA': 'Females', 'Boston Marathon': 'Females', 'COMPAS': 'Males', 'LAW': 'Females'}
dataset_dict = {'bostonmarathon': 'Boston Marathon', 'NBAWNBA': '(W)NBA', 'COMPASSEX': 'COMPAS', 'LAW': 'LAW'}


def get_files(directory, pipeline):
    temp = []
    pre_temp = []
    if pipeline is None:
        for dirpath, dirnames, filenames in os.walk(directory):
            for file in filenames:
                if not file.endswith('.png'):  # Exclude .png files
                    temp.append(os.path.join(dirpath, file))
    else:
        for dirpath, dirnames, filenames in os.walk(directory):
            for dirname in dirnames:
                # Check if the dirname contains the substring
                if pipeline in dirname:
                    pre_temp.append(os.path.join(dirpath, dirname))
        for tempies in pre_temp:
            for dirpath, dirnames, filenames in os.walk(tempies):
                for file in filenames:
                    if not file.endswith('.png'):  # Exclude .png files
                        temp.append(os.path.join(dirpath, file))
    return temp


def PlotGraphs():
    print(experiment_name)
    # metrics = get_files('./FairRank/Results/seed42/' + experiment_name + '/' + flip_choice + '/', None)

    csvs = get_files('./FairRank/ResultsCSVs/', None)

    csvs_initial = get_files('FairRank/ResultsInitial', None)
    res_csvs = [i for i in csvs if 'ndcg.csv' not in i and 'skews.csv' not in i]
    res_case = [i for i in res_csvs if 'CaseStudies' in i and 'AvgExp' not in i]
    res_synth = [i for i in res_csvs if 'Synthetic' in i and 'AvgExp' not in i]
    # res_skews = [i for i in csvs if 'skews.csv' in i]
    res_skews_initial = [i for i in csvs_initial if 'skews.csv' in i]
    res_case_avg_exp = [i for i in res_csvs if 'CaseStudies' in i and 'AvgExp_0' in i]

    # for csv_file in res_case:
    #     plot_case(csv_file)
    for csv_file in res_synth:
        plot_synth(csv_file)
    # for csv_file in res_skews:
    #     plot_skew(csv_file)
    # for csv_file in res_skews_initial:
    #     plot_skew(csv_file)
    # for csv_file in res_case_avg_exp:
    #     csv_corres_file = csv_file.replace('AvgExp_0', 'AvgExp_1')
    #     plot_case_avg_exp(csv_file, csv_corres_file)
    # plot_legend('synth')
    # plot_legend()


def plot_case_avg_exp(grp_0_file, grp_1_file):
    df_adv = pd.read_csv(grp_0_file)
    df_dis = pd.read_csv(grp_1_file)

    df_adv_cleaned = df_adv.dropna(axis=0, how='any')
    df_dis_cleaned = df_dis.dropna(axis=0, how='any')
    # rearrange rows by column 'Inference %', from 0% to 100%
    df_adv_cleaned.sort_values(by=df_adv_cleaned.columns[0], inplace=True)
    df_dis_cleaned.sort_values(by=df_dis_cleaned.columns[0], inplace=True)

    # # move 100 row to bottom
    # row_to_move_adv = df_adv_cleaned[df_adv_cleaned[0] == 100]
    # row_to_move_dis = df_dis_cleaned[df_dis_cleaned[0] == 100]
    #
    # # Remove the row from the original DataFrame
    # df_adv_cleaned = df_adv_cleaned[df_adv_cleaned[0] != 100]
    # df_dis_cleaned = df_adv_cleaned[df_dis_cleaned[0] != 100]

    # Create a new DataFrame with the desired row as the bottom row
    # df_adv_cleaned = pd.concat([df_adv_cleaned, row_to_move_adv], ignore_index=True)
    # df_dis_cleaned = pd.concat([df_dis_cleaned, row_to_move_dis], ignore_index=True)

    df_adv_cleaned.to_csv(grp_0_file, index=False)
    df_dis_cleaned.to_csv(grp_1_file, index=False)

    dataset = dataset_dict[re.split(regex_pattern, grp_0_file)[-4]]
    infer_choice = re.split(regex_pattern, grp_0_file)[-5]

    num_rows = 3
    num_cols = 2

    for column in df_dis_cleaned.columns[1:]:  # Start from the second column (excluding the first column as x-labels)
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 8))

        data1 = df_dis_cleaned[['Wrong Inference Percentage', column]]
        data2 = df_adv_cleaned[['Wrong Inference Percentage', column]]
        counter = 0
        # Merge the dataframes vertically
        # combined_data = pd.concat([data1.iloc[:, 0],data1.iloc[:, 1:], data2.iloc[:, 1:]], axis=1, ignore_index=True)
        combined_data = pd.concat([data1, data2], axis=0, ignore_index=True)
        combined_data.columns = [r'$g_{dis}$', r'$g_{adv}$']

        grouped_data = combined_data.groupby('Wrong Inference Percentage').mean()
        # Create a bar graph for the current pair of columns
        for i in range(num_rows):
            for j in range(num_cols):
                ax = axs[i, j]
                ax.bar(grouped_data.index, grouped_data[combined_data.columns[1:]], label=['$g_{dis}$', '$g_{adv}$'])
                ax.set_title(f'Bar Graph for {column}')
                ax.set_xlabel('Wrong Inference Percentage')
                ax.set_ylabel('Mean Value')
                ax.legend()

                plt.tight_layout()

                # create directory
                directory = './FairRank/Graphs/' + infer_choice + '/' + dataset + '/'

                # check that directory exists or create it
                if not os.path.exists(directory):
                    os.makedirs(directory)

            plt.tight_layout()
            # save plot
            plt.savefig(str(directory) + 'AvgExp.pdf')
    # plt.close()


def plot_synth(csv_file_path):
    df = pd.read_csv(csv_file_path)
    # remove rows with missing columns from df
    df_cleaned = df.dropna(axis=0, how='any')
    # rearrange rows by column 'Inference %', from 0% to 100%
    df_cleaned.sort_values(by=['Wrong Inference Percentage'], inplace=True)

    # move 100 row to bottom
    row_to_move = df_cleaned[df_cleaned['Wrong Inference Percentage'] == 100]

    # Remove the row from the original DataFrame
    df_cleaned = df_cleaned[df_cleaned['Wrong Inference Percentage'] != 100]

    # Create a new DataFrame with the desired row as the bottom row
    df_cleaned = pd.concat([df_cleaned, row_to_move], ignore_index=True)

    df_cleaned.to_csv(csv_file_path, index=False)

    # get dataset from file name
    dataset = re.split(regex_pattern, csv_file_path)[-3]

    # get metric from file name
    y = re.split(regex_pattern, csv_file_path)[-2]

    # plot line graphs with same colors for every three columns after the first column
    colors = ['#F00000', '#3D6D9E', '#FFC725', '#3EC372', '#808080']
    # colors = ['#1F77B4', '#FFC725', '#2CA02C', '#D62728']
    markers = ['*', 'o', '^']
    linestyles = ['-', '--', ':']

    new_names = {
        'ULTR': 'ListNet',
        'FLTR': 'FairLTR',
        'ULTR + PostF': 'DCS',
        'ULTRH + PostF': 'Hidden + DCS',
        'LTR + PostF': 'LTR + DCS'
    }
    # change column names
    df_cleaned.columns = df_cleaned.columns.map(lambda x: change_text(x, new_names))

    data_columns = df_cleaned[1:]
    print('HERE ', data_columns)

    fig, ax = plt.subplots(figsize=(4.5, 3))
    x_data = df_cleaned.iloc[:, 0]
    # x_data_percent = x_data.apply(lambda x: f'{x}%')

    y_data = df_cleaned.iloc[:, 1:]

    # for column in y_data.columns:
    #     ax.plot(x_data, y_data[column], label=column)
    # print(csv_file_path)
    # plt.show()

    # Loop through the data columns and plot each set of three columns with the same color
    for i, column in enumerate(y_data.columns):
        color = colors[i % len(colors)]  # Cycle through the colors
        # print("x =", df_cleaned['Wrong Inference Percentage'], "y =", df_cleaned[column])
        ax.plot(x_data.astype(int), y_data[column], label=column, color=color)
        # ax.plot(df_cleaned['Wrong Inference Percentage'], df_cleaned[column], label=column, color=color)
    # plt.show()

    for i, column in enumerate(df_cleaned.columns[1:]):
        color = colors[i // 3 % len(colors)]
        marker = markers[i % len(markers)]
        linestyle = linestyles[i % len(linestyles)]
        y_data = df_cleaned[column]  # Get the y-axis data for the current column
        markersize = 6
        ax.plot(x_data.astype(int), y_data, label=column, color=color, marker=marker, markerfacecolor='none', linestyle=linestyle,
                markersize=markersize, lw=1)
        # removing markers for presentation
        ax.plot(x_data.astype(int), y_data, label=column, color=color, linestyle=linestyle, lw=1)

    # plt.show()
    lw = 1
    custom_legend = [
        Patch(color='#F00000', label='ListNet'),
        Patch(color='#3D6D9E', label='FairLTR'),
        Patch(color='#FFC725', label='DCS'),
        Patch(color='#3EC372', label='Hidden + DCS'),
        Patch(color= '#808080', label='LTR + DCS'),
        Line2D([0], [0], color='black', lw=lw, linestyle='-', marker='*', markersize=markersize,
               label=r'$g_{dis}\rightleftarrows g_{adv}$',
               markerfacecolor='none'),
        Line2D([0], [0], color='black', lw=lw, linestyle='--', marker='o', markersize=markersize,
               label=r'$g_{dis}\rightarrow g_{adv}$',
               markerfacecolor='none'),
        Line2D([0], [0], color='black', lw=lw, linestyle=':', marker='^', markersize=markersize,
               label=r'$g_{dis}\leftarrow g_{adv}$',
               markerfacecolor='none'),
    ]

    results_to_search = get_files('./FairRank/Results/seed42/' + dataset + '/both/', None)
    # search for path that has gamma=0.0 and 'Colorblind
    ultrh_path = [i for i in results_to_search if 'gamma=0.0' in i and 'Colorblind' in i]
    ltr_path = [i for i in results_to_search if 'gamma=0.0' in i and 'BlindGroundTruth' in i]
    # get y metric

    if 'NDCG' in y:
        dataset = dataset_dict[dataset]
        ultrh_path = [i for i in ultrh_path if 'ndcg' in i]
        ltr_path = [i for i in ltr_path if 'ndcg' in i]
        match = re.search(r'\d+', y)
        # ultrh_value = float(combine.get_NDCG(ultrh_path[0], 100))
        # plt.title(str(dataset), fontsize='xx-large', color='white')
        plt.title(str(dataset), fontsize='xx-large')
        ax.yaxis.set_major_formatter('{:.3f}'.format)
        if match:
            value = int(match.group())
            # ultrh_path.replace = ultrh_path[0].replace("\\", "/")
            ultrh_value = float(combine.get_NDCG(ultrh_path[0], value))
            ltr_value = float(combine.get_NDCG(ltr_path[0], value))

            # ax.yaxis.set_major_formatter('{:.3f}'.format)

            #if 'LAW' in dataset:
            plt.ylabel(str('NDCG@' + str(value)), fontsize='x-large')  # , fontsize=15)
                # ax.yaxis.set_major_formatter('{:.3f}'.format)
            # else:
            #     plt.ylabel(' ')
                # ax.set_yticklabels([])

    elif y == 'ExpR' or y == 'NDKL':

        ultrh_path = [i for i in ultrh_path if 'metrics' in i]
        ltr_path = [i for i in ltr_path if 'metrics' in i]
        # dataset_dict = {'bostonmarathon': 'Boston Marathon', 'NBAWNBA': '(W)NBA', 'COMPASSEX': 'COMPAS',
        #                 'LAW': 'LAW'}
        dataset = dataset_dict[dataset]
        plt.title(str(dataset), fontsize='xx-large')
        if y == 'ExpR':
            ultrh_value = float(combine.get_ExpR(ultrh_path[0]))
            ltr_value = float(combine.get_ExpR(ltr_path[0]))
            #if 'LAW' in dataset:
            plt.ylabel(str('DAdv/Adv Average Exposure Ratio'), fontsize=11)  # , fontsize=15)
        else:  # y=='NDKL'
            ultrh_value = float(combine.get_NDKL(ultrh_path[0]))
            ltr_value = float(combine.get_NDKL(ltr_path[0]))

            #if 'LAW' in dataset:
            plt.ylabel(str('NDKL'), fontsize='x-large')  # , fontsize=15)
                #ax.set_yticklabels([])

    # plt.show()

    if y == 'ExpR':
        y_max = 1.26  # max(plt.ylim()[1], 1.21)
        y_min = 0.64  # min(plt.ylim()[0], 0.64)

        # if "osto" in dataset:
        #     y_max = 1.5

        # y_min = min(plt.ylim()[0], ultrh_value)
        # y_max = max(plt.ylim()[1], ultrh_value)

    elif 'COMPAS' in dataset and (y == 'NDCG10' or y == 'NDCG50'):
        y_min = min(plt.ylim()[0], ultrh_value)
        y_max = 0.75

    elif 'COMPAS' in dataset and 'NDCG' in y:
        y_min = min(plt.ylim()[0], ultrh_value)
        y_max = 0.68
    else:

        y_min = min(plt.ylim()[0], ultrh_value)
        y_max = max(plt.ylim()[1], ultrh_value)
    plt.ylim(y_min, y_max)

    center_x = (plt.xlim()[0] + plt.xlim()[1]) / 10
    # center_y_1 = y_max - 1/8 *(y_max - y_min)
    if y == 'NDKL' and dataset == 'LAW':
        center_y_1 = 0.03
    else:
        center_y_1 = y_max - 1 / 8 * (y_max - y_min)
    # print("center_x=", center_y)

    # plot straight line

    # plt.show()
    # label_text = 'Hidden'
    plt.axhline(y=ultrh_value, color='cyan', linestyle='--', label='ULTRH')
    plt.axhline(y=ltr_value, color='blue', linestyle=':', label='LTR')
    # plt.text(max(plt.xlim()), ultrh_value, label_text, color='black', fontsize='small')

    # point to line

    # plt.annotate('Hidden', xy=(85, ultrh_value), xytext=(85, center_y_1),
    #              arrowprops=dict(arrowstyle='->'),
    #              fontsize=10, color='b', weight='bold')

    # create directory
    directory = './FairRank/Graphs/Synthetic/' + dataset + '/'

    # check that directory exists or create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.xlabel(r'Wrong Inference $\epsilon$ (%)', fontsize='x-large')  # , fontsize=8)

    plt.tight_layout()
    # save plot
    plt.savefig(str(directory) + y + '.pdf')
    # if "oston" in dataset:
    #     plt.savefig(str(directory) + y + '.png')
    plt.close()


def plot_case(csv_file_path):
    """
    This function plots all graphs per ranking
    :param csv_file_path:
    :return:
    """
    df = pd.read_csv(csv_file_path)
    # change column name 'Wrong Inference Percentage'
    df.rename(columns={'Wrong Inference Percentage': 'Inference Service'}, inplace=True)
    # remove rows with missing columns from df
    df_cleaned = df.dropna(axis=0, how='any')
    # rearrange rows by column 'Inference Service', put row named "GAPI" at the top

    print('here', df_cleaned)
    # move GAPI row to top
    row_to_move = df_cleaned[df_cleaned['Inference Service'] == "GAPI"]  #
    #
    # # Remove the row from the original DataFrame
    df_cleaned = df_cleaned[df_cleaned['Inference Service'] != "GAPI"]

    # Create a new DataFrame with the desired row as the first row
    df_cleaned = pd.concat([row_to_move, df_cleaned], ignore_index=True)

    df_cleaned.to_csv(csv_file_path, index=False)
    # df_cleaned.set_index('Inference Service', inplace=True)

    # check if df_cleaned is empty
    if df_cleaned.empty:
        return
    print(df_cleaned)

    # get dataset from file name
    dataset = re.split(regex_pattern, csv_file_path)[-3]

    # get metric from file name
    y = re.split(regex_pattern, csv_file_path)[-2]

    # Create a custom order for the 'Inference Service' categories
    custom_order = ['GAPI', 'BTN', 'NMSOR']
    df_cleaned['Inference Service'] = pd.Categorical(df_cleaned['Inference Service'], categories=custom_order)

    colors = ['#F00000', '#3D6D9E', '#FFC725', '#3EC372', '#808080']

    fig, ax = plt.subplots(figsize=(2.9, 3))

    # plot bar graph grouped by column named 'Inference Service'

    df_cleaned.groupby('Inference Service').mean().plot(kind='bar', stacked=False, legend=False, edgecolor='k',
                                                        color=colors, ax=ax, width=0.8)

    plt.xlabel("Inference Service")  # , fontsize=8)

    plt.xticks(rotation=0)

    # if 'NBA' in dataset and (y == 'ExpR' or y == 'NDKL'):
    #     # plot legend
    #     plt.legend(loc='center', ncol=2, fontsize='small')

    # plot ULTRH line

    results_to_search = get_files('./FairRank/Results/seed42/' + dataset + '/CaseStudies/', None)
    # search for path that ha gamma=0.0 and 'Colorblind'
    ultrh_path = [i for i in results_to_search if 'gamma=0.0' in i and 'Colorblind' in i]
    ltr_path = [i for i in results_to_search if 'gamma=0.0' in i and 'BlindGroundTruth' in i]
    # get y metric
    y_min = 0
    if 'NDCG' in y:
        ultrh_path = [i for i in ultrh_path if 'ndcg' in i]
        ltr_path = [i for i in ltr_path if 'ndcg' in i]
        match = re.search(r'\d+', y)
        plt.title(str(dataset), fontsize='xx-large')
        if match:
            value = int(match.group())
            # ultrh_path.replace = ultrh_path[0].replace("\\", "/")
            ultrh_value = float(combine.get_NDCG(ultrh_path[0], value))
            ltr_value = float(combine.get_NDCG(ltr_path[0], value))

            plt.ylabel(str('NDCG@' + str(value)))
            dataset = dataset_dict[dataset]
            plt.title(str(dataset), fontsize='xx-large')

            # if 'NBA' in dataset:
            #     plt.ylabel(str('NDCG@' + str(value)))  # , fontsize=20)
            # else:
            #     plt.ylabel(' ')
            #     ax.set_yticklabels([])


    elif y == 'ExpR' or y == 'NDKL':

        ultrh_path = [i for i in ultrh_path if 'metrics' in i]
        ltr_path = [i for i in ltr_path if 'metrics' in i]
        # dataset_dict = {'bostonmarathon': 'Boston Marathon', 'NBAWNBA': '(W)NBA', 'COMPASSEX': 'COMPAS'}
        dataset = dataset_dict[dataset]
        plt.title(str(dataset), fontsize='xx-large')

        if y == 'ExpR':
            ultrh_value = float(combine.get_ExpR(ultrh_path[0]))
            ltr_value = float(combine.get_ExpR(ltr_path[0]))
            plt.ylabel(str('DAdv/Adv Average Exposure Ratio'))
            # if 'NBA' in dataset:
            #     plt.ylabel(str('DAdv/Adv Average Exposure Ratio'))  # , fontsize=18)
        else:  # y=='NDKL'
            ultrh_value = float(combine.get_NDKL(ultrh_path[0]))
            ltr_value = float(combine.get_NDKL(ltr_path[0]))
            plt.ylabel(str('NDKL'))
            # if 'NBA' in dataset:
            #     plt.ylabel(str('NDKL'))  # , fontsize=20)

    # if 'NBA' in dataset and y == 'ExpR':
    #     y_max = 1.05
    # else:
    #     y_max = max(plt.ylim()[1], ultrh_value)
    #
    # y_min = min(plt.ylim()[0], ultrh_value)
    # # y_max = max(plt.ylim()[1], ultrh_value)
    if y == 'ExpR':
        #y_min = 0.5
        y_max = 1.2
    elif y == 'NDCG10':
        #y_min = 0.3
        y_max = 0.8
    elif 'NDCG' in y:
        #y_min = 0.3
        y_max = 0.72
    else:
        #y_min = min(plt.ylim()[0], ultrh_value)
        y_max = max(plt.ylim()[1], ultrh_value)
    plt.ylim(y_min, y_max)

    x_min = plt.xlim()[0]
    x_max = plt.xlim()[1]

    # plot straight line
    label_text = 'ULTRH'
    plt.axhline(y=ultrh_value, color='cyan', linestyle='--', label='ULTRH', linewidth=1.5)
    plt.axhline(y=ltr_value, color='blue', linestyle=':', label='LTR', linewidth=1.5)
    # plt.text(max(plt.xlim()), ultrh_value, label_text, fontsize='small', color='black')
    center_x_0 = x_max - 1 / 1.5 * (x_max - x_min)
    center_x_1 = x_max - 0.64 * (x_max - x_min)
    center_y = y_max - .07 * (y_max - y_min)
    # print("center_x=", center_y))

    # point to line
    # plt.annotate('LTR:hid', xy=(center_x_1, ultrh_value), xytext=(center_x_1, center_y),
    #              arrowprops=dict(arrowstyle='->'),
    #              fontsize=10, color='b', weight='bold', horizontalalignment='center')

    # create directory
    # if dataset_dict[dataset] in dataset_dict:
    #     directory = './FairRank/Graphs/CaseStudies/' + dataset_dict[dataset] + '/'
    # else: directory = './FairRank/Graphs/CaseStudies/' + dataset + '/'

    directory = './FairRank/Graphs/CaseStudies/' + dataset + '/'

    # check that directory exists or create it
    if not os.path.exists(directory):
        os.makedirs(directory)
    ax.tick_params(axis='x', which='both', length=0)
    plt.tight_layout()
    # save plot
    plt.savefig(str(directory) + y + '.pdf')
    plt.close()


def plot_skew(metrics_file_path):
    skew_file = pd.read_csv(metrics_file_path)

    skew_file['Ideal'] = 1

    # get dataset from file path
    dataset = re.split(regex_pattern, metrics_file_path)[-5]
    dataset = dataset_dict[dataset]
    g_dis = protected_group_dict[dataset]

    if g_dis == 'Females':
        g_adv = 'Males'
    else:
        g_adv = 'Females'

    skew_file.rename(columns={'Group 0': str(g_adv), 'Group 1': str(g_dis)}, inplace=True)
    sns.set(style="white")
    fig, ax = plt.subplots(figsize=(4, 4))

    sns.set(font_scale=1.2,  # Font scale factor (adjust to your preference)
            rc={"font.style": "normal",  # Set to "normal", "italic", or "oblique"
                "font.family": "serif",  # Choose your preferred font family
                # Font size
                "font.weight": "normal"  # Set to "normal", "bold", or a numeric value
                })

    sns.lineplot(data=skew_file[[g_dis, g_adv, "Ideal"]], dashes=False, ax=ax)

    # pipe, graph_title = get_graph_name(metrics_file_path)

    ax.set_title(dataset, fontsize='x-large', fontfamily='serif')

    # ax.title.set_position([0.5, -0.1])

    # ax.set_xticks(range(0, len(metrics_file), 200))

    # Set the x and y limits to start from 0
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.xlabel("Ranking position", fontfamily='serif')
    plt.ylabel("Skew", fontfamily='serif')
    plt.tight_layout()
    plt.legend(frameon=False, fontsize='xx-small', loc='upper right')
    # if dataset=='LAW':
    #     plt.legend(frameon=False, fontsize='xx-small',loc='upper right')
    # else:
    #     plt.legend("", frameon=False)
    # check = metrics_file_path.split(os.sep)[-2].split("/")[-1]
    # check2 = metrics_file_path.split(os.sep)
    """ DIRECTORY MANAGEMENT """
    graph_path = Path(
        "./FairRank/Graphs/Initial/" +
        dataset + "/" + str(metrics_file_path.split(os.sep)[-2].split("/")[-1]))

    if not os.path.exists(graph_path):
        os.makedirs(graph_path)

    plt.savefig(os.path.join(graph_path, str(dataset) + '_skews.pdf'))
    plt.close()

    return


def plot_legend(option='case'):
    # Create a separate figure and axis for the legend
    figsize = (8, 1)
    if option == 'synth':
        figsize = (12.5, 1)
    legend_fig, legend_ax = plt.subplots(figsize=figsize)
    lw = 1
    markersize = 8

    if option == 'synth':
        legend_items = [
            Patch(color='#F00000', label='LTR:inf'),
            Patch(color='#3D6D9E', label='FairLTR:inf'),
            Patch(color='#FFC725', label='LTR:inf-FairRR:inf'),
            Patch(color='#3EC372', label='LTR:hid-FairRR:inf'),
            Patch(color='#808080', label='LTR-FairRR:inf'),
            Line2D([0], [0], color='cyan', lw=1.5, linestyle='--', label='LTR:hid'),
            Line2D([0], [0], color='blue', lw=1.5, linestyle=':', label='LTR'),
            Line2D([0], [0], color='black', lw=lw, linestyle='-', marker='*', markersize=markersize,
                   label=r'$g_{dis}\,\rightleftarrows\,g_{adv}$',
                   markerfacecolor='none'),
            Line2D([0], [0], color='black', lw=lw, linestyle='--', marker='o', markersize=markersize,
                   label=r'$g_{dis}\rightarrow g_{adv}$',
                   markerfacecolor='none'),
            Line2D([0], [0], color='black', lw=lw, linestyle=':', marker='^', markersize=markersize,
                   label=r'$g_{dis}\leftarrow g_{adv}$',
                   markerfacecolor='none'),

        ]
        # plt.text(-0.05, 0.47, 'Legend', fontsize=10, weight='bold')
        ncol = 5


    else:

        # Create custom legend items using matplotlib.patches.Patch
        legend_items = [
            Patch(color='#F00000', label='LTR:inf'),
            Patch(color='#3D6D9E', label='FairLTR:inf'),
            Patch(color='#FFC725', label='LTR:inf-FairRR:inf'),
            Patch(color='#3EC372', label='LTR:hid-FairRR:inf'),
            Patch(color='#808080', label='LTR-FairRR:inf'),
            Line2D([0], [0], color='cyan', lw=1.5, linestyle='--', label='LTR:hid'),
        Line2D([0], [0], color='blue', lw=1.5, linestyle=':', label='LTR'),]
        # Line2D([0], [0], color='black', lw=lw, linestyle='--',
        #        label='ULTRH',
        #        markerfacecolor='none')

        # Write legend
        # plt.text(-0.04, 0.47, 'Legend', fontsize=10, weight='bold')
        ncol = 4

    for item in legend_items[0:5]:
        item.set_edgecolor('black')

    plt.axis('off')

    # Add the legend to the separate legend axis
    legend_ax.legend(handles=legend_items, loc='center', ncol=ncol, edgecolor='k')
    plt.tight_layout()

    # Remove the axis and display only the legend
    # legend_fig.savefig('legend.pdf', bbox_inches='tight', dpi=300)

    # Display the legend figure
    # plt.show()

    if option == 'synth':
        plt.savefig('legend_synth.pdf')
    else:
        plt.savefig('legend_case.pdf')

    # plt.savefig('legend_' + str(option) + 'forquals.pdf')

    return


def change_text(column_name, names):
    parts = column_name.split('_')
    if parts[0] in names:
        parts[0] = names[parts[0]]
    return '_'.join(parts)


###################################################################################################
# res_skews = [i for i in metrics if 'skews' in i]
# res_metrics = [i for i in metrics if 'metrics' in i]
# res_ndcg = [i for i in metrics if 'ndcg' in i]
#
# for file in res_skews:
#     plot_skew(file, flip_choice)
# plot_ndkl(flip_choice)
# #
# # for file in res_ndcg:
# #     plot_ndcg(file)
# # # plot_joint_ndcg(res_ndcg)
# #
# # plot_avg_exp()
# # # plot_posDiff()
# # # plot_scoreDis()

# def plot_ndcg(ndcg_file_path):
#     """
#     This function plots ndcg graphs per ranking
#     :param ndcg_file_path:
#     :return:
#     """
#     metrics_file = pd.read_csv(ndcg_file_path)
#     # metrics_file.rename(columns={'Group 0': str(unprotected_group), 'Group 1': str(protected_group)}, inplace=True)
#     sns.set(style="white")
#     fig, ax = plt.subplots()
#     chart = sns.lineplot(data=metrics_file[["NDCG"]], dashes=False, ax=ax)
#
#     if 'Initial' in ndcg_file_path:
#         graph_title = 'Initial'
#         pipe = 'Initial'
#
#     else:
#         pipe, graph_title = get_graph_name(ndcg_file_path)
#
#     chart.set_title(graph_title, fontdict={'size': 15})
#
#     # ax.set_xticks(range(0, len(metrics_file), 100))
#
#     plt.xlabel("Ranking position")
#     plt.ylabel("NDCG")
#
#     plt.legend(fontsize='large')
#
#     # plt.ylim(None, 1)
#     # ax.yaxis.set_major_locator(ticker.MultipleLocator(.05))
#
#     y_indices_to_trace = [10, 50, 100, 250, 500]
#     # y_indices_to_trace = [1, 2, 3, 4, 5, 6, 7, 8, 9]
#
#     x_values = metrics_file.index
#
#     x_values_to_trace = x_values[y_indices_to_trace]
#
#     ax.set_xlim(left=0)
#     ax.set_ylim(bottom=0)
#
#     xmin, xmax = plt.xlim()
#     ymin, ymax = plt.ylim()
#
#     # Trace out the x and y values using vertical and horizontal lines
#     for x, y in zip(x_values_to_trace, metrics_file.loc[y_indices_to_trace, 'NDCG']):
#         plt.plot([x, x], [ymin, y], color='grey', linestyle='--')
#         plt.plot([xmin, x], [y, y], color='grey', linestyle='--')
#
#     """ DIRECTORY MANAGEMENT """
#     graph_path = Path(
#         './FairRank/Graphs/flipchoice-' + flip_choice + '/' +
#         os.path.basename(settings["READ_FILE_SETTINGS"]["PATH"]).split('.')[
#             0] + "/" + pipe + '/' + str(ndcg_file_path.split(os.sep)[-2].split("/")[-1]))
#
#     if not os.path.exists(graph_path):
#         os.makedirs(graph_path)
#
#     plt.savefig(os.path.join(graph_path, 'ndcg.png'))
#     plt.close()
#
#     return
#
#
# def go_and_come(res_path, merged, suffix_1):
#     suffix_2 = re.split(regex_pattern, res_path)[-4]
#     print('suffix_2', suffix_2)
#     df = pd.read_csv(res_path)
#     merged = pd.merge(merged, df, on='Position', suffixes=('_' + str(suffix_1), '_' + str(suffix_2)))
#     column_names = {'NDCG_' + str(suffix_1): str(suffix_1), 'NDCG_' + str(suffix_2): str(suffix_2)}
#     merged = merged.rename(columns=column_names)
#     suffix_1 = suffix_2
#
#     return merged, suffix_1
#
#
# def plot_joint_ndcg(res_ndcg):
#     """
#     This function plots ndcg graphs per position. It takes the positions from
#     different files across inferences for the same gamma value for inferred and detconst
#     sort pipes. It plots across gamma values for GroundTruth
#     :param res_ndcg: all the ndcg files
#     :return:
#     """
#
#     # ['Initial', 'DetConstSort', 'Inferred', 'GroundTruth']
#     pipeline = list(set([re.split(regex_pattern, r)[5] for r in res_ndcg]))
#
#     for pipe in pipeline:
#         suffix_1 = 'init'
#         ndcg_file_save_path = Path('./FairRank/Graphs/flipchoice-' + flip_choice + '/' +
#                                    os.path.basename(settings["READ_FILE_SETTINGS"]["PATH"]).split('.')[
#                                        0] + "/" + pipe + '/')
#
#         if not os.path.exists(ndcg_file_save_path):
#             os.makedirs(ndcg_file_save_path)
#
#         # create dataframe with arbitary position
#         merged = pd.DataFrame(columns=['Position', 'NDCG'])
#         merged['Position'] = range(1, 10000)
#
#         if pipe == 'DetConstSort':
#             res_DCS = [i for i in res_ndcg if 'DetConstSort' in i]
#             for res_path in res_DCS:
#                 merged, suffix_1 = go_and_come(res_path, merged, suffix_1)
#             plot_merged_ndcg(merged, pipe, gamma="NA")
#
#         elif pipe == 'Inferred':
#             res_inf = [i for i in res_ndcg if 'Inferred' in i and 'DetConstSort' not in i]
#             gamma_pipe = list(set([re.split(regex_pattern, r)[-6] for r in res_inf]))
#             for g in gamma_pipe:
#                 # merged = pd.DataFrame(columns=['Position', 'NDCG'])
#                 res_g = [i for i in res_inf if '=' + str(g) in i]
#                 for path in res_g:
#                     merged, suffix_1 = go_and_come(path, merged, suffix_1)
#                 plot_merged_ndcg(merged, pipe, gamma=g)
#         elif pipe == 'GroundTruth':
#             res_GT = [i for i in res_ndcg if 'GroundTruth' in i]
#             for GT in res_GT:
#                 merged, suffix_1 = go_and_come(GT, merged, suffix_1)
#             plot_merged_ndcg(merged, pipe, gamma='Across gamma')
#         elif pipe == 'InitialTraining':
#             res_IT = [i for i in res_ndcg if 'InitialTraining' in i]
#             for IT in res_IT:
#                 merged, suffix_1 = go_and_come(IT, merged, suffix_1)
#             plot_merged_ndcg(merged, pipe, gamma='Train')
#         merged.to_csv(os.path.join(ndcg_file_save_path, 'merged_ndcg.csv'))
#
#
# def plot_merged_ndcg(merged, pipe, gamma=None):
#     """
#
#     :param merged:
#     :param pipe:
#     :param gamma:
#     :return:
#     """
#
#     test_df = pd.read_csv('./FairRank/Datasets/' + experiment_name + '/Testing/Testing_' + experiment_name + '.csv')
#     rank_length = len(test_df)
#     print('rank length = ', rank_length)
#
#     sns.set(style="white")
#
#     if "NDCG" in merged.columns:
#         merged.rename(columns={'NDCG': '90'}, inplace=True)
#     merged = merged.drop(columns=['Position', 'init'], axis=1)
#
#     if gamma == 'Across gamma' or gamma == 'Train':
#         ne_columns = merged.columns
#     else:
#         ne_columns = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100']
#     merged = merged[ne_columns]
#
#     # print merged
#
#     fig, ax = plt.subplots(figsize=(20, 10))
#     # add more space on the right for the labels
#     # plt.subplots_adjust(left=.15, right=.80)
#
#     NDCG_values = [50]
#     # NDCG_values = [10, 50, 100, 250, 500, rank_length]
#     # NDCG_values = [1700, rank_length]
#     palette = sns.color_palette('Set1', len(NDCG_values))  # Generate a color palette
#
#     for i, value in enumerate(NDCG_values):
#         merged.reset_index(inplace=True, drop=True)
#         row_index = int(value)
#
#         if row_index < merged.shape[0]:
#             row_data = merged.iloc[row_index].to_dict()
#             df = pd.DataFrame([row_data])
#             df = df.rename(index={0: 'NDCG at ' + str(value)})
#
#             df = df.T
#
#             sns.lineplot(data=df, ax=ax, palette=[palette[i]])
#
#             # Define the coordinates for the arrow
#             x_pos = plt.xlim()[1] + 0.4
#             # y_pos = plt.ylim()[0]
#
#     # plt.annotate('Higher is better', xy=(x_pos, plt.ylim()[0]),
#     #              xytext=(x_pos, plt.ylim()[1]), arrowprops=dict(facecolor='red', edgecolor='red', arrowstyle='<-'),
#     #              ha='center', annotation_clip=False)
#
#     # if gamma == 'Across gamma':
#     #     plt.xlabel("Gamma values")
#     #     ax.set_title('Dataset: ' + experiment_name + ', method: DELTR, GroundTruth', fontdict={'size': 20})
#     # else:
#     #     plt.xlabel("Wrong Inference percentage, gamma = " + str(gamma), fontdict={'size': 20})
#     # plt.ylabel("NDCG ", fontdict={'size': 20})
#
#     # if pipe == 'Inferred':
#     #     ax.set_title('Dataset: ' + experiment_name + ', method: DELTR', fontdict={'size': 20})
#     # elif pipe == 'DetConstSort':
#     #     ax.set_title('Dataset: ' + experiment_name + ', method: DetConstSort', fontdict={'size': 20})
#
#     plt.tick_params(axis='x', labelsize=40)
#     plt.tick_params(axis='y', labelsize=30)
#
#     for tick in ax.get_xticklabels() + ax.get_yticklabels():
#         tick.set_fontfamily('serif')
#
#     for tick in ax.get_xticklabels():
#         tick.set_weight('bold')
#
#     if len(NDCG_values) > 1:
#         plt.ylabel('NDCG', fontsize=50, **graph_font, weight='bold', labelpad=10)
#     else:
#         plt.ylabel('NDCG@' + str(NDCG_values[0]), fontsize=50, **graph_font, weight='bold')
#     plt.xlabel(' ')
#     # remove top boundary of the plot
#     # ax.spines['left'].set_visible(True)
#
#     # ax.legend().set_visible(False)
#     legend = ax.legend(loc='upper right')
#     legend.remove()
#
#     """ DIRECTORY MANAGEMENT """
#     graph_path = Path(
#         './FairRank/Graphs/flipchoice-' + flip_choice + '/' +
#         os.path.basename(settings["READ_FILE_SETTINGS"]["PATH"]).split('.')[
#             0] + "/" + pipe)
#
#     if not os.path.exists(graph_path):
#         os.makedirs(graph_path)
#
#     plt.savefig(os.path.join(graph_path, 'pos_' + str(value) + 'gamma_' + str(gamma) + '_ndcg.png'))
#     # combine plots on same graph
#     # plt.close()
#
#
# def plot_avg_exp():
#     paths = get_files('./FairRank/Results/' + experiment_name + '/' + flip_choice, None)
#
#     pipeline = list(set([re.split(regex_pattern, path)[5] for path in paths]))
#     # inferred_pipe = [p for p in paths if 'Inferred' in p]
#
#     groups = {protected_group, unprotected_group}
#
#     for group in groups:
#
#         for pipe in pipeline:
#             if pipe != 'Initial':
#                 exp_file_path = "FairRank/Graphs/flipchoice-" + flip_choice + "/" + experiment_name + "/" + pipe + '/' + pipe + "_ndkl.csv"
#
#                 if pipe == "GroundTruth":
#                     plot_avg_exp_graph(group, exp_file_path, inf='False', gamma='False', graph_id='GT')
#                 elif pipe == 'DetConstSort':
#                     plot_avg_exp_graph(group, exp_file_path, inf='False', gamma='True', graph_id='DetConstSort')
#                 else:
#                     exp_files = get_files('FairRank/Graphs/flipchoice-' + flip_choice + "/" + experiment_name, None)
#                     inf_files = [i for i in exp_files if i.endswith('.csv') and 'inference' in i]
#                     gamma_files = [i for i in exp_files if i.endswith('.csv') and 'Inferred_gamma' in i]
#
#                     for file in inf_files:
#                         plot_avg_exp_graph(group, file, inf='True', gamma='False',
#                                            graph_id=re.split(regex_pattern, file)[6])
#                     for file in gamma_files:
#                         plot_avg_exp_graph(group, file, inf='False', gamma='True',
#                                            graph_id='inferred_gamma=' + re.split(regex_pattern, file)[7])
#
#                 # plot_avg_exp_graph(group, exp_file_path, inf='True', gamma='False')
#                 # plot_avg_exp_graph(group, exp_file_path, inf='False', gamma='False')
#
#     return
#
#
# def plot_avg_exp_graph(group, exp_file_path, inf, gamma, graph_id):
#     exp_file = pd.read_csv(exp_file_path)
#     sns.set(style="whitegrid")
#
#     sns.color_palette("pastel")
#     fig, ax = plt.subplots(1, 1, figsize=(10, 10))
#     # ax.set_ylim(bottom=0.10)
#
#     if inf == 'True':
#
#         chart = sns.barplot(x='gamma_value', y='True Avg_Exp(' + group + ')',
#                             data=exp_file[["gamma_value", 'True Avg_Exp(' + group + ')']])
#         chart.set_title("Average Exposure for " + group + ' with' + graph_id + ' wrong inference',
#                         fontdict={'size': 20})
#
#         ax.set_xlabel("Gamma", fontsize=20)  # Set x label font size to 14
#         ax.set_ylabel("Average Exposure", fontsize=20)  # Set y label font size to 14
#
#     elif gamma == 'True':
#         chart = sns.barplot(x='inf_level', y='True Avg_Exp(' + group + ')',
#                             data=exp_file[["inf_level", 'True Avg_Exp(' + group + ')']])
#
#         chart.set_title(
#             "Average Exposure for " + group + " with GAMMA = " +
#             graph_id, fontdict={'size': 20})
#
#         ax.set_xlabel("Wrong Inference Percentage", fontsize=20)  # Set x label font size to 14
#         ax.set_ylabel("Average Exposure", fontsize=20)  # Set y label font size to 14
#
#     if inf == 'False' and gamma == 'False':
#         chart = sns.barplot(x='gamma_value', y='True Avg_Exp(' + group + ')',
#                             data=exp_file[["gamma_value", 'True Avg_Exp(' + group + ')']])
#
#         chart.set_title("Average Exposure for " + group + " across gamma values and groundtruth labels",
#                         fontdict={'size': 20})
#         ax.set_xlabel("Gamma values", fontsize=20)  # Set x label font size to 14
#         ax.set_ylabel("Average Exposure", fontsize=20)  # Set y label font size to 14
#
#     # ax.yaxis.set_major_locator(ticker.MultipleLocator(.005))
#
#     # Rotate the x-axis labels
#
#     ax.set_xticklabels(ax.get_xticklabels())
#     plt.tick_params(axis='x', labelsize=20)  # Increase x tick size to 12
#     plt.tick_params(axis='y', labelsize=20)  # Increase y tick size to 12
#
#     fig.tight_layout()
#
#     for label in (ax.get_xticklabels() + ax.get_yticklabels() +
#                   [ax.xaxis.label, ax.yaxis.label, ax.title]):
#         label.set_weight('bold')
#
#     """ DIRECTORY MANAGEMENT """
#     graph_path = Path(
#         './FairRank/Graphs/flipchoice-' + flip_choice + '/' +
#         os.path.basename(settings["READ_FILE_SETTINGS"]["PATH"]).split('.')[
#             0] + "/")
#
#     if not os.path.exists(graph_path):
#         os.makedirs(graph_path)
#
#     plt.savefig(os.path.join(graph_path, 'avg_exp_' + group + "_" + graph_id +
#                              '.png'))
#     plt.close()
#
#     if inf == 'False' and gamma == 'False':
#
#         joint_exp_file = exp_file.groupby('gamma_value')[
#             ['True Avg_Exp(Males)', 'True Avg_Exp(Females)']].mean().reset_index()
#         # print(joint_exp_file)
#         sns.set_style('whitegrid')
#         fig, ax = plt.subplots(1, 1, figsize=(10, 10))
#         ax = sns.barplot(x='gamma_value', y='value', hue='variable', data=pd.melt(joint_exp_file, ['gamma_value']),
#                          palette='bright')
#
#         ax.set(xlabel='Gamma values', ylabel='Average Exposure',
#                title=str('Average Exposure across Gamma Values for both groups, GT'))
#         # ax.legend(bbox_to_anchor=(1.05, .5), title="", loc="upper right")
#         ax.legend(title="", loc="upper right")
#         ax.set_ylim(bottom=0.10)
#         # Set the font size for the xlabel
#         ax.xaxis.label.set_fontsize(20)
#
#         # Set the font size for the ylabel
#         ax.yaxis.label.set_fontsize(20)
#
#         # Set the font size for the title
#         ax.title.set_fontsize(20)
#
#         plt.tick_params(axis='x', labelsize=15)  # Increase x tick size to 12
#         plt.tick_params(axis='y', labelsize=15)
#
#         plt.savefig(os.path.join(graph_path, 'avg_exp_both_groups' + graph_id + '.png'))
#
#         plt.close()
#
#     elif gamma == 'True':
#
#         joint_exp_file = exp_file.groupby('inf_level')[
#             ['True Avg_Exp(Males)', 'True Avg_Exp(Females)']].mean().reset_index()
#         # print(joint_exp_file)
#         sns.set_style('whitegrid')
#         fig, ax = plt.subplots(1, 1, figsize=(20, 10))
#         ax = sns.barplot(x='inf_level', y='value', hue='variable', data=pd.melt(joint_exp_file, ['inf_level']),
#                          palette='bright')
#         # if graph_id == 'DetConstSort':
#         #     ax.set(xlabel='Dataset: ' + experiment_name + ', method: DetConstSort', ylabel='Average Exposure',
#         #            title=str('Average Exposure across wrong inference levels for both groups'))
#         # else:
#         #      ax.set(xlabel='Dataset: ' + experiment_name + ', method: DELTR, ' + graph_id.split('_')[-1],
#         #             title=str('Average Exposure across wrong inference levels for both groups'))
#
#         legend = ax.legend(title="", loc="upper right")
#         legend.remove()
#         ax.set_ylim(bottom=0.10)
#         print(graph_id)
#
#         # ax.xaxis.label.set_fontsize(20)
#         #
#         # # Set the font size for the ylabel
#         # ax.yaxis.label.set_fontsize(20)
#
#         # Set the font size for the title
#         ax.title.set_fontsize(12)
#
#         plt.tick_params(axis='x', labelsize=40)
#         plt.tick_params(axis='y', labelsize=30)
#
#         for tick in ax.get_xticklabels() + ax.get_yticklabels():
#             tick.set_fontfamily('serif')
#
#         for tick in ax.get_xticklabels():
#             tick.set_weight('bold')
#
#         plt.ylabel('Average Exposure', fontsize=50, **graph_font, weight='bold', labelpad=10)
#         plt.xlabel(' ')
#         # remove top boundary of the plot
#         ax.spines['top'].set_visible(False)
#
#         plt.savefig(os.path.join(graph_path, 'avg_exp_both_groups' + graph_id + '.png'))
#
#         plt.close()
#
#     elif inf == 'True':
#
#         joint_exp_file = exp_file.groupby('gamma_value')[
#             ['True Avg_Exp(Males)', 'True Avg_Exp(Females)']].mean().reset_index()
#         # print(joint_exp_file)
#         sns.set_style('whitegrid')
#         fig, ax = plt.subplots(1, 1, figsize=(10, 10))
#         ax = sns.barplot(x='gamma_value', y='value', hue='variable', data=pd.melt(joint_exp_file, ['gamma_value']),
#                          palette='bright')
#         ax.set(xlabel='Wrong Inference Percentage', ylabel='Average Exposure',
#                title=str('Average Exposure across Gamma Values ' +
#                          graph_id + ' wrong inference'))
#         # ax.legend(bbox_to_anchor=(1.05, .5), title="", loc="upper right")
#         ax.legend(title="", loc="upper right")
#         ax.set_ylim(bottom=0.10)
#         ax.xaxis.label.set_fontsize(20)
#
#         # Set the font size for the ylabel
#         ax.yaxis.label.set_fontsize(20)
#
#         # Set the font size for the title
#         ax.title.set_fontsize(17)
#
#         plt.tick_params(axis='x', labelsize=15)  # Increase x tick size to 12
#         plt.tick_params(axis='y', labelsize=15)
#
#         plt.savefig(os.path.join(graph_path, 'avg_exp_' + graph_id + '.png'))
#
#         plt.close()
#
#
# def plot_ndkl(flip_choice):
#     paths = get_files('./FairRank/Results/seed42/' + experiment_name + '/' + flip_choice, None)
#     pipeline = list(set([re.split(regex_pattern, path)[6] for path in paths]))
#
#     for pipe in pipeline:
#         if pipe != 'Initial':
#
#             NDKL_PATH = './FairRank/Graphs/flipchoice-' + flip_choice + '/' + experiment_name + '/' + pipe + '/' + pipe + '_ndkl.csv'
#
#             with open(NDKL_PATH, 'w', newline='') as csvfile:
#                 csvwriter = csv.writer(csvfile)
#                 csvwriter.writerow(
#                     ['graph_name', 'gamma_value', 'True NDKL', 'False NDKL', 'True Avg_Exp(' + unprotected_group + ')',
#                      'True Avg_Exp(' + protected_group + ')', 'True PEx', 'False Avg_Exp(' + unprotected_group + ')',
#                      'False Avg_Exp(' + protected_group + ')', 'False PEx', 'inf_level', 'pipe', 'KendallTau',
#                      'NDCG@10',
#                      'NDCG@50'])
#
#             metrics = get_files('./FairRank/Results/seed42/' + experiment_name + '/' + flip_choice + '/', pipe)
#             res_metrics = [i for i in metrics if 'metrics' in i]
#
#             for file in res_metrics:
#                 print(file)
#                 make_metric_csv(file)
#
#             if pipe == "GroundTruth":
#                 plot_ndkl_graph(NDKL_PATH, pipe, inf='False', gamma='False', graph_id='GT')
#             elif pipe == "Inferred":
#                 plot_ndkl_inf(NDKL_PATH, pipe)
#             elif pipe == "DetConstSort":
#                 plot_ndkl_graph(NDKL_PATH, pipe, inf='False', gamma='True', graph_id='DetConstSort')
#             else:
#                 print('pipe =', pipe)
#                 #plot_ndkl_graph(NDKL_PATH, pipe, inf='False', gamma='True', graph_id='InitialTrain')
#
#     return
#
#
# def plot_ndkl_inf(path, pipe):
#     file = pd.read_csv(path)
#     grouped = file.groupby("inf_level")
#     grouped_by_gamma = file.groupby("gamma_value")
#     for inf_level, group in grouped:
#         new_dataframe = group.copy()
#         inf_path = os.path.dirname(path) + '/' + pipe + '_' + str(inf_level) + "%_inference_ndkl.csv"
#         new_dataframe.to_csv(inf_path)
#         plot_ndkl_graph(inf_path, pipe, inf='True', gamma='False', graph_id=str(inf_level) + "_Inferred")
#
#     for gamma_value, group in grouped_by_gamma:
#         new_dataframe = group.copy()
#         gamma_path = os.path.dirname(path) + '/' + pipe + "_gamma=" + str(gamma_value) + "_ndkl.csv"
#         new_dataframe.to_csv(gamma_path)
#         plot_ndkl_graph(gamma_path, pipe, inf='False', gamma='True', graph_id="gamma=" + str(gamma_value))
#
#
# def plot_ndkl_graph(path, pipe, inf, gamma, graph_id):
#     ndkl_file = pd.read_csv(path)
#     sns.set(style="whitegrid")
#     fig, ax = plt.subplots(1, 1, figsize=(15, 7.5))
#
#     # law, NBA, COMPAS
#     # ax.yaxis.set_major_locator(ticker.MultipleLocator(.0025))
#     # ax.yaxis.set_major_locator(ticker.MultipleLocator(.05))
#     # ax.yaxis.set_major_locator(ticker.MultipleLocator(.001))
#     if inf == 'True':
#         chart = sns.barplot(x='gamma_value', y='True NDKL', data=ndkl_file[['gamma_value', 'True NDKL']], width=0.5)
#
#         chart.set_title("NDKL across gamma values for " + re.split(regex_pattern, graph_id)[0] + "% wrong inference",
#                         fontdict={'size': 10})
#         ax.set_xlabel("Gamma values", fontsize=10)
#         ax.set_ylabel("NDKL", fontsize=10)
#     elif gamma == 'True':
#
#         chart = sns.barplot(x='inf_level', y='True NDKL', data=ndkl_file[['inf_level', 'True NDKL']])
#
#         # if pipe == 'DetConstSort':
#         #     chart.set_title('Dataset: ' + experiment_name + ', method: DetConstSort', fontdict={'size': 20})
#         if graph_id == 'InitialTrain':
#             chart.set_title('Dataset: ' + experiment_name + ', Initial Training Data', fontdict={'size': 20})
#         else:
#
#             # chart.set_title(
#             #     'Dataset: ' + experiment_name + ', method: DELTR, gamma = ' + re.split(regex_pattern, graph_id)[1],
#             #     fontdict={'size': 20})
#
#             # ax.set_xlabel("Wrong inference Percentage", fontsize=30)
#             # ax.set_ylabel("NDKL", fontsize=30)
#             plt.tick_params(axis='x', labelsize=40)
#             plt.tick_params(axis='y', labelsize=30)
#
#             for tick in ax.get_xticklabels() + ax.get_yticklabels():
#                 tick.set_fontfamily('serif')
#
#             # for tick in ax.get_xticklabels():
#             #     tick.set_weight('bold')
#             #
#             # for tick in ax.get_yticklabels():
#             #     tick.set_weight('normal')
#
#             plt.ylabel('NDKL', fontsize=38, **graph_font, weight='bold', labelpad=10)
#             plt.xlabel(' ')
#             # remove top boundary of the plot
#             ax.spines['top'].set_visible(False)
#
#     elif gamma == 'False' and inf == 'False':
#
#         chart = sns.barplot(x='gamma_value', y='True NDKL', data=ndkl_file[['gamma_value', 'True NDKL']])
#
#         chart.set_title("NDKL across gamma values for ground truth labels ",
#                         fontdict={'size': 10})
#         ax.set_xlabel("Gamma values", fontsize=30)
#         ax.set_ylabel("NDKL", fontsize=30)
#
#     plt.subplots_adjust(right=0.85, left=0.20)
#
#     x_pos = plt.xlim()[1] + 0.2
#     y_pos = plt.ylim()[0]
#
#     # Add the arrow and text label
#     # plt.annotate('Lower is better', xy=(x_pos, plt.ylim()[1]),
#     #              xytext=(x_pos, y_pos),
#     #              arrowprops=dict(facecolor='red', edgecolor='red', arrowstyle='<-'),
#     #              ha='center', annotation_clip=False)
#
#     # Rotate the x-axis labels
#     ax.set_xticklabels(ax.get_xticklabels())
#
#     plt.tick_params(axis='x', labelsize=30)  # Increase x tick size to 12
#     plt.tick_params(axis='y', labelsize=30)  # Increase y tick size to 12
#
#     # fig.tight_layout()
#
#     for label in (ax.get_xticklabels() + ax.get_yticklabels() +
#                   [ax.xaxis.label, ax.yaxis.label, ax.title]):
#         label.set_weight('bold')
#
#     """ DIRECTORY MANAGEMENT """
#     graph_path = Path(
#         './FairRank/Graphs/flipchoice-' + flip_choice + '/' +
#         os.path.basename(settings["READ_FILE_SETTINGS"]["PATH"]).split('.')[
#             0] + '/' + pipe + '/')
#
#     if not os.path.exists(graph_path):
#         os.makedirs(graph_path)
#
#     plt.savefig(
#         os.path.join(graph_path,
#                      graph_id + pipe + '_ndkl.png'))
#     plt.close()
#
#
# # Use this when you have no special naming style
# def get_graph_name(metrics_file_path):
#     graph_name = re.split(regex_pattern, metrics_file_path)
#     pipe = graph_name[5]
#     graph_title = ''
#
#     if pipe == 'Inferred':
#         print(flip_choice)
#         if flip_choice == "CaseStudies":
#             graph_title = str(graph_name[-4]) + ' ,GAMMA = ' + str(int(float(graph_name[-9]))) + ",Default = " + str(int(float(graph_name[-5])))
#         else:
#             graph_title = graph_name[-4] + '% Wrong Inference, GAMMA = ' + str(int(float(graph_name[-6])))
#     elif pipe == 'GroundTruth' or pipe == 'Colorblind':
#         if flip_choice == "CaseStudies":
#             graph_title = "GroundTruth" + ' ,GAMMA = ' + str(int(float(graph_name[-4])))
#         else:
#             graph_title = '0% Wrong Inference, GAMMA = ' + str(int(float(graph_name[-4])))
#     elif pipe == 'DetConstSort' and 'Inferred' in graph_name:
#         graph_title = graph_name[-4] + '% Wrong Inference'
#     elif pipe == 'DetConstSort' and 'Inferred' not in graph_name:
#         graph_title = '0 % Wrong Inference'
#         print('')
#     elif pipe == 'InitialTraining':
#         graph_title = 'Training data'
#     elif pipe == 'InitialTesting':
#         graph_title = 'Testing data'
#
#     return pipe, graph_title
#
#
# def make_metric_csv(metrics_file_path):
#     metrics_df = pd.read_csv(metrics_file_path)
#     pipeline, graph_title = get_graph_name(metrics_file_path)
#     graph_title_split = re.split(regex_pattern, graph_title)
#
#     gamma_value = graph_title_split[-1]
#     if flip_choice == "CaseStudies":
#         inf_level = graph_title_split[0]
#     elif gamma_value == 'data':
#         inf_level = 0
#     else:
#         print(graph_title_split)
#         print('here,  ' + pipeline)
#         inf_level = int(float(graph_title_split[0].replace('%', '')))
#
#     metrics_df = metrics_df.T
#     metrics_df.columns = metrics_df.iloc[0]
#     metrics_df.reset_index(inplace=True)
#     metrics_df.drop(index=metrics_df.index[0], axis=0, inplace=True)
#
#     metrics_df.columns = metrics_df.iloc[0]
#
#     sns.set(style="white")
#
#     True_NDKL = "{:.3f}".format(float(metrics_df.iloc[0, 0]))
#     if len(metrics_df.columns) >= 3:
#         if isinstance(metrics_df.iloc[0, 3], str):
#             False_NDKL = 0
#         else:
#             False_NDKL = "{:.3f}".format(float(metrics_df.iloc[0, 3]))
#     else:
#         # Handle the case where False_NDKL has insufficient elements
#         print("False_NDKL does not have enough elements.")
#         False_NDKL = 0
#     True_Avg_Exp = metrics_df.iloc[0, 1].strip("[]")
#     True_Avg_Exp = np.array(re.split(r'\s+', True_Avg_Exp.strip()))
#     True_exp_ratio = float(True_Avg_Exp[1]) / float(True_Avg_Exp[0])
#
#     if len(metrics_df.columns) >= 4:
#         False_Avg_Exp = metrics_df.iloc[0, 4].strip("[]")
#         False_Avg_Exp = np.array(re.split(r'\s+', False_Avg_Exp.strip()))
#         False_exp_ratio = float(False_Avg_Exp[-1]) / float(False_Avg_Exp[0])
#     else:
#         # Handle the case where False_Avg_Exp has insufficient elements
#         print("False_Avg_Exp does not have enough elements.")
#         False_Avg_Exp = np.array([0, 0])
#         False_exp_ratio = 0
#         # False_Avg_Exp[0] = 0
#         # False_Avg_Exp[1] = 0
#     # False_exp_ratio = float(False_Avg_Exp[1]) / float(False_Avg_Exp[0])
#     if len(metrics_df.columns) > 2:
#         kTau = metrics_df.iloc[0, 2]
#     else:
#         kTau = 0
#
#     # search for corresponding ndcg file
#     # NDKL_PATH = './FairRank/Graphs/flipchoice-' + flip_choice + '/' + experiment_name + '/' + pipe + '/' + pipe + '_ndkl.csv'
#     # ndcg_path = './FairRank/ResultsCSVS/collated_NDCG' + experiment_name + '
#     ndcg_file_path = metrics_file_path.replace('metrics', 'ndcg')
#     ndcg_file_path = metrics_file_path.replace('Results', 'ResultsCSVS')
#     #ndcg_file_path = metrics_file_path.replace('Results', 'ResultsCSVS')
#
#     ndcg_file = pd.read_csv(ndcg_file_path)
#
#     NDCG10 = ndcg_file.loc[10, 'NDCG']
#     NDCG50 = ndcg_file.loc[50, 'NDCG']
#
#     NDKL_PATH = 'FairRank/Graphs/flipchoice-' + flip_choice + '/' + experiment_name + '/' + pipeline + '/' + pipeline + '_ndkl.csv'
#
#     with open(NDKL_PATH, 'a', newline='') as csvfile:
#         csvwriter = csv.writer(csvfile)
#         csvwriter.writerow(
#             [graph_title, gamma_value, True_NDKL, False_NDKL, "{:.3f}".format(float(True_Avg_Exp[0])),
#              "{:.3f}".format(float(True_Avg_Exp[1])),
#              "{:.2f}".format(float(True_exp_ratio)), "{:.3f}".format(float(False_Avg_Exp[0])),
#              "{:.3f}".format(float(False_Avg_Exp[-1])),
#              "{:.2f}".format(float(False_exp_ratio)), inf_level, pipeline, "{:.3f}".format(float(kTau)),
#              "{:.3f}".format(float(NDCG10)), "{:.3f}".format(float(NDCG50))])
#
#     return
#
#
# def plot_skew(metrics_file_path, flip_choice):
#     metrics_file = pd.read_csv(metrics_file_path)
#
#     metrics_file['Ideal'] = 1
#     metrics_file.rename(columns={'Group 0': str(unprotected_group), 'Group 1': str(protected_group)}, inplace=True)
#     sns.set(style="white")
#     fig, ax = plt.subplots(1, 1, figsize=(10, 5))
#     sns.lineplot(data=metrics_file[[unprotected_group, protected_group, "Ideal"]], dashes=False, ax=ax)
#
#     pipe, graph_title = get_graph_name(metrics_file_path)
#
#     ax.set_title(graph_title + ", " + str(experiment_name), fontdict={'size': 15})
#
#     ax.title.set_position([0.5, -0.1])
#
#     # ax.set_xticks(range(0, len(metrics_file), 200))
#
#     # Set the x and y limits to start from 0
#     ax.set_xlim(left=0)
#     ax.set_ylim(bottom=0)
#
#     plt.xlabel("Ranking position")
#     plt.ylabel("Skew")
#
#     plt.legend()
#     check = metrics_file_path.split(os.sep)[-2].split("/")[-1]
#     check2 = metrics_file_path.split(os.sep)
#     """ DIRECTORY MANAGEMENT """
#     graph_path = Path(
#         "./FairRank/Graphs/flipchoice-" + flip_choice + "/" +
#         os.path.basename(settings["READ_FILE_SETTINGS"]["PATH"]).split('.')[
#             0] + "/" + pipe + "/" + str(metrics_file_path.split(os.sep)[-2].split("/")[-1]))
#
#     if not os.path.exists(graph_path):
#         os.makedirs(graph_path)
#
#     plt.savefig(os.path.join(graph_path, 'skews.png'))
#     plt.close()
#
#     return
#
#
def PlotLoss():
    loss_options = ['nonBlind', 'Blind']
    #experiment =
    for option in loss_options:
        loss_files = get_files("./FairRank/DELTRLoss/" + option + "/" + experiment_name, None)
        #loss_files = get_files("./FairRank/DELTRLoss/" + option, None)
        print('loss', loss_files)

        for file in loss_files:
            print(file)
            loss_df = pd.read_csv(file)
            sns.set(style="white")
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            chart = sns.lineplot(data=loss_df[["loss"]], legend=False)
            # ax.yaxis.set_major_locator(ticker.MultipleLocator(.25))
            # ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
            # plt.legend(loc="upper right", legend = False)
            # plt.ylim(4334, 4335)
            # plt.figure(figsize=(8, 6))
            plt.subplots_adjust(left=.25, bottom=0.15)

            delimiters = "(", ")"
            regex_pattern = '|'.join(map(re.escape, delimiters))
            graph_name = re.split(regex_pattern, file)
            #print('graph_name', graph_name[0].split('/')[4].replace("\\", ""))

            chart.set_title(graph_name[1].split(',')[1] + ", " + experiment_name+ ", " + option, fontdict={'size': 15})

            plt.xlabel("Iterations")
            plt.ylabel("DELTR Loss")
            # plt.legend(fontsize='large')

            """DIRECTORY MANAGEMENT"""
            graph_path = Path(
                "./FairRank/DELTRLoss/" + option + "/" + experiment_name + '/Graphs/Loss/'
            )

            if not os.path.exists(graph_path):
                os.makedirs(graph_path)
            plt.savefig(os.path.join(graph_path, os.path.basename(file) + '.png'))
            plt.close()


#
#
# def PlotLossExposure():
#     u_files = get_files("./FairRank/DELTRLoss/" + experiment_name, None)
#     for file in u_files:
#         print(file)
#         loss_df = pd.read_csv(file)
#         sns.set(style="white")
#         fig, ax = plt.subplots(1, 1, figsize=(5, 5))
#         chart = sns.lineplot(data=loss_df[["loss_exposure"]], legend=False)
#         # ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
#         # plt.legend(loc="upper right")
#         # plt.ylim(4800)
#         # plt.figure(figsize=(8, 6))
#         plt.subplots_adjust(left=.2, bottom=0.15)
#
#         delimiters = "(", ")"
#         regex_pattern = '|'.join(map(re.escape, delimiters))
#         graph_name = re.split(regex_pattern, file)
#
#         chart.set_title(graph_name[1].split(',')[1], fontdict={'size': 15})
#
#         plt.xlabel("Iterations")
#         plt.ylabel("U")
#         # plt.legend(fontsize='large')
#
#         """DIRECTORY MANAGEMENT"""
#         graph_path = Path(
#             "./FairRank/DELTRLoss/" + experiment_name + '/Graphs/U/'
#         )
#
#         if not os.path.exists(graph_path):
#             os.makedirs(graph_path)
#         plt.savefig(os.path.join(graph_path, os.path.basename(file) + '.png'))
#         plt.close()
#
#     return
#
#
# def PlotListLoss():
#     u_files = get_files("./FairRank/DELTRLoss/" + experiment_name, None)
#     for file in u_files:
#         print(file)
#         loss_df = pd.read_csv(file, encoding="utf-8")
#         sns.set(style="whitegrid")
#         fig, ax = plt.subplots(1, 1, figsize=(5, 5))
#         chart = sns.lineplot(data=loss_df[["loss_standard"]])
#         # ax.yaxis.set_major_locator(ticker.MultipleLocator(.1))
#
#         plt.legend(loc="upper right")
#         # plt.ylim(4800)
#         # plt.figure(figsize=(8, 6))
#         plt.subplots_adjust(left=.2, bottom=0.15)
#
#         delimiters = "(", ")"
#         regex_pattern = '|'.join(map(re.escape, delimiters))
#         graph_name = re.split(regex_pattern, file)
#
#         chart.set_title(graph_name[1], fontdict={'size': 15})
#
#         plt.xlabel("Iterations")
#         plt.ylabel("Listwise loss")
#         plt.legend(fontsize='large')
#
#         """DIRECTORY MANAGEMENT"""
#         graph_path = Path(
#             "./FairRank/DELTRLoss/" + experiment_name + '/Graphs/ListLoss/'
#         )
#
#         if not os.path.exists(graph_path):
#             os.makedirs(graph_path)
#         plt.savefig(os.path.join(graph_path, os.path.basename(file) + '.png'))
#         plt.close()
#
#     return
#
#
# def plot_posDiff():
#     base1 = ['./FairRank/Datasets/' + experiment_name + '/Ranked/' + flip_choice + "/" + str(
#         settings["GRAPH_OPTIONS"]["difference_base1"]) + '/']
#
#     base2 = ['./FairRank/Datasets/' + experiment_name + '/Ranked/' + flip_choice + "/" + str(
#         settings["GRAPH_OPTIONS"]["difference_base2"]) + '/']
#
#     bases = [base1, base2]
#
#     compare = ['./FairRank/Datasets/' + experiment_name + '/' + str(
#         settings["GRAPH_OPTIONS"]["difference_compare"]) + '/' + flip_choice + '/']
#
#     for base in bases:
#         for folder in base:
#             files = get_files(folder, None)
#             for file in files:
#                 cut = re.split(regex_pattern, file)
#
#                 base_pipe = cut[6]
#                 if base_pipe == "GroundTruth": base_pipe = "DELTR, gamma = " + str(get_string_after(cut, 'gamma'))
#                 df1 = pd.read_csv(file)
#                 for folder_2 in compare:
#                     files_to_compare = get_files(folder_2, None)
#
#                     for file_2 in files_to_compare:
#                         df2 = pd.read_csv(file_2)
#                         MAR, formatted_ARC, merged_df = MARC(df1, df2)
#
#                         fig, ax = plt.subplots(1, 1, figsize=(30, 7))
#                         chart = sns.histplot(merged_df['position_diff'])
#                         chart.set_title(
#                             str(re.split(regex_pattern, str(os.path.basename(file_2)))[-3] + '% Wrong Inference'),
#                             fontdict={'size': 30})
#
#                         ax.set_xlabel(
#                             'Positional Difference, base: ' + str(base_pipe) + ', MARC = ' + str(MAR) + ',ARC = ' + str(
#                                 formatted_ARC), fontsize=30)
#
#                         ax.set_ylabel("Frequency", fontsize=30)
#
#                         plt.subplots_adjust(left=.2, bottom=0.15)
#
#                         plt.tick_params(axis='x', labelsize=20)  # Increase x tick size to 12
#                         plt.tick_params(axis='y', labelsize=20)  # Increase y tick size to 12
#
#                         """DIRECTORY MANAGEMENT"""
#                         graph_path = Path(
#                             './FairRank/Graphs/flipchoice-' + flip_choice + '/' + experiment_name + '/PossDiff/base-' + str(
#                                 os.path.basename(file)) + '/')
#                         graph_path = str(graph_path)
#                         if not os.path.exists(graph_path):
#                             os.makedirs(graph_path, exist_ok=True)
#
#                         p = re.split(regex_pattern, os.path.basename(file_2))
#
#                         plt.savefig(os.path.join(graph_path, p[0] + p[6] + p[8] + '.png'))
#
#                         print('saved')
#                         plt.close()
#
#
# def plot_scoreDis():
#     files = get_files('./FairRank/Datasets/' + experiment_name + '/Ranked/' + flip_choice, None)
#     for file in files:
#         df = pd.read_csv(file)
#         pathsplit = re.split(regex_pattern, file)
#         print(pathsplit)
#
#         pipe = re.split(regex_pattern, file)[6]
#         # method = 'DetConstSort'
#         if pipe == 'Inferred' or pipe == 'GroundTruth':
#             method = 'DELTR'
#         else:
#             method = 'DetConstSort'
#
#         gamma = get_string_after(pathsplit, 'gamma')
#         inf = get_string_before(pathsplit, 'Inferred')
#
#         # fig, ax = plt.subplots(1, 1, figsize=(30, 50))
#         sns.displot(df['judgement'])
#         if pipe == 'GroundTruth':
#             plt.title('0% Wrong Inference')
#         else:
#             plt.title(
#                 str(re.split(regex_pattern, str(os.path.basename(file)))[-3] + '% Wrong Inference'))
#
#         plt.xlabel(
#             'method: ' + method + ', gamma = ' + gamma)
#
#         plt.ylabel("Score Distribution")
#
#         plt.subplots_adjust(top=.9)
#
#         plt.tick_params(axis='x')  # Increase x tick size to 12
#         plt.tick_params(axis='y')  # Increase y tick size to 12
#
#         """DIRECTORY MANAGEMENT"""
#         graph_path = Path(
#             './FairRank/Graphs/flipchoice-' + flip_choice + '/' + experiment_name + '/ScoreDis/' + str(
#                 os.path.basename(file)) + '/')
#         graph_path = str(graph_path)
#         if not os.path.exists(graph_path):
#             os.makedirs(graph_path, exist_ok=True)
#
#         p = re.split(regex_pattern, os.path.basename(file))
#
#         plt.savefig(os.path.join(graph_path, p[0] + p[6] + p[8] + '.png'))
#
#         print('saved')
#         plt.close()
#
#
def get_string_after(lst, target_string):
    for string in lst:
        if target_string == string:
            index = lst.index(target_string)
            if index < len(lst) - 1:
                return lst[index + 1]
    return None  # Return None if the target string is not found or is the last element


def get_string_before(lst, target_string):
    for string in lst:
        if target_string == string:
            index = lst.index(target_string)
            if index > 0:
                return lst[index - 1]
    return None  # Return None if the target string is not found or is the first element
