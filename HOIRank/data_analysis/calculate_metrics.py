import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from ..data_analysis import avgExp, NDCG, NDKL, skew, kendallTau
import csv
import os

with open('./HOIRank/settings.json', 'r') as f:
    settings = json.load(f)
experiment_name = os.path.basename(settings["READ_FILE_SETTINGS"]["PATH"]).split('.')[0]


# flip_choice = settings["DELTR_OPTIONS"]["flip_choice"]
# seed = settings["DELTR_OPTIONS"]["seed"]


def get_files(directory):
    temp = []
    for dirpath, dirnames, filenames in os.walk(directory):
        for file in filenames:
            match = re.search(experiment_name, file)
            if match:
                # temp.append(directory + '/' + file)
                temp.append(os.path.join(dirpath, file))
    return temp


def CalculateInitialMetrics(flip_choice):
    """ GET VARIABLES FROM SETTINGS """
    dataset_paths = ["./HOIRank/Datasets/" + experiment_name + "/Testing/Testing_" + experiment_name + ".csv",
                     "./HOIRank/Datasets/" + experiment_name + "/Training/Training_" + experiment_name + ".csv"]
    for dataset_path in dataset_paths:
        GT = pd.read_csv(dataset_path)
        print(GT)

        GT_ranking_ids = np.array(GT.iloc[:, 1])  # doc IDs
        GT_group_ids = np.array(GT.iloc[:, 2])  # protected attribute column
        score_col_index = GT.columns.get_loc(os.path.basename(settings["DELTR_OPTIONS"]["SCORE_COLUMN"]))
        GT_score = np.array(GT.iloc[:, score_col_index])  # numerical scoring column from settings.json
        # protected_attribute = settings["READ_FILE_SETTINGS"]["DEMO_COL"]  # TODO decide if automating this or not

        """ DIRECTORY MANAGEMENT """
        # results_path = Path(
        #     "./HOIRank/Results/" + os.path.basename(settings["READ_FILE_SETTINGS"]["PATH"]).split('.')[
        #         0] + "/" + flip_choice + "/InitialTraining")
        results_path = Path(
            "./HOIRank/ResultsInitial/" + os.path.basename(settings["READ_FILE_SETTINGS"]["PATH"]).split('.')[
                0] + "/Initial" + dataset_path.split('/')[4])
        if not os.path.exists(results_path):
            os.makedirs(results_path)

        """ CALCULATE NDKL """
        ndkl = "NDKL: ", NDKL(GT_ranking_ids, GT_group_ids)

        """ CALCULATE AVERAGE EXPOSURE """
        avg_exp = "Average Exposure: ", avgExp.avg_exp(GT_ranking_ids, GT_group_ids)

        """ CALCULATE Kendall's Tau """
        kendall_tau = "Kendall's TAU: ", kendallTau.kT(GT_ranking_ids, GT_ranking_ids)

        """ CALCULATE False NDKL """
        false_ndkl = "False NDKL: ", NDKL(GT_ranking_ids, GT_group_ids)

        """ CALCULATE False AVERAGE EXPOSURE """
        false_avg_exp = "False Average Exposure: ", avgExp.avg_exp(GT_ranking_ids, GT_group_ids)

        metrics_path = results_path / "initial_metrics.csv"
        with open(metrics_path, 'w') as f_metrics:
            print("Writing to metrics csv.")
            writer = csv.writer(f_metrics)
            writer.writerow(ndkl)
            writer.writerow(avg_exp)
            writer.writerow(kendall_tau)
            writer.writerow(false_ndkl)
            writer.writerow(false_avg_exp)

        """ CALCULATE SKEW """
        print("Calculating skew...")
        skew_path = results_path / "initial_skews.csv"
        skew_data = []
        for i in range(1, len(GT) + 1):
            skew_0 = skew(GT_ranking_ids, GT_group_ids, 0, i)
            skew_1 = skew(GT_ranking_ids, GT_group_ids, 1, i)
            skew_data.append([i, skew_0, skew_1])

        print("Finished calculating skews.")
        skew_header = ["Position", "Group 0", "Group 1"]
        with open(skew_path, 'w') as f_skew:
            print("Writing to skews csv.")
            writer = csv.writer(f_skew)
            # write the header
            writer.writerow(skew_header)

            # write the data
            writer.writerows(skew_data)

        print("Skews written to csv.")

        """ CALCULATE NDCG """
        print("Calculating NDCG...")
        ndcg_path = results_path / "initial_ndcg.csv"
        ndcg_data = []
        for i in range(1, len(GT) + 1):
            ndcg = NDCG(GT_ranking_ids, GT_score, i)
            ndcg_data.append([i, ndcg])

        print("Finished calculating NDCG.")
        ndcg_header = ["Position", "NDCG"]
        with open(ndcg_path, 'w') as f_ndcg:
            print("Writing to NDCG csv.")
            writer = csv.writer(f_ndcg)
            # write the header
            writer.writerow(ndcg_header)

            # write the data
            writer.writerows(ndcg_data)

        print("NDCG written to csv.")


def CalculateResultsMetrics(flip_choice, seed="no_seed"):
    ranked = get_files('./HOIRank/Datasets/' + experiment_name + '/Ranked/' + flip_choice + '/' + str(seed) + '/')

    gt_ranked = get_files('./HOIRank/Datasets/' + experiment_name + '/Ranked/' + flip_choice + '/' +
                          str(seed) + '/GroundTruth_Ranked/')

    blind_gt_ranked = get_files('./HOIRank/Datasets/' + experiment_name + '/Ranked/' + flip_choice + '/' + str(seed) +
                                '/BlindGroundTruth_Ranked/')
    # get the ground truth ranked files for DetConstSort

    dcs_gt_ranked = get_files(
        './HOIRank/Datasets/' + experiment_name + '/Ranked/' + flip_choice +
        '/' + str(seed) + '/DetConstSort_Ranked')

    for file in ranked:
        print(file)
        gamma = extract_string(file, 'gamma', ')')
        iteration = extract_string(file, 'iterations=', ',')

        current_gt = []
        current_dcs_gt = []
        current_dcs_nothidden_gt = []
        current_dcs_blind_gt = []
        current_blind_gt = []

        # filter through the ground truth ranked files to find the one that matches the current ranked file
        for gt_file in gt_ranked:
            if gamma in gt_file and iteration in gt_file:
                current_gt.append(gt_file)
        for dcs_file in dcs_gt_ranked:
            if gamma in dcs_file and iteration in dcs_file and '0_Inferred' in dcs_file and "DetConstSortHidden" in dcs_file:
                current_dcs_gt.append(dcs_file)
            elif gamma in dcs_file and iteration in dcs_file and '0_Inferred' in dcs_file and "DetConstSortNotHidden" in dcs_file:
                current_dcs_nothidden_gt.append(dcs_file)
            elif gamma in dcs_file and iteration in dcs_file and '0_Inferred' in dcs_file and "DetConstSortBlind" in dcs_file:
                current_dcs_blind_gt.append(dcs_file)
        for blind_file in blind_gt_ranked:
            if gamma in blind_file and iteration in blind_file:
                current_blind_gt.append(blind_file)

        if "DetConstSortHidden" in file:
            calc_metrics_util(file, current_dcs_gt[0], seed, flip_choice)
        elif "DetConstSortNotHidden" in file:
            calc_metrics_util(file, current_dcs_nothidden_gt[0], seed, flip_choice)
        elif "Blind" in file:
            if "DetConstSort" in file:
                calc_metrics_util(file, current_dcs_blind_gt[0], seed, flip_choice)

            # insert demographics into file
            temp_ranking = pd.read_csv(file)
            gt_data = pd.read_csv('./HOIRank/Datasets/' + experiment_name + '/Testing/Testing_' +
                                  experiment_name + '.csv', index_col=False)
            temp_ranking["Gender"] = temp_ranking['doc_id'].apply(
                lambda x: gt_data.loc[gt_data['doc_id'] == x, 'Gender'].iloc[0])
            temp_ranking['InferredGender'] = temp_ranking['Gender']
            temp_ranking.to_csv(file, index=False)

            calc_metrics_util(file, current_blind_gt[0], seed, flip_choice)
        else:
            calc_metrics_util(file, current_gt[0], seed, flip_choice)


def calc_metrics_util(dataset_path, gt_path, seedy, flip_choice):

    path_components = re.split(r'[\\/]', dataset_path)

    """ GET VARIABLES """
    ranking = pd.read_csv(dataset_path)
    rank_name = path_components[-1]
    gt_ranking = pd.read_csv(gt_path)
    print(rank_name)

    ranking_ids = np.array(ranking.iloc[:, 0])  # doc IDs
    group_ids = np.array(ranking.iloc[:, 1])  # protected attribute column
    inferred_group_ids = np.array(ranking.iloc[:, -1])  # inferred protected attribute column
    score_col_index = 2  # score column index
    GT_score_index = -2  # GT score column index
    score = np.array(ranking.iloc[:, score_col_index])  # numerical scoring column from settings.json
    gt_ids = np.array(gt_ranking.iloc[:, 0])  # doc IDs
    # we use the groundtruth score column to calculate NDCG
    GT_score = np.array(ranking.iloc[:, GT_score_index])  # numerical scoring column from settings.json
    # normalize GT_score to be between 0 and 1
    GT_score_normalized = (GT_score - np.min(GT_score)) / (np.max(GT_score) - np.min(GT_score))

    """ DIRECTORY MANAGEMENT """
    if flip_choice == "CaseStudies":
        results_path = Path(
            "./HOIRank/Results/" + "CaseStudies/" +
            os.path.basename(settings["READ_FILE_SETTINGS"]["PATH"]).split('.')[
                0]+ "/" + rank_name)
    else:
        results_path = Path(
            "./HOIRank/Results/" + "seed" + str(seedy) + "/" +
            os.path.basename(settings["READ_FILE_SETTINGS"]["PATH"]).split('.')[
                0] + "/" + flip_choice + "/" + rank_name)
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    """ CALCULATE True NDKL """
    ndkl = "True NDKL: ", NDKL(ranking_ids, group_ids)

    """ CALCULATE True AVERAGE EXPOSURE """
    avg_exp = "True Average Exposure: ", avgExp.avg_exp(ranking_ids, group_ids)

    """ CALCULATE Kendall's Tau """
    # kendall_tau = "Kendall's TAU: ", kendallTau.kT(ranking_ids, gt_ids)

    """ CALCULATE False NDKL """
    false_ndkl = "False NDKL: ", NDKL(ranking_ids, inferred_group_ids)

    """ CALCULATE False AVERAGE EXPOSURE """
    false_avg_exp = "False Average Exposure: ", avgExp.avg_exp(ranking_ids, inferred_group_ids)

    metrics_path = results_path / "metrics.csv"
    with open(metrics_path, 'w') as f_metrics:
        print("Writing to metrics csv.")
        writer = csv.writer(f_metrics)
        writer.writerow(ndkl)
        writer.writerow(avg_exp)
        #        writer.writerow(kendall_tau)
        writer.writerow(false_ndkl)
        writer.writerow(false_avg_exp)

    """ CALCULATE SKEW """
    print("Calculating skew...")
    skew_path = results_path / "skews.csv"
    skew_data = []
    for i in range(1, len(ranking) + 1):
        skew_0 = skew(ranking_ids, group_ids, 0, i)
        skew_1 = skew(ranking_ids, group_ids, 1, i)
        skew_data.append([i, skew_0, skew_1])

    print("Finished calculating skews.")
    skew_header = ["Position", "Group 0", "Group 1"]
    with open(skew_path, 'w') as f_skew:
        print("Writing to skews csv.")
        writer = csv.writer(f_skew)
        # write the header
        writer.writerow(skew_header)

        # write the data
        writer.writerows(skew_data)

    print("Skews written to csv.")

    """ CALCULATE NDCG """
    print("Calculating NDCG...")
    ndcg_path = results_path / "ndcg.csv"
    ndcg_data = []
    for i in range(1, len(ranking) + 1):
        ndcg = NDCG(ranking_ids, GT_score_normalized, i)
        ndcg_data.append([i, ndcg])

    print("Finished calculating NDCG.")
    ndcg_header = ["Position", "NDCG"]
    with open(ndcg_path, 'w') as f_ndcg:
        print("Writing to NDCG csv.")
        writer = csv.writer(f_ndcg)
        # write the header
        writer.writerow(ndcg_header)

        # write the data
        writer.writerows(ndcg_data)

    print("NDCG written to csv.")

    # """CALCULATE EXPOSURE RATIO"""
    # print("calculating exposure ratio")
    # exp_ratio_path = results_path / "expratio.csv"


def extract_string(input_string, start_string, end_string):
    """
    checks a longer string and returns the substring between the start and end string
    :param input_string: the longer string
    :param start_string: the string to start at
    :param end_string: the string to end at
    :return: substring between start and end string
    """
    start = input_string.find(start_string)
    end = input_string.find(end_string)
    return input_string[start + len(start_string):end]


def substring_match(longstring1, longstring2, substring):
    """
    checks 2 separate strings to see if a substring is present in both and in the same position having the same start and end strings,
    without supplying the start and end strings
    :param longstring1:
    :param longstring2:
    :param substring:
    :return:
    """
    start1 = longstring1.find(substring)
    end1 = longstring1.find(substring) + len(substring)
    start2 = longstring2.find(substring)
    end2 = longstring2.find(substring) + len(substring)
    if start1 == start2 and end1 == end2:
        return True


# Function to recursively search for matching filepaths
def find_matching_filepaths(root_dir, target_filename):
    matching_filepaths = []

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file == target_filename:
                matching_filepaths.append(os.path.join(root, file))

    return matching_filepaths


# Function to collate ndcg.csv files by column
def collate_ndcg_csv(matching_filepaths, output_file):
    data_columns = []

    # Read the data from each ndcg.csv file and store it in data_columns
    for filepath in matching_filepaths:
        with open(filepath, 'r') as input_csv:
            reader = csv.reader(input_csv)
            data_column = []

            # Read each column in the CSV
            for row in reader:
                data_column.append(row)

            data_columns.append(data_column)

    # Determine the maximum number of rows in the columns
    max_rows = max(len(column) for column in data_columns)

    # Transpose the data_columns to collate by column
    collated_data = []
    for row_index in range(max_rows):
        collated_row = []
        for column in data_columns:
            if row_index < len(column):
                collated_row.extend(column[row_index])
            else:
                collated_row.extend([''] * len(column[0]))  # Fill missing rows with empty strings
        collated_data.append(collated_row)

    # Write the collated data to the output file
    with open(output_file, 'w', newline='') as output_csv:
        writer = csv.writer(output_csv)
        writer.writerows(collated_data)


def collate_NDCGS():
    root_directory = "./HOIRank/Results"
    target_filename = "ndcg.csv"
    output_file = "collated_ndcg.csv"

    # Find matching filepaths in the subdirectories
    matching_filepaths = find_matching_filepaths(root_directory, target_filename)

    # Check if there are exactly 5 matching filepaths
    if len(matching_filepaths) == 5:
        # Collate the ndcg.csv files by column into a single file
        collate_ndcg_csv(matching_filepaths, output_file)
        print("Collation completed successfully.")
    else:
        print("Not exactly 5 matching filepaths found.")
