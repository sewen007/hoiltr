import pandas as pd
import os, json, re, csv
import chardet
import HOIRank.result_graphing.plot as plot
from pathlib import Path
import numpy as np

with open('./HOIRank/settings.json', 'r') as f:
    settings = json.load(f)
experiment_name = os.path.basename(settings["READ_FILE_SETTINGS"]["PATH"]).split('.')[0]
flip_choices = settings["DELTR_OPTIONS"]["flip_choices"]

delimiters = "_", "/", "\\", "(", ")", "=", ",", '\"', " "
regex_pattern = '|'.join(map(re.escape, delimiters))
casestudy_rowheader = ['Wrong Inference Percentage', 'ULTR', 'FLTR', 'ULTR + PostF', 'ULTRH + PostF', 'LTR + PostF']
simulation_rowheader = ['Wrong Inference Percentage', 'ULTR_1', 'ULTR_2', 'ULTR_3', 'FLTR_1', 'FLTR_2', 'FLTR_3',
                        'ULTR + PostF_1', 'ULTR + PostF_2', 'ULTR + PostF_3', 'ULTRH + PostF_1', 'ULTRH + PostF_2',
                        'ULTRH + PostF_3', 'LTR + PostF_1', 'LTR + PostF_2', 'LTR + PostF_3']


def get_files(directory, exclude_name=None):
    temp = []
    for dirpath, dirnames, filenames in os.walk(directory):
        for file in filenames:
            # check if exclude_name is None:
            if exclude_name is None:
                temp.append(os.path.join(dirpath, file))
            else:
                if exclude_name and exclude_name not in dirpath:
                    temp.append(os.path.join(dirpath, file))
    return temp


def get_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']


def CollateNDCGandSkews():
    """
    collates NDCG and Skew files from the 5 seeds into one csv file
    :return:
    """
    # create directory to save collated NDCG files
    collated_NDCG_path = "./HOIRank/ResultsCSVS/collatedNDCG/"
    if not os.path.exists(collated_NDCG_path):
        os.makedirs(collated_NDCG_path)

    seed_count = 0

    # access results path
    result_parent_directory_files = get_files("./HOIRank/Results/", exclude_name="CaseStudies")

    # split result in seeds
    result_seeds = list(set([re.split(regex_pattern, path)[3] for path in result_parent_directory_files]))

    for result_seed in result_seeds:
        seed_count += 1
        print("Collating NDCG and Skews files for seed " + str(seed_count) + "...")

        # select files that have the same seed and have ndcg in them
        ndcg_seed_files = [file for file in result_parent_directory_files if result_seed in file and "ndcg" in file and
                           experiment_name in file]
        skew_seed_files = [file for file in result_parent_directory_files if result_seed in file and "skews" in file]

        # collate the ndcg and skew files storing the average of the 5 seeds
        for ndcg_seed_file in ndcg_seed_files:
            collate_five_metrics(ndcg_seed_file, seed_count, collated_NDCG_path, metric='NDCG')
        # for skew_seed_file in skew_seed_files:
        #     collate_five_metrics(skew_seed_file, seed_count, collated_NDCG_path, metric='Skews')


def Make_Metric_Csvs():
    """
    makes empty csvs for each metric for each dataset collating Casestudies and Synthetic
    results
    :return:
    """

    go_make_empty_metrics(metric='NDKL')
    go_make_empty_metrics(metric='ExpR')
    go_make_empty_metrics(metric='NDCG10')
    go_make_empty_metrics(metric='NDCG50')
    go_make_empty_metrics(metric='NDCG100')
    go_make_empty_metrics(metric='AvgExp_0')
    go_make_empty_metrics(metric='AvgExp_1')
    # for ndkl and expr
    directories = ["./HOIRank/Results/seed42/", "./HOIRank/Results/seed123/", "./HOIRank/Results/seed789/",
                   "./HOIRank/Results/seed5678/", "./HOIRank/Results/seed9999/", "./HOIRank/Results/CaseStudies/"]
    # directories = ["./HOIRank/Results/CaseStudies/"]

    seed_count = 0
    for directory in directories:
        if 'seed' in directory:
            seed_count += 1
        print("Collating metrics for seed " + str(seed_count) + "...")
        # get all files in directory
        files = get_files(directory, exclude_name=None)
        # get metric files
        metric_files = [file for file in files if "metrics" in file and experiment_name in file]

        for metric_file in metric_files:
            go_store_NDKL(metric_file, seed_count)
            go_store_ExpR(metric_file, seed_count, None)
            go_store_ExpR(metric_file, seed_count, 0)
            go_store_ExpR(metric_file, seed_count, 1)
            # for collated NDCG paths. We want to do this once (as average NDCG is already derived), so we select
            # only seed42 (colud be any seed).
            if "CaseStudies" not in metric_file: # for synthetic
                # for LAW, we don't do case studies
                if "seed42" in metric_file:
                    ndcg_file = metric_file.replace('metrics', 'ndcg')
                    ndcg_file = ndcg_file.replace('Results', 'ResultsCSVs')
                    ndcg_file = ndcg_file.replace('seed42', 'collatedNDCG')
                    go_store_NDCG(ndcg_file, 50)
                    go_store_NDCG(ndcg_file, 100)
                    go_store_NDCG(ndcg_file, 10)

            else:  # for case studies
                ndcg_file = metric_file.replace('metrics', 'ndcg')
                # ndcg_file = ndcg_file.replace('Results', 'ResultsCSVs')
                go_store_NDCG(ndcg_file, 50)
                go_store_NDCG(ndcg_file, 100)
                go_store_NDCG(ndcg_file, 10)


def go_make_empty_metrics(metric, directory="./HOIRank/Results/seed42/"):
    # get all datasets
    # list_of_datasets = get_files(directory)
    # datasets = list(set([re.split(regex_pattern, path)[4] for path in list_of_datasets]))

    # make empty csvs for dataset
    # infer_choices = list(set([re.split(regex_pattern, path)[5] for path in list_of_datasets]))
    simulation_choices = ['CaseStudies', 'Synthetic']
    if experiment_name == 'LAW':
        simulation_choices = ['Synthetic']

    # for dataset in datasets:
    for simulation_choice in simulation_choices:
        csv_path = "./HOIRank/ResultsCSVs/" + simulation_choice + '_' + experiment_name + "_" + metric + ".csv"
        with open(csv_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            if simulation_choice == 'CaseStudies':
                csvwriter.writerow(casestudy_rowheader)
            else:
                csvwriter.writerow(simulation_rowheader)
    return


def go_store_NDKL(metric_file, count):
    # we need infer choice, dataset, pipeline
    # get infer choice
    # simulation_choice = infer_choice
    if 'seed' in metric_file:
        simulation_choice = "Synthetic"
        infer_choice = re.split(regex_pattern, metric_file)[5]
        # get pipeline
        pipeline = re.split(regex_pattern, metric_file)[6]
    else:
        simulation_choice = "CaseStudies"
        infer_choice = "CaseStudies"
        # get pipeline
        pipeline = re.split(regex_pattern, metric_file)[5]

    # get dataset
    dataset = re.split(regex_pattern, metric_file)[4]

    print('pipeline: ' + pipeline)
    if "GroundTruth" in pipeline or pipeline == "DetConstSort" or pipeline == "Colorblind" or pipeline == "BlindGroundTruth":
        # do nothing : because ground truth is already contained in each inferred, detconstsort is now either hidden
        # or not and colorblind is a line drawn on its own. BlindGroundtruth will also be a line drawn on its own
        pass
    else:
        # get col NDKL value

        col, inf = go_get_col(pipeline, infer_choice, metric_file)
        write_path = "./HOIRank/ResultsCSVs/" + simulation_choice + '_' + dataset + "_NDKL.csv"

        new_value = get_NDKL(metric_file)

        if simulation_choice != 'CaseStudies':
            old_value = get_previous_seed_value(write_path, inf, col)

            update_value = updated_value(old_value, new_value, count)

            new_data = {'Wrong Inference Percentage': inf, col: update_value}
        else:
            new_data = {'Wrong Inference Percentage': inf, col: new_value}
        existing_data = []

        with open(write_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)

            for row in reader:
                # this one gives continuity
                existing_data.append(row)
            # without this line, we only get header
            existing_data.append(new_data)

        for row in existing_data:
            if row['Wrong Inference Percentage'] == inf:
                # next 2 lines gives us full rows
                row[col] = get_NDKL(metric_file)
                row.update(new_data)
                break
        write_row(write_path, existing_data, simulation_choice)


def updated_value(old_value, new_value, count):
    new_value = new_value
    if count == 1:
        return new_value
    else:  # path exist, count>1
        if count < 5:
            return float(str(old_value)) + float(str(new_value))
        elif count == 5:
            return (float(str(old_value)) + float(str(new_value))) / 5


def get_previous_seed_value(file_path, row_name, col_name):
    # Read the CSV file
    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        rows = list(reader)

    # Ensure the file is not empty and has a header row
    if not rows:
        raise ValueError("The CSV file is empty.")

    # Find the column index by the column name
    header = rows[0]
    if col_name not in header:
        return 0
    col_index = header.index(col_name)

    # Find the row index by the row name
    row_index = None
    for i, row in enumerate(rows[1:], start=1):  # Skip header row
        if row[0] == row_name:
            row_index = i
            break

    if row_index is None:
        return 0

    # Get the old value at the specified position
    old_value = rows[row_index][col_index]

    # Return 0 if the old value is empty or does not exist
    if old_value == '':
        return 0
    else:
        return float(old_value)


def write_row(write_path, existing_data, simulation_choice):
    with open(write_path, 'w', newline='') as csvfile:
        if simulation_choice == 'CaseStudies':
            fieldnames = ['Wrong Inference Percentage', 'ULTR', 'FLTR', 'ULTR + PostF', 'ULTRH + PostF', 'LTR + PostF']

        else:

            fieldnames = ['Wrong Inference Percentage', 'ULTR_1', 'ULTR_2', 'ULTR_3', 'FLTR_1', 'FLTR_2', 'FLTR_3',
                          'ULTR + PostF_1', 'ULTR + PostF_2', 'ULTR + PostF_3', 'ULTRH + PostF_1', 'ULTRH + PostF_2',
                          'ULTRH + PostF_3', 'LTR + PostF_1', 'LTR + PostF_2', 'LTR + PostF_3']
        csvwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
        csvwriter.writeheader()
        csvwriter.writerows(existing_data)


def get_NDKL(metric_file):
    # get NDKL value
    print(metric_file)
    with open(metric_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if 'True NDKL' in line:
                # return value next column
                columns = line.split()
                if len(columns) > 1:
                    return "{:.3f}".format(float(re.split(',', columns[2])[1]))
                else:
                    pass


def get_ExpR(metric_file, group=None):
    # get ExpR value
    print(metric_file)
    with open(metric_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if 'True Average Exposure' in line:
                # return value next column
                columns = line.split()
                if len(columns) > 1:
                    ans_0 = columns[3].split('[')[1]
                    ans_1 = columns[4].split(']')[0]
                    if group == 0:
                        return "{:.3f}".format(float(ans_0))
                    elif group == 1:
                        return "{:.3f}".format(float(ans_1))
                    else:
                        return "{:.3f}".format(float(ans_1) / float(ans_0))

                else:
                    pass


def go_store_ExpR(metric_file, count, group=None):
    # go_make_empty_metrics(metric='ExpR')
    # we need infer choice, dataset, pipeline
    # get infer choice
    if 'seed' in metric_file:
        simulation_choice = "Synthetic"
        infer_choice = re.split(regex_pattern, metric_file)[5]
        # get pipeline
        pipeline = re.split(regex_pattern, metric_file)[6]
    else:
        simulation_choice = "CaseStudies"
        infer_choice = "CaseStudies"
        # get pipeline
        pipeline = re.split(regex_pattern, metric_file)[5]
    dataset = re.split(regex_pattern, metric_file)[4]
    # get pipeline

    print('pipeline: ' + pipeline)
    if "GroundTruth" in pipeline or pipeline == "DetConstSort" or pipeline == "Colorblind" or pipeline == "BlindGroundTruth":
        # do nothing
        pass
    else:
        # get col value
        col, inf = go_get_col(pipeline, infer_choice, metric_file)
        if group is not None:
            write_path = "./HOIRank/ResultsCSVs/" + simulation_choice + '_' + dataset + "_AvgExp_" + str(
                group) + ".csv"
        else:
            write_path = "./HOIRank/ResultsCSVs/" + simulation_choice + '_' + dataset + "_ExpR.csv"

        new_value = get_ExpR(metric_file, group)

        if simulation_choice == 'Synthetic':
            old_value = get_previous_seed_value(write_path, inf, col)
            update_value = updated_value(old_value, new_value, count)
            new_data = {'Wrong Inference Percentage': inf, col: update_value}
        else:
            new_data = {'Wrong Inference Percentage': inf, col: new_value}
        existing_data = []
        with open(write_path, 'r') as csvfile:

            reader = csv.DictReader(csvfile)
            for row in reader:
                existing_data.append(row)
            existing_data.append(new_data)
        for row in existing_data:
            if row['Wrong Inference Percentage'] == inf:
                row[col] = get_ExpR(metric_file, group)
                row.update(new_data)
                break

        write_row(write_path, existing_data, simulation_choice)


def go_get_col(pipeline, infer_choice, metric_file):
    file_split = re.split(regex_pattern, metric_file)
    print(file_split)
    col = None
    if pipeline == 'Colorblind':
        inf = 'colorblind'
        if infer_choice == 'both':
            col = 'ULTRH_1'
        elif infer_choice == 'protected':
            col = 'ULTRH_2'
        elif infer_choice == 'unprotected':
            col = 'ULTRH_3'
        elif infer_choice == 'CaseStudies':
            col = 'ULTRH'
    elif pipeline == 'GroundTruth':
        inf = ''
        col = ''
    else:
        inf = file_split[-4]
        if pipeline == 'DetConstSortHidden':
            if infer_choice == 'both':
                col = 'ULTRH + PostF_1'
            elif infer_choice == 'protected':
                col = 'ULTRH + PostF_2'
            elif infer_choice == 'unprotected':
                col = 'ULTRH + PostF_3'
            elif infer_choice == 'CaseStudies':
                col = 'ULTRH + PostF'
        elif pipeline == 'DetConstSortNotHidden':
            if infer_choice == 'both':
                col = 'ULTR + PostF_1'
            elif infer_choice == 'protected':
                col = 'ULTR + PostF_2'
            elif infer_choice == 'unprotected':
                col = 'ULTR + PostF_3'
            elif infer_choice == 'CaseStudies':
                col = 'ULTR + PostF'
                print('col: ', col)
        elif pipeline == 'DetConstSortBlind':
            if infer_choice == 'both':
                col = 'LTR + PostF_1'
            elif infer_choice == 'protected':
                col = 'LTR + PostF_2'
            elif infer_choice == 'unprotected':
                col = 'LTR + PostF_3'
            elif infer_choice == 'CaseStudies':
                col = 'LTR + PostF'
        elif pipeline == 'Inferred':
            # get gamma_value
            file_split = re.split(regex_pattern, metric_file)
            gamma_value = int(float(plot.get_string_after(file_split, 'gamma')))
            if infer_choice == 'both' and gamma_value == 0:
                col = 'ULTR_1'
            elif infer_choice == 'protected' and gamma_value == 0:
                col = 'ULTR_2'
            elif infer_choice == 'unprotected' and gamma_value == 0:
                col = 'ULTR_3'
            elif infer_choice == 'CaseStudies' and gamma_value == 0:
                col = 'ULTR'
            elif infer_choice == 'both' and gamma_value != 0:
                col = 'FLTR_1'
            elif infer_choice == 'protected' and gamma_value != 0:
                col = 'FLTR_2'
            elif infer_choice == 'unprotected' and gamma_value != 0:
                col = 'FLTR_3'
            elif infer_choice == 'CaseStudies' and gamma_value != 0:
                col = 'FLTR'
        else:
            print("pipeline not found")

    return col, inf


def go_store_NDCG(metric_file, value):
    # we need infer choice, dataset, pipeline
    # get infer choice
    # go_make_empty_metrics(metric='NDCG' + str(value))
    dataset = re.split(regex_pattern, metric_file)[4]
    if 'seed' in metric_file or 'collatedNDCG' in metric_file:
        simulation_choice = "Synthetic"
        infer_choice = re.split(regex_pattern, metric_file)[5]
        # get pipeline
        pipeline = re.split(regex_pattern, metric_file)[6]
    else:
        simulation_choice = "CaseStudies"
        infer_choice = "CaseStudies"
        # get pipeline
        pipeline = re.split(regex_pattern, metric_file)[5]
        if pipeline == 'both' or pipeline == 'protected' or pipeline == 'unprotected':
            pipeline = re.split(regex_pattern, metric_file)[6]

    # get pipeline
    # pipeline = re.split(regex_pattern, metric_file)[6]
    print('pipeline: ' + pipeline)
    if "GroundTruth" in pipeline or pipeline == "DetConstSort" or pipeline == "Colorblind" or pipeline == "BlindGroundTruth":
        # do nothing
        pass
    else:
        # get col value
        col, inf = go_get_col(pipeline, infer_choice, metric_file)
        write_path = "./HOIRank/ResultsCSVs/" + simulation_choice + '_' + dataset + "_NDCG" + str(value) + ".csv"
        # write_path = "./HOIRank/ResultsCSVs/" + dataset + "_NDCG.csv"
        new_data = {'Wrong Inference Percentage': inf, col: get_NDCG(metric_file, value)}
        existing_data = []
        with open(write_path, 'r') as csvfile:

            reader = csv.DictReader(csvfile)
            for row in reader:
                existing_data.append(row)
            existing_data.append(new_data)
        for row in existing_data:
            if row['Wrong Inference Percentage'] == inf:
                row[col] = get_NDCG(metric_file, value)
                row.update(new_data)
                break

        write_row(write_path, existing_data, simulation_choice)


def get_NDCG(metric_file, value):
    # get NDCG value
    ndcg_file = pd.read_csv(metric_file)
    NDCG = ndcg_file.loc[value, 'NDCG']
    return NDCG


def collate_five_metrics(file, count, collated_path, metric='NDCG'):
    # print('count = ', count)
    # read file containing values
    df = pd.read_csv(file)
    dirname = os.path.dirname(file)
    filename_split = re.split(regex_pattern, file)
    write_path = Path(
        collated_path + filename_split[4] + "/" + filename_split[5] + "/" + os.path.basename(dirname))

    if not os.path.exists(write_path):  # count=1
        os.makedirs(write_path)
    if count == 1:
        write_path = str(write_path) + "/" + str(metric).lower() + ".csv"
        if metric == 'NDCG':
            write_df = df[['Position', 'NDCG']].copy()
        elif metric == 'Skews':
            write_df = df[['Position', 'Group 0', 'Group 1']].copy()
        write_df.to_csv(write_path, index=False)
    else:  # path exist, count>1
        write_path = str(write_path) + "/" + str(metric).lower() + ".csv"
        # read file
        write_df = pd.read_csv(write_path)
        # if count>1 and < 5
        if count < 5:
            # read NDCG column and create new items as an average of previous
            # write_df['NDCG'] = write_df['NDCG'] + df['NDCG']
            for col in df.columns[1:]:
                write_df[col] = write_df[col] + df[col]
            # write_df.to_csv(write_path, index=False)
        else:  # count = 5
            for col in df.columns[1:]:
                write_df[col] = (write_df[col] + df[col]) / 5
            # write_df['NDCG'] = (write_df['NDCG'] + ndcg_df['NDCG']) / 5
        write_df.to_csv(write_path, index=False)
