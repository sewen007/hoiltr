import csv
import re
import json
import pickle
import numpy as np
import pandas as pd
import os

with open('./HOIRank/settings.json', 'r') as f:
    settings = json.load(f)

experiment_name = os.path.basename(settings["READ_FILE_SETTINGS"]["PATH"]).split('.')[0]
# flip_choice = settings["DELTR_OPTIONS"]["flip_choice"]
filename = os.path.basename(settings["READ_FILE_SETTINGS"]["PATH"]).split('.')[0]


def get_files(directory):
    temp = []
    for dirpath, dirnames, filenames in os.walk(directory):
        for file in filenames:
            match = re.search(experiment_name, file)
            if match:
                # temp.append(directory + '/' + file)
                temp.append(os.path.join(dirpath, file))
    return temp


models = get_files('./HOIRank/Models/' + experiment_name)
blind_models = get_files('./HOIRank/BlindModels/' + experiment_name)
gt = get_files('./HOIRank/Datasets/' + experiment_name + '/Testing')


def writeRanked(writefile, dict):
    # field names
    fields = ['doc_id', 'Gender', 'judgement', 'GT_score', 'InferredGender']

    # writing to csv file
    with open(writefile, 'w') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)

        # writing the fields
        csvwriter.writerow(fields)

        for player in dict.keys():
            csvwriter.writerow(dict.get(player))
    print("SUCCESS! Saved to: " + writefile)


def RankGroundTruth(flip_choice):
    write_path = './HOIRank/Datasets/' + filename + '/Ranked/' + flip_choice + '/GroundTruth_Ranked'
    blind_write_path = './HOIRank/Datasets/' + filename + '/Ranked/' + flip_choice + '/BlindGroundTruth_Ranked'
    if not os.path.exists(write_path):
        os.makedirs(write_path)
    write_path = write_path + '/GroundTruth_Ranked('

    if not os.path.exists(blind_write_path):
        os.makedirs(blind_write_path)
    blind_write_path = blind_write_path + '/BlindGroundTruth_Ranked('

    rank_not_inferred(gt, models, write_path)
    rank_not_inferred(gt, blind_models, blind_write_path, blind=True)


def rank_not_inferred(files, predictors, path, uniform=False, blind=False):
    for file in files:
        print(file)
        print(models)
        for model in predictors:
            print(model)
            params = re.findall(r"\(([^)]+)\)", model)[0]
            print(params)
            write_file = path + params + ')_' + \
                         os.path.basename(settings["READ_FILE_SETTINGS"]["PATH"]).split('.')[0] + '.csv'
            print("Ranking File: " + file + " With Model: " + model)
            print("Results will be saved to: " + write_file)
            filehandler = open(model, 'rb')
            DELTR = pickle.load(filehandler)

            test_data = pd.read_csv(file, index_col=False)

            # convert Gender to 1 if blind model
            if blind:
                test_data['Gender'] = 1

            print('test_data = ', test_data)
            print("Testing file size: " + str(test_data.shape))
            numeric_cols = list(test_data.select_dtypes(include=[np.number]).columns.values)
            formatted_data = test_data[numeric_cols]
            print("data will be ranking using", numeric_cols)

            score_column = settings["DELTR_OPTIONS"]["SCORE_COLUMN"]
            lower_better = settings["READ_FILE_SETTINGS"]["LOWER_SCORE_BETTER"].lower()
            normalized = settings["DELTR_OPTIONS"]["NORMALIZE_SCORE_COLUMN"].lower()

            # TODO remove lower better
            if lower_better == "true" and normalized == "true":
                formatted_data['normalized_score'] = formatted_data.apply(
                    lambda row: (formatted_data[score_column].max() - row[score_column]) / (
                            formatted_data[score_column].max() - formatted_data[score_column].min()), axis=1)
                formatted_data = formatted_data.drop(score_column, axis=1)
            elif lower_better == "false" and normalized == "true":
                formatted_data['normalized_score'] = formatted_data.apply(
                    lambda row: (row[score_column] - formatted_data[score_column].min()) / (
                            formatted_data[score_column].max() - formatted_data[score_column].min()), axis=1)
                formatted_data = formatted_data.drop(score_column, axis=1)

            # not sure why this was included here. Why are we sampling?
            # formatted_data = formatted_data.sample(frac=1)

            print('formatted data', formatted_data)

            result = DELTR.rank(formatted_data, has_judgment=True)

            # include ground truth score for NDCG calculation later
            gt_data = pd.read_csv('./HOIRank/Datasets/' + experiment_name + '/Testing/Testing_' +
                                  experiment_name + '.csv', index_col=False)
            result["GT_score"] = result['doc_id'].apply(
                lambda x: gt_data.loc[gt_data['doc_id'] == x, score_column].iloc[0])
            result['InferredGender'] = result['Gender']

            # if we had used colorblind ranking, now return groundtruth Gender
            if uniform:
                result['Gender'] = result['doc_id'].apply(
                    lambda x: gt_data.loc[gt_data['doc_id'] == x, "Gender"].iloc[0])
            print("SUCCESS! Saved to: " + write_file)
            result.to_csv(write_file, index=False)


def RankInferred(flip_choice):
    # Create the path to save the ranked files
    write_path_base = './HOIRank/Datasets/' + filename + '/Ranked/' + flip_choice + '/Inferred_Ranked/'
    if not os.path.exists(write_path_base):
        os.makedirs(write_path_base)

    inferred_path = './HOIRank/Datasets/' + experiment_name + '/Inferred/' + flip_choice + '/'

    inferred_directories = [content for content in os.listdir(inferred_path) if
                            os.path.isdir(os.path.join(inferred_path, content))]

    for directory in inferred_directories:
        write_path = os.path.join(write_path_base, directory)
        if not os.path.exists(write_path):
            os.makedirs(write_path)

        inferred = get_files(os.path.join(inferred_path, directory))  # Assuming get_files is defined elsewhere
        gt_dict = {}

        for gt_file in gt:
            for index, row in pd.read_csv(gt_file).iterrows():
                if int(row["doc_id"]) not in gt_dict:
                    gt_dict[int(row["doc_id"])] = int(row["Gender"])
                else:
                    print("There are duplicates in the ranking. Something went wrong.")
                    exit(1)

        for file in inferred:
            for model in models:
                params = re.findall(r"\(([^)]+)\)", model)[0]
                write_file = os.path.join(write_path,
                                          'Inferred_Ranked(' + params + ')_' + os.path.basename(file).split('.')[
                                              0] + '.csv')
                print("Ranking Inferred File: " + file + " With Model: " + model)
                print("Results will be saved to: " + write_file)

                with open(model, 'rb') as filehandler:
                    DELTR = pickle.load(filehandler)

                test_data = pd.read_csv(file)
                numeric_cols = list(test_data.select_dtypes(include=[np.number]).columns.values)
                test_data.drop(columns=[col for col in test_data if col not in numeric_cols], inplace=True)

                if flip_choice == "CaseStudies":
                    test_data.drop("Gender", axis=1, inplace=True)
                    test_data.rename(columns={"InferredGender": "Gender"}, inplace=True)

                score_column = settings["DELTR_OPTIONS"]["SCORE_COLUMN"]
                lower_better = settings["READ_FILE_SETTINGS"]["LOWER_SCORE_BETTER"].lower()
                normalized = settings["DELTR_OPTIONS"]["NORMALIZE_SCORE_COLUMN"].lower()

                if lower_better == "true" and normalized == "true":
                    test_data['normalized_score'] = test_data.apply(
                        lambda row: (test_data[score_column].max() - row[score_column]) / (
                                    test_data[score_column].max() - test_data[score_column].min()), axis=1)
                    back_up_test = test_data.copy()
                    test_data = test_data.drop(score_column, axis=1)
                elif lower_better == "false" and normalized == "true":
                    test_data['normalized_score'] = test_data.apply(
                        lambda row: (row[score_column] - test_data[score_column].min()) / (
                                    test_data[score_column].max() - test_data[score_column].min()), axis=1)
                    back_up_test = test_data.copy()
                    test_data = test_data.drop(score_column, axis=1)
                    test_data = test_data.drop("normalized_score", axis=1)

                result = DELTR.rank(test_data, has_judgment=False)
                result["GT_score"] = result['doc_id'].apply(
                    lambda x: back_up_test.loc[back_up_test['doc_id'] == x, score_column].iloc[0])

                gt_inferred_combined_dict = {}
                for index, row in result.iterrows():
                    gt_inferred_combined_dict[row["doc_id"]] = [str(int(row["doc_id"])),
                                                                str(gt_dict.get(int(row["doc_id"]), "N/A")),
                                                                str(row["judgement"]), str(row["GT_score"]),
                                                                str(int(row["Gender"]))]

                writeRanked(write_file, gt_inferred_combined_dict)


def RankColorblind(flip_choice):
    # create the path to save the ranked files
    write_path = './HOIRank/Datasets/' + filename + '/Ranked/' + flip_choice + '/Colorblind_Ranked/'
    if not os.path.exists(write_path):
        os.makedirs(write_path)
    for file in gt:
        # make files with uniform gender
        # make_groundtruth_sex_uniform(file, flip_choice, 0)
        make_groundtruth_sex_uniform(file, flip_choice, 1)
    # get files with uniform gender
    test_uniform_files = get_files(
        './HOIRank/Datasets/' + experiment_name + '/Transit/' + flip_choice + '/Testing_UniformSex/')

    write_path = write_path + '/Colorblind_Ranked('

    # rank files
    rank_not_inferred(test_uniform_files, models, write_path, uniform=True)

    return


def make_groundtruth_sex_uniform(path, flip_choice, sex=0):
    """
    This function takes a file and changes all ground truth sex to 0's or 1's
    :param path:
    :param sex:
    :return:
    """
    write_path = './HOIRank/Datasets/' + experiment_name + '/Transit/' + flip_choice + '/Testing_UniformSex/'

    if not os.path.exists(write_path):
        os.makedirs(write_path)

    background_file = pd.read_csv(path)

    # store uniform gender as Gender
    background_file.loc[:, 'Gender'] = sex
    background_file.to_csv(write_path + '_' + str(int(sex)) + '_' + os.path.basename(path), index=False)
