import csv
import json
import os
import pickle
import time
import config

import numpy as np
from fairsearchdeltr import Deltr
import pandas as pd

with open('./HOIRank/settings.json', 'r') as f:
    settings = json.load(f)

experiment_name = os.path.basename(settings["READ_FILE_SETTINGS"]["PATH"]).split('.')[0]
protected_group = config.protected_group


def Train():
    START = time.time()

    filename = os.path.basename(settings["READ_FILE_SETTINGS"]["PATH"]).split('.')[0]
    read_file = './HOIRank/Datasets/' + filename + '/Training/' + 'Training_' + filename + '.csv'

    if not os.path.isfile(read_file):
        print("This file: " + read_file + "does not exist, check read file options in settings.json")
        return

    train_data = pd.read_csv(read_file)

    # reorder doc_ids
    train_data['rearranged_id'] = range(1, len(train_data['doc_id']) + 1)
    train_data['doc_id'] = train_data['rearranged_id']

    # reset index - not important as we do not use the index
    train_data = train_data.drop('rearranged_id', axis=1)

    # Only get the numeric colums in the training dataset and make a new dataframe
    numeric_cols = list(train_data.select_dtypes(include=[np.number]).columns.values)
    print("numeric columns:", numeric_cols)
    formatted_data = train_data[numeric_cols].copy()
    formatted_data = formatted_data.reset_index(drop=True)

    score_column = settings["DELTR_OPTIONS"]["SCORE_COLUMN"]

    # lower_better will always be false. May remove
    # TODO
    lower_better = settings["READ_FILE_SETTINGS"]["LOWER_SCORE_BETTER"].lower()
    normalized = settings["DELTR_OPTIONS"]["NORMALIZE_SCORE_COLUMN"].lower()
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

    # set up the DELTR object, # column name of the protected attribute (index after query and document id)
    if settings["READ_FILE_SETTINGS"]["GENDER_EXPERIMENT"].lower() == "true":
        protected_attribute = "Gender"
    elif settings["READ_FILE_SETTINGS"]["RACE_EXPERIMENT"].lower() == "true":
        protected_attribute = "Race"
    gamma = settings["DELTR_OPTIONS"]["gamma"]  # list of the gamma parameters
    number_of_iterations = settings["DELTR_OPTIONS"][
        "num_iterations"]  # number of iterations the training should run
    standardize = True if settings["DELTR_OPTIONS"][
                              "standardize"].lower() == "true" else False  # let's apply standardization to the features

    for g in gamma:

        # create the Deltr object
        dtr = Deltr("Gender", g, number_of_iterations, learning_rate=0.00001,
                    lambdaa=0.001, init_var=0.01, standardize=standardize)

        print(formatted_data)

        # train the model
        print("Beginning to train the model with parameters: \n"
              "Protected Attribute: " + protected_attribute + "\n"
                                                              "Protected group: " + protected_group + "\n"
                                                                                                      "Gamma: " + str(
            g) + "\n"
                 "Number of Iterations: " + str(number_of_iterations) + "\n"
                                                                        "Training data size: " + str(
            formatted_data.shape) + "\n"
                                    "Standardize: " + str(standardize))
        print("This could take a while...")

        dtr.train(formatted_data)

        loss_dir_path = './HOIRank/DELTRLoss/nonBlind/' + experiment_name
        if not os.path.exists(loss_dir_path):
            os.makedirs(loss_dir_path)

        LOSS_PATH = "./HOIRank/DELTRLoss/" + experiment_name + "/(num_iterations=" + str(
            number_of_iterations) + ",gamma=" + str(g) + ")" + filename + ".csv"

        with open(LOSS_PATH, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(
                ['iteration', 'loss', 'loss_exposure', 'DELTR_Loss(calculated)', 'loss_standard', 'omega'])
            i = 0
            print('here,', dtr.log)

            for train_step in dtr.log:
                DELTR_Loss = train_step.loss_standard + g * train_step.loss_exposure
                csvwriter.writerow([str(i), str(train_step.loss), str(train_step.loss_exposure), DELTR_Loss,
                                    str(train_step.loss_standard), train_step.omega])
                i += 1

        FILE_PATH = r"./HOIRank/Models/" + experiment_name + '/' + "(num_iterations=" + str(
            number_of_iterations) + ",gamma=" + str(g) + ")" + filename + ".obj"
        # make file in 'Model' folder, pickle the model, and dump it there

        file = open(FILE_PATH, "wb")
        pickle.dump(dtr, file)
        print("SAVED MODEL TO PATH: " + FILE_PATH)

    print("SUCCESS! Time Taken: ", time.time() - START)


def TrainBlind():
    START = time.time()
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)

    filename = os.path.basename(settings["READ_FILE_SETTINGS"]["PATH"]).split('.')[0]
    read_file = './HOIRank/Datasets/' + filename + '/Training/' + 'Training_' + filename + '.csv'

    if not os.path.isfile(read_file):
        print("This file: " + read_file + "does not exist, check read file options in settings.json")
        return

    train_data = pd.read_csv(read_file)

    # reorder doc_ids
    train_data['rearranged_id'] = range(1, len(train_data['doc_id']) + 1)
    train_data['doc_id'] = train_data['rearranged_id']

    # make all Gender values 1
    train_data['Gender'] = 1

    print(train_data)

    # reset index - not important as we do not use the index
    train_data = train_data.drop('rearranged_id', axis=1)

    # Only get the numeric colums in the training dataset and make a new dataframe
    numeric_cols = list(train_data.select_dtypes(include=[np.number]).columns.values)
    print("numeric columns:", numeric_cols)
    formatted_data = train_data[numeric_cols].copy()
    formatted_data = formatted_data.reset_index(drop=True)

    score_column = settings["DELTR_OPTIONS"]["SCORE_COLUMN"]

    # lower_better will always be false. May remove
    # TODO
    lower_better = settings["READ_FILE_SETTINGS"]["LOWER_SCORE_BETTER"].lower()
    normalized = settings["DELTR_OPTIONS"]["NORMALIZE_SCORE_COLUMN"].lower()
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

    # set up the DELTR object, # column name of the protected attribute (index after query and document id)
    if settings["READ_FILE_SETTINGS"]["GENDER_EXPERIMENT"].lower() == "true":
        protected_attribute = "Gender"
    elif settings["READ_FILE_SETTINGS"]["RACE_EXPERIMENT"].lower() == "true":
        protected_attribute = "Race"
    gamma = settings["DELTR_OPTIONS"]["gamma"]  # list of the gamma parameters
    number_of_iterations = settings["DELTR_OPTIONS"][
        "num_iterations"]  # number of iterations the training should run
    standardize = True if settings["DELTR_OPTIONS"][
                              "standardize"].lower() == "true" else False  # let's apply standardization to the features

    for g in gamma:

        # create the Deltr object
        dtr = Deltr("Gender", g, number_of_iterations, learning_rate=0.00001,
                    lambdaa=0.001, init_var=0.01, standardize=standardize)

        print(formatted_data)

        # train the model
        print("Beginning to train the model with parameters: \n"
              "Protected Attribute: " + protected_attribute + "\n"
                                                              "Protected group: " + protected_group + "\n"
                                                                                                      "Gamma: " + str(
            g) + "\n"
                 "Number of Iterations: " + str(number_of_iterations) + "\n"
                                                                        "Training data size: " + str(
            formatted_data.shape) + "\n"
                                    "Standardize: " + str(standardize))
        print("This could take a while...")

        dtr.train(formatted_data)

        loss_dir_path = './HOIRank/DELTRLoss/Blind/' + experiment_name
        if not os.path.exists(loss_dir_path):
            os.makedirs(loss_dir_path)

        LOSS_PATH = loss_dir_path + "/(num_iterations=" + str(
            number_of_iterations) + ",gamma=" + str(g) + ")" + filename + ".csv"

        with open(LOSS_PATH, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(
                ['iteration', 'loss', 'loss_exposure', 'DELTR_Loss(calculated)', 'loss_standard', 'omega'])
            i = 0
            print('here,', dtr.log)

            for train_step in dtr.log:
                DELTR_Loss = train_step.loss_standard + g * train_step.loss_exposure
                csvwriter.writerow([str(i), str(train_step.loss), str(train_step.loss_exposure), DELTR_Loss,
                                    str(train_step.loss_standard), train_step.omega])
                i += 1

        FILE_PATH = r"./HOIRank/BlindModels/" + experiment_name + '/' + "(num_iterations=" + str(
            number_of_iterations) + ",gamma=" + str(g) + ")" + filename + ".obj"
        # make file in 'Model' folder, pickle the model, and dump it there

        file = open(FILE_PATH, "wb")
        pickle.dump(dtr, file)
        print("SAVED MODEL TO PATH: " + FILE_PATH)

    print("SUCCESS! Time Taken: ", time.time() - START)