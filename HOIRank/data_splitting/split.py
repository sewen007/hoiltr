import csv
import random
import json
import os
import copy
import pandas as pd

with open('./HOIRank/settings.json', 'r') as f:
    settings = json.load(f)
experiment_name = os.path.basename(settings["READ_FILE_SETTINGS"]["PATH"]).split('.')[0]
variants = settings["DELTR_OPTIONS"]["wrong_inference_percent"]
protected_attribute = settings["READ_FILE_SETTINGS"]["DEMO_COL"]
swap_percent = settings["DELTR_OPTIONS"]["swap_percent"]


def Split():
    train_split = settings["DATA_SPLIT"]["TRAIN_PCT"]
    filename = os.path.basename(settings["READ_FILE_SETTINGS"]["PATH"]).split('.')[0]

    write_path = './HOIRank/Datasets/' + filename + '/Training'
    if not os.path.exists(write_path):
        os.makedirs(write_path)

    write_path = './HOIRank/Datasets/' + filename + '/Testing'
    if not os.path.exists(write_path):
        os.makedirs(write_path)

    read_file = './HOIRank/Datasets/' + filename + '/Cleaned' + '/Cleaned_' + filename + '.csv'
    train_file = './HOIRank/Datasets/' + filename + '/Training' + '/Training_' + filename + '.csv'
    test_file = './HOIRank/Datasets/' + filename + '/Testing' + "/Testing_" + filename + '.csv'

    if os.path.isfile(train_file):
        print("This File Has Already Been Split Into Training File at Path: " + train_file)
        return
    if os.path.isfile(test_file):
        print("This File Has Already Been Split Into Testing File at Path: " + test_file)
        return

    # Open the file
    with open(read_file, mode='r') as file:
        # reading the CSV file
        csvFile = csv.reader(file)
        # Skip the headers row
        headers = next(csvFile)
        # Write headers to both train and test data sets
        writeData(train_file, headers)
        writeData(test_file, headers)
        for lines in csvFile:
            if random.random() < train_split:
                writeData(train_file, lines)
            else:
                writeData(test_file, lines)

    # reset doc_id's
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    train_df['doc_id'] = range(1, len(train_df) + 1)
    test_df['doc_id'] = range(1, len(test_df) + 1)

    print(test_df.dtypes)
    print(train_df.dtypes)

    test_df.to_csv(test_file, index=False)
    train_df.to_csv(train_file, index=False)

    print("Success!: Saved Train File to: " + train_file)
    print("Success!: Saved Test File to: " + test_file)


def writeData(write_file, fields):
    # writing to csv file
    with open(write_file, 'a') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)

        # writing the fields
        csvwriter.writerow(fields)


def select_and_assign(lst, percent):
    """Select a percentage of items from lst and assign new_value to only those selected items."""
    num_to_select = int(len(lst) * percent / 100)
    selected_indices = random.sample(range(len(lst)), num_to_select)
    new_lst = lst.copy()
    for i in selected_indices:
        new_lst[i] = 1 - lst[i]
    return new_lst


def VariantSplit(flip_choice, seedy):
    swap_percentage = 0
    csv_path = './HOIRank/Datasets/' + experiment_name + '/Testing' + '/Testing_' + experiment_name + '.csv'
    dataframe_to_swap = pd.read_csv(csv_path)
    protected_column = dataframe_to_swap[protected_attribute]
    size = len(protected_column)
    swapped_column = copy.copy(protected_column)
    flip_choice = flip_choice

    random.seed(seedy)

    zero_indices = []
    one_indices = []

    for i in range(len(protected_column)):
        if protected_column[i] == 0:
            zero_indices.append(i)
        elif protected_column[i] == 1:
            one_indices.append(i)

    random.shuffle(zero_indices)
    random.shuffle(one_indices)

    while swap_percentage <= 1.0:
        if flip_choice == 'both':
            indices_to_swap = zero_indices[:int(swap_percentage * size)] + one_indices[:int(swap_percentage * size)]
        elif flip_choice == 'protected':
            indices_to_swap = one_indices[:int(swap_percentage * size)]
        elif flip_choice == 'unprotected':
            indices_to_swap = zero_indices[:int(swap_percentage * size)]

        for index in indices_to_swap:
            if protected_column[index] == 1:
                swapped_column[index] = 0
            elif protected_column[index] == 0:
                swapped_column[index] = 1

        dataframe_to_swap[str(protected_attribute)] = swapped_column
        unnamed_columns = [col for col in dataframe_to_swap.columns if col.startswith("Unnamed")]
        dataframe_to_swap = dataframe_to_swap.drop(columns=unnamed_columns)

        write_path = './HOIRank/Datasets/' + experiment_name + '/Inferred/' + flip_choice + '/' + str(
            int(float(swap_percentage * 100))) + "/"
        if not os.path.exists(write_path):
            os.makedirs(write_path)
        csv_file = str(seedy) + "_Inferred_" + experiment_name + ".csv"
        print(csv_file)
        dataframe_to_swap.to_csv(write_path + csv_file, index=False)
        swap_percentage += swap_percent
        swap_percentage = float(format(swap_percentage, '.1f'))
