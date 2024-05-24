import random
import os
import json
import time
import pandas as pd
import csv

with open('./HOIRank/settings.json', 'r') as f:
    settings = json.load(f)
experiment_name = os.path.basename(settings["READ_FILE_SETTINGS"]["PATH"]).split('.')[0]
prot_attr = os.path.basename(settings["READ_FILE_SETTINGS"]["DEMO_COL"]).split('.')[0]
wrong_inference_levels = settings["DELTR_OPTIONS"]["wrong_inference_percent"]


def select_and_assign(attr_list, percent):
    """Select a percentage of items from attribute list and change value"""
    num_to_select = int(len(attr_list) * percent / 100)
    selected_indices = random.sample(range(len(attr_list)), num_to_select)
    attr_list = attr_list.copy()
    for i in selected_indices:
        attr_list[i] = abs(1 - int(attr_list[i]))
    return attr_list


def GenerateFiles():
    START = time.time()
    read_file = './HOIRank/Datasets/' + experiment_name + '/Training/' + 'Training_' + experiment_name + '.csv'

    if not os.path.isfile(read_file):
        print("This file: " + read_file + "does not exist, check read file options in settings.json")
        return

    write_path = './HOIRank/Datasets/' + experiment_name + '/Variant_Inferences'
    if not os.path.exists(write_path):
        os.makedirs(write_path)

    for level in wrong_inference_levels:
        train_data = pd.read_csv(read_file)
        train_data[str(prot_attr) + " - " + str(level) + " percent wrong"] = select_and_assign(
            train_data[str(prot_attr)], level)

        gen_file = './HOIRank/Datasets/' + experiment_name + '/Variant_Inferences' + '/Variant_Inferences_' + experiment_name + '_' + str(
            level) + '.csv'

        if os.path.isfile(gen_file):
            print("This File Has Already Been Split Into Variant_Inference File at Path: " + gen_file)
            return
        train_data.to_csv(gen_file, index=False)
        print("Success!: Saved Train File to: " + gen_file)

    return


def writeData(write_file, fields):
    # writing to csv file
    with open(write_file, 'a') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)

        # writing the fields
        csvwriter.writerow(fields)
