import os
import pandas as pd
import json

with open('./HOIRank/settings.json', 'r') as f:
    settings = json.load(f)


def Clean():
    # Make sure file is a csv file
    if settings["READ_FILE_SETTINGS"]["PATH"][-4:] != '.csv':
        raise Exception(settings["READ_FILE_SETTINGS"]["PATH"] + " is not a csv file")

    filename = os.path.basename(settings["READ_FILE_SETTINGS"]["PATH"]).split('.')[0]

    # read file in
    try:
        cleaned_dataset = pd.read_csv(settings["READ_FILE_SETTINGS"]["PATH"])
        # if dataset = compas, then reverse the scores
        if filename == "COMPAS":
            format_compas()
    except:
        print("This File Cannot Be Found, Check Your Path in settings.json: " + settings["READ_FILE_SETTINGS"]["PATH"])
        return

    # Add the document and query ids that are necessary for DELTR to run
    if "doc_id" not in cleaned_dataset:
        cleaned_dataset.insert(0, 'doc_id', range(1, len(cleaned_dataset) + 1))
    if "q_id" not in cleaned_dataset:
        cleaned_dataset.insert(0, 'q_id', 1)

    # Make sure that the demographic column passed in exists in the dataset
    if settings["READ_FILE_SETTINGS"]["DEMO_COL"] not in cleaned_dataset:
        print("Make sure you have the right demographic column listed in settings.json!"
              "(Name of column that contains demographic information)")
        return

    # Replace gender column with 0's and 1's
    cleaned_dataset = cleaned_dataset.replace(settings["GENDER_DATA_DEFINE"])

    if settings["READ_FILE_SETTINGS"]["GENDER_EXPERIMENT"] == "True":
        cleaned_dataset.rename(columns={
            settings["READ_FILE_SETTINGS"]["DEMO_COL"]: 'Gender'
        }, inplace=True)
        neworder = ['q_id', 'doc_id', 'Gender']
    elif settings["READ_FILE_SETTINGS"]["RACE_EXPERIMENT"] == "True":
        cleaned_dataset.rename(columns={
            settings["READ_FILE_SETTINGS"]["DEMO_COL"]: 'Race'
        }, inplace=True)
        neworder = ['q_id', 'doc_id', 'Race']
    else:
        print("YOU NEED TO SPECIFY IF GENDER EXPERIMENT OR RACE EXPERIMENT IN 'settings.json' FILE")
        exit(70)

    for column in settings["READ_FILE_SETTINGS"]["ADDITIONAL_COLUMNS"]:
        if column not in cleaned_dataset:
            print(column + " does not exist in original dataset, "
                           "check your additional columns in settings.json")
            return

    neworder.extend(settings["READ_FILE_SETTINGS"]["ADDITIONAL_COLUMNS"])
    neworder.extend([settings["READ_FILE_SETTINGS"]["SCORE_COL"]])
    ordered_cleaned = cleaned_dataset[neworder]
    if settings["READ_FILE_SETTINGS"]["LOWER_SCORE_BETTER"].lower() == "true":
        ordered_cleaned = ordered_cleaned.sort_values([settings["READ_FILE_SETTINGS"]["SCORE_COL"]])
    elif settings["READ_FILE_SETTINGS"]["LOWER_SCORE_BETTER"].lower() == "false":
        ordered_cleaned = ordered_cleaned.sort_values([settings["READ_FILE_SETTINGS"]["SCORE_COL"]], ascending=False)
    else:
        print("Be Sure to mark down if a higher score is better in the settings.json file")

    print(ordered_cleaned)

    write_path = './FairRank/Datasets/' + filename + '/Cleaned'
    if not os.path.exists(write_path):
        os.makedirs(write_path)

    ordered_cleaned.to_csv(write_path + '/Cleaned_' + filename + '.csv', index=False)
    print("Saved To: " + write_path + '/Cleaned_' + filename + '.csv')


def reverse_scores(scores):
    reversed_scores = [-score for score in scores]
    return reversed_scores


def format_compas():
    compas_sex = pd.read_csv('Cleaned_COMPAS_SEX.csv')
    compas_sex['new_raw_score'] = reverse_scores(compas_sex['raw_score'])

    # arrange in descending order of scores
    compas_sex.sort_values(by='raw_score', ascending=True)
    compas_sex.drop(['raw_score'], axis=1, inplace=True)
    compas_sex = compas_sex.rename(columns={'new_raw_score': 'raw_score'})
    compas_sex.sort_values(by='raw_score', ascending=False, inplace=True)
    compas_sex['new_id'] = range(1, len(compas_sex['doc_id']) + 1)
    compas_sex = compas_sex.drop(columns='new_id', axis=1)
    compas_sex.to_csv('Cleaned_COMPASSEX.csv', index=False)
    return
