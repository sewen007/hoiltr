import csv
import json
import urllib.request
from urllib.request import urlopen
import requests
import os
from time import sleep
import pandas as pd

with open('./FairRank/settings.json', 'r') as f:
    settings = json.load(f)

unknown_value = settings["GENDER_DATA_DEFINE"]["Default"]


def infer_with_btn(firstName, api_key):
    """
    :param firstName: the first name of the person's gender you are trying to identify
    :return: (str) The Gender Data Define Mappings in settings.json
    (example: "0" for Inferred Male, "1" if Inferred Female
    """
    data_define = settings["GENDER_DATA_DEFINE"]
    url = settings["INFERENCE_METHODS"]["BTN"]["URL"]
    api_call = url + "name=" + firstName + "&key=" + api_key
    try:
        result = urllib.request.urlopen(api_call)
    except UnicodeEncodeError:
        print("Error: Could not be identified")
        return (data_define['Default'], False)

    # api_call = url + "name=" + firstName + "&key=" + api_key
    # result = urllib.request.urlopen(api_call)
    api_call_result = result.read().decode('utf-8')
    json_obj = json.loads(api_call_result)

    # data_define = settings["GENDER_DATA_DEFINE"]

    try:
        inferred_gender = str(json_obj[0]['gender'])
        return (data_define[inferred_gender], True)
    except KeyError:
        print(firstName + ": Could not be identified")
        return (data_define['Default'], False)


def BehindTheName():
    """
    Give it name in format "FIRST LAST". Also important to run this function and change the default option in
    settings.json to male and female so that you have two datasets where the default for one is male and the default for
    the other is female.
    :return:
    """

    filename = os.path.basename(settings["READ_FILE_SETTINGS"]["PATH"]).split('.')[0]
    write_path = './FairRank/Datasets/' + filename + '/Inferred/CaseStudies/BTN'
    if not os.path.exists(write_path):
        os.makedirs(write_path)

    results_path = write_path + '/results.csv'

    read_file = './FairRank/Datasets/' + filename + '/Testing' + "/Testing_" + filename + '.csv'
    try:
        test_data = pd.read_csv(read_file)
    except FileNotFoundError:
        print("This file does not exist: " + read_file)

    print("Reading From: ", read_file)

    inferred = []
    api_keys = settings["INFERENCE_METHODS"]["BTN"]["API_KEY"]
    day_calls = 0
    total_hour_calls = 0
    api_calls_counter = 0
    api_index = 0

    # For Accuracy Calculation
    total_including = 0
    total_excluding = 0
    correct = 0
    unidentified = 0
    num_female = 0
    num_male = 0
    overall_correct = 0

    for index, row in test_data.iterrows():
        day_calls += 1
        total_hour_calls += 1
        api_calls_counter += 1
        if api_calls_counter >= 400:
            print("MAX Calls REACHED for API_KEY: ", api_keys[api_index])
            api_index += 1
            api_calls_counter = 0
        if day_calls >= ((len(api_keys) * 4000) - 100):
            print("CallNumber: " + str(day_calls))
            print("Waiting for 24 hours: Maximum Limit of Calls Reached Per Day Across All API_KEYS")
            sleep(86520)
            total_hour_calls = 0
            day_calls = 0
            api_index = 0
        if total_hour_calls >= len(api_keys) * 400:
            print("Pausing at Call: ", day_calls)
            print("Waiting for One Hour: Maximum Limit of Calls Reached Per Hour Across All API_KEYS")
            sleep(3720)
            total_hour_calls = 0
            api_index = 0
            print("Going Back to First API_KEY: ", api_keys[api_index])
        sleep(0.5)
        first = row[settings["INFERENCE_METHODS"]["INFER_COL"]].split(" ")[0]
        print("Testing Name: " + first + ", With Behind the Name")
        res = infer_with_btn(first, api_keys[api_index])
        inferred.append(res[0])

        actual = str(row["Gender"])
        male = settings["GENDER_DATA_DEFINE"]["Male"]
        female = settings["GENDER_DATA_DEFINE"]["Female"]

        if not res[1]:
            # When the name is unidentifiable
            if res[0] == actual:
                overall_correct += 1

            if actual == male:
                num_male += 1
            elif actual == female:
                num_female += 1

            unidentified += 1
            total_including += 1
            continue
        if res[0] == actual:
            # When the name is identifiable and the inference is correct
            overall_correct += 1
            correct += 1
            total_including += 1
            total_excluding += 1
        else:
            # When the name is identifiable and the inference is NOT correct
            total_including += 1
            total_excluding += 1

    print("Total Calls: ", total_including)
    print("Percent Unidentifiable: ", unidentified / total_including)
    print("Accuracy Excluding Unidentifiable Results: ", correct / total_excluding)
    print("Accuracy Including Unidentifiable Results: ", correct / total_including)
    print("Percent of Females Unidentified: ", num_female / unidentified)
    print("Percent of Males Unidentified: ", num_male / unidentified)
    print("Overall Accuracy: ", overall_correct / total_including)

    test_data.insert(3, "InferredGender", inferred)

    write_path = './FairRank/Datasets/' + filename + '/Inferred/CaseStudies/BTN'
    if not os.path.exists(write_path):
        os.makedirs(write_path)

    # save results to a csv file
    write_result(write_path, total_including, unidentified, correct, total_excluding, num_female, num_male,
                 overall_correct)

    # write the inferred data to a csv file
    test_data.to_csv(
        write_path + "/(Default=" + settings["GENDER_DATA_DEFINE"]["Default"] + ")BTN_Inferred_" + filename + ".csv",
        index=False)

    print("SUCCESS! Saved to: " + write_path + "/(Default=" + settings["GENDER_DATA_DEFINE"][
        "Default"] + ")BTN_Inferred_" + filename + ".csv")


def infer_with_namesor(first_name):
    url = settings["INFERENCE_METHODS"]["NMSOR"]["URL"]
    api_key = settings["INFERENCE_METHODS"]["NMSOR"]["API_KEY"]

    payload = {
        "personalNames": [
            {
                "id": "b590b04c-da23-4f2f-a334-aee384ee420a",
                "firstName": first_name
            }
        ]
    }
    headers = {
        "X-API-KEY": api_key,
        "Accept": "application/json",
        "Content-Type": "application/json"
    }

    response = requests.request("POST", url, json=payload, headers=headers)
    res_json = response.json()

    if res_json["personalNames"][0]["score"] < 10:
        print(first_name + ': Could not be identified')
        return (settings["GENDER_DATA_DEFINE"]["Default"], False)
    else:
        return (settings["GENDER_DATA_DEFINE"][res_json["personalNames"][0]["likelyGender"]], True)


def NameSor():
    """
       Give it name in format "FIRST LAST". Also important to run this function and change the default option in
       settings.json to male and female so that you have two datasets where the default for one is male and the default for
       the other is female.
       :return:
       """
    filename = os.path.basename(settings["READ_FILE_SETTINGS"]["PATH"]).split('.')[0]
    read_file = './FairRank/Datasets/' + filename + '/Testing/' + 'Testing_' + filename + '.csv'
    try:
        test_data = pd.read_csv(read_file)
    except FileNotFoundError:
        print("This file does not exist: " + read_file)

    inferred = []

    # For Accuracy Calculation
    total_including = 0
    total_excluding = 0
    correct = 0
    unidentified = 0
    num_female = 0
    num_male = 0
    overall_correct = 0

    for index, row in test_data.iterrows():
        first = row[settings["INFERENCE_METHODS"]["INFER_COL"]].split(" ")[0]
        print("Testing Name: " + first + ", With NameSor")
        res = infer_with_namesor(first)
        inferred.append(res[0])

        actual = str(row["Gender"])
        male = settings["GENDER_DATA_DEFINE"]["Male"]
        female = settings["GENDER_DATA_DEFINE"]["Female"]

        if not res[1]:
            # When the name is unidentifiable
            if res[0] == actual:
                overall_correct += 1

            if actual == male:
                num_male += 1
            elif actual == female:
                num_female += 1

            unidentified += 1
            total_including += 1
            continue
        if res[0] == actual:
            # When the name is identifiable and the inference is correct
            overall_correct += 1
            correct += 1
            total_including += 1
            total_excluding += 1
        else:
            # When the name is identifiable and the inference is NOT correct
            total_including += 1
            total_excluding += 1

    print("Total Calls: ", total_including)
    print("Percent Unidentifiable: ", unidentified / total_including)
    print("Accuracy Excluding Unidentifiable Results: ", correct / total_excluding)
    print("Accuracy Including Unidentifiable Results: ", correct / total_including)
    print("Percent of Females Unidentified: ", num_female / unidentified)
    print("Percent of Males Unidentified: ", num_male / unidentified)
    print("Overall Accuracy: ", overall_correct / total_including)

    test_data.insert(3, "InferredGender", inferred)

    write_path = './FairRank/Datasets/' + filename + '/Inferred/CaseStudies/NMSOR'
    if not os.path.exists(write_path):
        os.makedirs(write_path)

    # save results to a csv file
    write_result(write_path, total_including, unidentified, correct, total_excluding, num_female, num_male,
                 overall_correct)

    test_data.to_csv(
        write_path + "/(Default=" + settings["GENDER_DATA_DEFINE"]["Default"] + ")NMSOR_Inferred_" + filename + ".csv",
        index=False)

    print(
        write_path + "/(Default=" + settings["GENDER_DATA_DEFINE"]["Default"] + ")NMSOR_Inferred_" + filename + ".csv")


def infer_with_gapi(firstname):
    myKey = settings["INFERENCE_METHODS"]["GAPI"]["API_KEY"]
    url = settings["INFERENCE_METHODS"]["GAPI"]["URL"] + myKey + "&name=" + firstname
    try:
        response = urlopen(url)
    except UnicodeEncodeError:
        print("Error: Could not be identified")
        return (settings["GENDER_DATA_DEFINE"]["Default"], False)

    decoded = response.read().decode('utf-8')
    data = json.loads(decoded)
    infer = data["gender"]

    try:
        return (settings["GENDER_DATA_DEFINE"][infer], True)
    except KeyError:
        print(firstname + ": could not be identified")
        return (settings["GENDER_DATA_DEFINE"]["Default"], False)


def GenderAPI():
    """
       Give it name in format "FIRST LAST". Also important to run this function and change the default option in
       settings.json to male and female so that you have two datasets where the default for one is male and the default for
       the other is female.
       :return:
    """
    filename = os.path.basename(settings["READ_FILE_SETTINGS"]["PATH"]).split('.')[0]
    read_file = './FairRank/Datasets/' + filename + '/Testing/' + 'Testing_' + filename + '.csv'
    try:
        test_data = pd.read_csv(read_file)
    except FileNotFoundError:
        print("This file does not exist: " + read_file)
        return

    inferred = []

    # For Accuracy Calculation
    total_including = 0
    total_excluding = 0
    correct = 0
    unidentified = 0
    num_female = 0
    num_male = 0
    overall_correct = 0

    for index, row in test_data.iterrows():
        first = row[settings["INFERENCE_METHODS"]["INFER_COL"]].split(" ")[0]
        print("Testing Name: " + first + ", With GenderAPI")
        res = infer_with_gapi(first)
        inferred.append(res[0])

        actual = str(row["Gender"])
        male = settings["GENDER_DATA_DEFINE"]["Male"]
        female = settings["GENDER_DATA_DEFINE"]["Female"]

        if not res[1]:
            # When the name is unidentifiable
            if res[0] == actual:
                overall_correct += 1

            if actual == male:
                num_male += 1
            elif actual == female:
                num_female += 1

            unidentified += 1
            total_including += 1
            continue
        if res[0] == actual:
            # When the name is identifiable and the inference is correct
            overall_correct += 1
            correct += 1
            total_including += 1
            total_excluding += 1
        else:
            # When the name is identifiable and the inference is NOT correct
            total_including += 1
            total_excluding += 1

    print("Total Calls: ", total_including)
    print("Percent Unidentifiable: ", unidentified / total_including)
    print("Accuracy Excluding Unidentifiable Results: ", correct / total_excluding)
    print("Accuracy Including Unidentifiable Results: ", correct / total_including)
    print("Percent of Females Unidentified: ", num_female / unidentified)
    print("Percent of Males Unidentified: ", num_male / unidentified)
    print("Overall Accuracy: ", overall_correct / total_including)

    test_data.insert(3, "InferredGender", inferred)

    write_path = './FairRank/Datasets/' + filename + '/Inferred/CaseStudies/GAPI'
    if not os.path.exists(write_path):
        os.makedirs(write_path)

    # save results to a csv file
    write_result(write_path, total_including, unidentified, correct, total_excluding, num_female, num_male,
                 overall_correct)

    test_data.to_csv(
        write_path + "/(Default=" + settings["GENDER_DATA_DEFINE"]["Default"] + ")GAPI_Inferred_" + filename + ".csv",
        index=False)

    print(write_path + "/(Default=" + settings["GENDER_DATA_DEFINE"]["Default"] + ")GAPI_Inferred_" + filename + ".csv")


def write_result(write_path, total_including, unidentified, correct, total_excluding, num_female, num_male,
                 overall_correct):
    # create file to save results
    results_path = write_path + '/results-' + str(unknown_value) + '.csv'
    with open(results_path, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        # write row headers

        csvwriter.writerow(["Total Calls", "Percent Unidentifiable", "Accuracy Excluding Unidentifiable Results",
                            "Accuracy Including Unidentifiable Results", "Percent of Females Unidentified",
                            "Percent of Males Unidentified", "Overall Accuracy"])
        csvwriter.writerow([total_including, unidentified / total_including, correct / total_excluding,
                            correct / total_including,
                            num_female / unidentified, num_male / unidentified, overall_correct / total_including])
