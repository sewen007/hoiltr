import json
import math
import os
import re
from collections import defaultdict as ddict
import operator
import pandas as pd
import HOIRank.data_ranking.rank as rank

# from learning_to_rank.listwise import ListNet as ln

with open('./HOIRank/settings.json', 'r') as f:
    settings = json.load(f)
experiment_name = os.path.basename(settings["READ_FILE_SETTINGS"]["PATH"]).split('.')[0]

with open('./HOIRank/settings.json', 'r') as f:
    settings = json.load(f)

# flip_choice = settings["DELTR_OPTIONS"]["flip_choice"]
filename = os.path.basename(settings["READ_FILE_SETTINGS"]["PATH"]).split('.')[0]
gt = './HOIRank/Datasets/' + experiment_name + '/Testing/Testing' + '_' + experiment_name + '.csv'
score_column = settings["DELTR_OPTIONS"]["SCORE_COLUMN"]


def detconstsort(a, k_max, p):
    scores = []
    for a_i in a.keys():
        for i_d, score in a[a_i].items():
            scores.append((a_i, i_d, score))
    attributes = a.keys()
    attribute_scores = {}

    # create and initialize counter for each attribute value
    counts_ai = {}
    minCounts_ai = {}
    totalCounts_ai = {}
    for a_i in a.keys():
        counts_ai[a_i] = 0
        minCounts_ai[a_i] = 0
        totalCounts_ai[a_i] = len(a[a_i])

    re_ranked_attr_list = {}
    re_ranked_score_list = {}
    maxIndices = {}

    lastEmpty = 0
    k = 0

    for i, a_i in enumerate(attributes):
        counts_ai[a_i] = 0
        minCounts_ai[a_i] = 0
        totalCounts_ai[a_i] = sum([1 for s in scores if s[0] == a_i])
        attribute_scores[a_i] = [(s[2], s[1]) for s in scores if
                                 s[0] == a_i]

    # print(attribute_scores)

    while lastEmpty <= k_max:

        if lastEmpty == len(scores):
            break

        k += 1
        tempMinAttrCount = ddict(int)
        changedMins = {}
        for a_i in attributes:
            tempMinAttrCount[a_i] = math.floor(k * p[a_i])
            if minCounts_ai[a_i] < tempMinAttrCount[a_i] and minCounts_ai[a_i] < totalCounts_ai[a_i]:
                changedMins[a_i] = attribute_scores[a_i][counts_ai[a_i]]

        if len(changedMins) != 0:
            ordChangedMins = sorted(changedMins.items(), key=lambda x: x[1][0], reverse=True)
            for a_i in ordChangedMins:
                re_ranked_attr_list[lastEmpty] = a_i[0]
                lastEmpty = int(lastEmpty)
                # print('here', attribute_scores[a_i[0]][counts_ai[a_i[0]]])
                re_ranked_score_list[lastEmpty] = attribute_scores[a_i[0]][counts_ai[a_i[0]]]
                maxIndices[lastEmpty] = k
                start = lastEmpty
                while start > 0 and maxIndices[start - 1] >= start and re_ranked_score_list[start - 1][0] < \
                        re_ranked_score_list[start][0]:
                    swap(re_ranked_score_list, start - 1, start)
                    swap(maxIndices, start - 1, start)
                    swap(re_ranked_attr_list, start - 1, start)
                    start -= 1
                counts_ai[a_i[0]] += 1
                lastEmpty += 1
            minCounts_ai = dict(tempMinAttrCount)

    re_ranked_attr_list = [re_ranked_attr_list[i] for i in sorted(re_ranked_attr_list)]
    re_ranked_score_list = [re_ranked_score_list[i] for i in sorted(re_ranked_score_list)]

    return re_ranked_attr_list, re_ranked_score_list


def swap(temp_list, pos_i, pos_j):
    temp = temp_list[pos_i]
    temp_list[pos_i] = temp_list[pos_j]
    temp_list[pos_j] = temp


def wrapper(url):
    """
    This is the wrapper code to convert detlr output to input 1 for detconstsort_rank
    :param url: url pointing to deltr output
    :return:
    """
    a = {}
    df = pd.read_csv(url)
    dff = df.groupby('Gender')
    for row in df['Gender']:
        a[row] = dict(zip(dff.get_group(row).doc_id, dff.get_group(row).judgement))
    return a


def getdist(df):
    # Given the ranked dataframe, return the true protected attr dist as a dictionary
    d = {}
    for index, row in df.iterrows():
        if row["Gender"] not in d:
            d[row['Gender']] = 1
        else:
            d[row['Gender']] += 1
    for attr in d:
        d[attr] = d[attr] / len(df)
    return d


def find_unaware_ranked(file):
    match = re.search('gamma=0.0', file)
    if match:
        return True
    else:
        return False


def infer_with_detconstsort(file, flip_choice, inferred=False, hidden=False, blind=False):
    filename = os.path.basename(settings["READ_FILE_SETTINGS"]["PATH"]).split('.')[0]
    test_data = pd.read_csv(gt, index_col=False)

    write_path = './HOIRank/Datasets/' + filename + '/Ranked/' + flip_choice + '/DetConstSort_Ranked'
    if not os.path.exists(write_path):
        os.makedirs(write_path)

    ranked_dict = {}
    if inferred and hidden:
        params = file.split('Colorblind_Ranked_Inferred\\Colorblind_Ranked(')[1]
        write_file = write_path + '/DetConstSortHidden_Ranked(' + params

    elif inferred and not hidden:
        if blind:
            params = file.split('Blind_Ranked_Inferred\\BlindGroundTruth_Ranked(')[1]
            write_file = write_path + '/DetConstSortBlind_Ranked(' + params
        else:
            params = file.split("Inferred_Ranked(")[1]
            write_file = write_path + '/DetConstSortNotHidden_Ranked(' + params
    # not inferred
    elif not inferred and not hidden:
        params = re.findall(r"\(([^)]+)\)", file)[0]
        write_file = write_path + '/DetConstSortNotHidden_Ranked(' + params + ')_' + \
                     os.path.basename(settings["READ_FILE_SETTINGS"]["PATH"]).split('.')[0] + '_0_Inferred.csv'

    elif not inferred and hidden:
        params = re.findall(r"\(([^)]+)\)", file)[0]
        write_file = write_path + '/DetConstSortHidden_Ranked(' + params + ')_' + \
                     os.path.basename(settings["READ_FILE_SETTINGS"]["PATH"]).split('.')[0] + '_0_Inferred.csv'

    data = pd.read_csv(file)
    a = wrapper(file)
    p = getdist(data)
    k_max = len(data.index)

    result = detconstsort(a, k_max, p)
    result_genders = result[0]

    # get the scores and doc_id
    result_scores = result[1]

    for i in range(k_max):
        if result_scores[i][1] not in ranked_dict.keys():
            # include ground truth score for utility calculation later
            gt_score = test_data.loc[test_data['doc_id'] == result_scores[i][1], score_column].iloc[0]
            # return groundtruth gender for fairness calculation later
            gt_gender = test_data.loc[test_data['doc_id'] == result_scores[i][1], 'Gender'].iloc[0]
            ranked_dict[result_scores[i][1]] = [result_scores[i][1], gt_gender, result_scores[i][0], gt_score,
                                                result_genders[i]]

        else:
            print("There are duplicates in the ranking, something went wrong.")
            return

    rank.writeRanked(write_file, ranked_dict)


def DetConstSortHidden(flip_choice):
    """
    This function takes in a file and ranks it using DetConstSort. This version uses the hidden attribute approach for the first part
    of the ranking. The second part of the ranking is done using the original DetConstSort algorithm.
    :return:
    """

    # get colorblind ranked files ranked by gamma = 0.0
    colorblind_ranked_gt = filter(find_unaware_ranked, rank.get_files(
        './HOIRank/Datasets/' + filename + '/Ranked/' + flip_choice + '/Colorblind_Ranked'))

    # make inferred versions of the files including version with gt inserted(0% wrong inference)
    for ranked in colorblind_ranked_gt:
        make_inferred_versions_gt_ranked(flip_choice, ranked)

    inferred_ranked = rank.get_files(
        './HOIRank/Datasets/' + filename + '/Transit/nonBlind/Colorblind_Ranked_Inferred')

    for file in inferred_ranked:
        print(file)
        infer_with_detconstsort(file, flip_choice, inferred=True, hidden=True)


def make_inferred_versions_gt_ranked(flip_choice, path, blind=False):
    """
    This function takes in a path to a ground truth ranked file and creates inferred versions of it
    :param path:
    :return: inferred versions of the ground truth ranked file
    """

    # this is the file to which you want to add inferred demographics or create inferred versions
    background_file = pd.read_csv(path)
    df_file = background_file[['doc_id', 'Gender', 'judgement']]

    # these are the files with inferred demographics from simulations and casestudies
    read_path = './HOIRank/Datasets/' + experiment_name + '/Inferred/' + flip_choice + '/'
    inference_files = rank.get_files(read_path)

    # these are the files with ground truth demographics
    test_read_path = './HOIRank/Datasets/' + experiment_name + '/Testing/'
    test_files = rank.get_files(test_read_path)

    if blind:
        write_path = './HOIRank/Datasets/' + filename + '/Transit/Blind/'+ flip_choice +'/Blind_Ranked_Inferred'
    else:
        write_path = './HOIRank/Datasets/' + filename + '/Transit/nonBlind/'+ flip_choice +'/Colorblind_Ranked_Inferred'
    if not os.path.exists(write_path):
        os.makedirs(write_path)

    # take each file with inferred demographics and add the inferred demographics to the background file
    for inference_file in inference_files:
        df = pd.read_csv(inference_file)
        if flip_choice == 'CaseStudies':
            df_file.loc[:, 'Gender'] = df_file['doc_id'].apply(
                lambda x: df.loc[df['doc_id'] == x, 'InferredGender'].iloc[0])
        else:
            df_file.loc[:, 'Gender'] = df_file['doc_id'].apply(lambda x: df.loc[df['doc_id'] == x, 'Gender'].iloc[0])
        df_file.to_csv(write_path + '/' + os.path.basename(path) + '_' +os.path.split(os.path.dirname(inference_file))[-1] +'_'+os.path.basename(inference_file), index=False)

    if flip_choice == "CaseStudies":

        for test_file in test_files:
            df = pd.read_csv(test_file)
            df_file.loc[:, 'Gender'] = df_file['doc_id'].apply(lambda x: df.loc[df['doc_id'] == x, 'Gender'].iloc[0])
            df_file.to_csv(write_path + '/' + os.path.basename(path) + '_0_Inferred', index=False)


def DetConstSortNotHidden(flip_choice):
    """
    This function takes in a file and ranks it using DetConstSort. This version uses the attributes during for the first part
    of the ranking. The second part of the ranking is done using the original DetConstSort algorithm.
    :return:
    """

    inferred_ranked = filter(find_unaware_ranked, rank.get_files(
        './HOIRank/Datasets/' + filename + '/Ranked/' + flip_choice + '/Inferred_Ranked'))

    # get ranked files ranked by gamma = 0.0 only for "CaseStudies" (the simulations have it already)
    if flip_choice == 'CaseStudies':
        ranked_gt = filter(find_unaware_ranked, rank.get_files(
            './HOIRank/Datasets/' + filename + '/Ranked/' + flip_choice + '/GroundTruth_Ranked'))

        for file in ranked_gt:
            print(file)
            infer_with_detconstsort(file, flip_choice, inferred=False, hidden=False)

    for file in inferred_ranked:
        print(file)
        infer_with_detconstsort(file, flip_choice, inferred=True, hidden=False)


def DetConstSortBlind(flip_choice):
    """
    This function takes in a file and ranks it using DetConstSort. This version uses the hidden attribute approach for the first part
    of the ranking. The second part of the ranking is done using the original DetConstSort algorithm.
    :return:
    """

    # get blind ranked files ranked by gamma = 0.0
    blind_ranked_gt = filter(find_unaware_ranked, rank.get_files(
        './HOIRank/Datasets/' + filename + '/Ranked/' + flip_choice + '/BlindGroundTruth_Ranked'))

    # make inferred versions of the files including version with gt inserted(0% wrong inference)
    for ranked in blind_ranked_gt:
        make_inferred_versions_gt_ranked(flip_choice, ranked, blind=True)
    inferred_ranked = rank.get_files(
        './HOIRank/Datasets/' + filename + '/Transit/Blind/' + flip_choice + '/Blind_Ranked_Inferred')
    for file in inferred_ranked:
        print(file)
        infer_with_detconstsort(file, flip_choice, inferred=True, hidden=False, blind=True)
