from FairRank import *

start = time.time()
# settings_files = ["./FairRank/settings-BM.json", "./FairRank/settings-COMPAS.json", "./FairRank/settings-LAW.json", "./FairRank/settings-NBA.json"]
# for settings_file in settings_files:
#     setting = settings_file
#     with open(setting, 'r') as f:
#         settings = json.load(f)

flip_choices = settings["DELTR_OPTIONS"]["flip_choices"]
# flip_choice = "both"

seeds = settings["DELTR_OPTIONS"]["seeds"]


# seed = settings["DELTR_OPTIONS"]["seed"]


def full_experiment(flip_choice):
    print(flip_choice)
    # CalculateInitialMetrics(flip_choice)
    #
    # # Rank Ground Truth Datasets, Rank Inferred Datasets. It is important to use the order set
    # print("RankGroundTruth()")
    # RankGroundTruth(flip_choice)

    # Rank Colorblind (Hidden)
    # print("RankColorblind()")
    # RankColorblind(flip_choice)

    # print("RankInferred()")
    # RankInferred(flip_choice)

    # DetConstSort Ranking
    # print("DetConstSortHidden()")
    # DetConstSortHidden(flip_choice)
    # print("DetConstSortNotHidden()")
    # DetConstSortNotHidden(flip_choice)
    print("DetConstSortBlind()")
    DetConstSortBlind(flip_choice)

    for seed in seeds:
        #     # print(seed)
        #     # print("VariantSplit()")
        #     # if flip_choice != "CaseStudies":
        #     #     VariantSplit(flip_choice, seed)

        # Calculating the Metrics
        print("CalculateResultsMetrics()")
        CalculateResultsMetrics(seed, flip_choice)


# for flip_choice in flip_choices:
#     print(flip_choice)
#
#     full_experiment(flip_choice)
#
#
# CollateNDCGandSkews()
#Make_Metric_Csvs()
#
# # after doing the experiments for all the datasets
#PlotGraphs()
#ParetoPlots()

# --------------------------------------
# optional
# --------------------------------------
# PlotLoss()
# PlotLossExposure()
# PlotListLoss()


# ----------------------------------------------------------------------------------------------------
########################################################################################################
########################################################################################################

# PlotGraphs()


# Clean the Dataset
# Clean()

# Split the Dataset
# Split()
# RankColorblind(flip_choice)
# DetConstSortHidden(flip_choice)
# DetConstSortNotHidden(flip_choice)
# CalculateInitialMetrics()
#
# # Infer demographic information using the test split
# BehindTheName()
# NameSor()
# GenderAPI()
#
# # Train the model using the train split
# #Train()
#
# # Rank Ground Truth Datasets, Rank Inferred Datasets. It is important to use the order set
# RankGroundTruth()
# RankColorblind()
# VariantSplit()
# RankInferred()
# #
# # # DetConstSort Ranking
# DetConstSortHidden()

# DetConstSortNotHidden()
#
# # Calculating the Metrics
# CalculateResultsMetrics()


# PlotLoss()
# PlotLossExposure()
# PlotListLoss()

# CombineResults()

#Make_Metric_Csvs()

# CalculateInitialMetrics("both")
# full_experiment("CaseStudies")
# for flip_choice in flip_choices:
#     print(flip_choice)
#     # CalculateInitialMetrics(flip_choice)
#     full_experiment(flip_choice)
# CollateNDCGandSkews()
# PlotGraphs(flip_choice)

# CollateNDCGandSkews()
# for flip_choice in flip_choices:
#     print(flip_choice)
#     full_experiment(flip_choice)
# CollateNDCGandSkews()
# Make_Metric_Csvs()
# PlotGraphs()

# full_experiment("both")
# CombineResults()

# Generate multiple results for each seed

# for seed in seeds:
# VariantSplit(seed)
# RankInferred()
# #
# # # DetConstSort Ranking
# DetConstSortHidden()
# DetConstSortNotHidden()
#
# # Calculating the Metrics
# CalculateResultsMetrics(seed, 'CaseStudies')
# Specify the directory where you want to start the renaming process
directory_to_search = 'FairRank/Results'

# Call the function to rename folders with '.csv' extension
rename_folders_with_csv(directory_to_search)

end = time.time()

print("time taken = ", end - start)
