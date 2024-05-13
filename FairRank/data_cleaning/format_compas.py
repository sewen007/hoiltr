import pandas as pd


def reverse_scores(scores):
    reversed_scores = [-score for score in scores]
    return reversed_scores


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
