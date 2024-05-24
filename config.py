import json
import os




with open('./HOIRank/settings.json') as f:

    settings = json.load(f)
experiment_name = os.path.basename(settings["READ_FILE_SETTINGS"]["PATH"]).split(".")[0]

gender_labels = settings["GENDER_DATA_DEFINE"]
protected_keys = [k for k, v in gender_labels.items() if v == '1']
unprotected_keys = [k for k, v in gender_labels.items() if v == '0']

if 'Female' in protected_keys:
    protected_group = 'Females'
    unprotected_group = 'Males'

if 'Female' in unprotected_keys:
    protected_group = 'Males'
    unprotected_group = 'Females'

