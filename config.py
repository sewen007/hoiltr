import json
import os

<<<<<<< HEAD
with open('./FairRank/settings.json') as f:
=======
with open('./HOIRank/settings.json') as f:
>>>>>>> 8a25b3dfffce5f61e30d7b49f8f92d83c869914c
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

<<<<<<< HEAD
#flip_choice = settings["DELTR_OPTIONS"]["flip_choice"]
=======

>>>>>>> 8a25b3dfffce5f61e30d7b49f8f92d83c869914c
