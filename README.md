# new_textual_features_for_public_speaking


This repository contains code for the extraction and evaluation of the new textual features extracted from transcripts of speech of performances of speakers on the competiton "Ma thèse en 180 secondes". 

In the main folder one can find text of the paper (Paper_Draft.pdf) where we discuss our results.

## preprocessing

Before feature extraction data has to be separated on the classes. For that one have to execute code in the file preprocessing/LabelProcessor_MT.py by writing:

" python3 LabelProcessor_MT.py "

Do not forget to specify internal variable rate_type with the score you want to work with.

## feature extraction

Code with feature extraction is located in the file preprocessing/TextProcessor_MT.py. To execute it you can use command:

" python3 TextProcessor_MT.py --dataset MT "

## SHAP values calculation

You can execute learning of classifier and SHAP values calculation by starting code from the file test_MT.py:

" python3 test_MT.py"

Do not forget to specify internal variable rate_type with the score you want to work with. Do not forget to specify the same variable in the feedback/SHAP.py.

## Results

Finally you will have results in the folder results/ + str(rate_type) depending on which rate_type you specified above.


## Contacts

With any questions you can contact me via alisa.barkar@talecom-paris.fr


