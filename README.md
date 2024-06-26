# Public Speaking Classification with Textual Features


This repository contains code for extracting and evaluating the new public speaking textual feature set (PSTFS) on the performance classification task. PSTFS is tested on the 3MT_French dataset (collected from the competition "Ma thèse en 180 seconds") [@Biancardi2024]. The first results with PSTFS are reported in the paper [@Barkar2023] and published on ICMI2023. This repository contains updated results for further research. 

## Paper and Report
The research paper [@Barkar2023] related to this project can be found [here](docs/Barkar2023.pdf).
A report with the updated detailed results can be found [here](docs/Report_Classification_Updates.pdf)

## Experimental Pipeline
![Presentation of the three main stages of the experimental pipeline: 1. Feature extraction: Features extracted out of speech transcripts, separated into categories and saved to .csv files with the ID corresponding to the IDs of the data samples in the 3MT_French dataset; 2. Training and testing: We trained several classical ML models for the classification task where we classify transcripts on the "low" and "high" quality speech based on extracted linguistic features solely; 3. Feature importance analysis: Correlation with human annotations and SHAP analysis.](images/schema.png)


## Classification Setup
In [@Barkar2023] we consider two classes of performance quality: Data points with human-evaluated scores equal to or higher than the median were classified as "high-quality". In comparison, those with scores lower than the median were classified as "low-quality". In the update of this experiment, we study the following classification setups: 

| Setup Keyword | Classes Sizes (Number of documents) |
|----------|----------|
| medianSep  | (TODO:fill) |
| meanSep | (TODO:fill)   |
| Q1Q3Sep | (TODO:fill)   |


For class separation execute code preprocessing/LabelProcessor_MT.py with corresponding Setup Keyword. See the example for the separation w.r.t. median:

```bash
python3 LabelProcessor_MT.py --setup medianSep --dimension dimension_of_interest --clip full
```

Here, as dimension_of_interest put the name of the dimension that you are interested in (it will be persuasiveness by default): 

| dimensions | options |
|----------|----------|
| persuasiveness  | mean, rms, harmmean, rater1, rater2, rater3 |
| engagement | mean, rms, harmmean, rater1, rater2, rater3 |
| confidence | mean, rms, harmmean, rater1, rater2, rater3 |
| global | mean, rms, harmmean, rater1, rater2, rater3 |

mean: simple arithmetic mean
rms: root mean squared
harmmean: harmonic mean
rater1: the rating of the first rater
rater2: the rating of the second rater
rate3: the rating of the third rater

You also can choose to work with the video clips (1 minute for the beginning, the middle and the end). The annotation schema for the clips was the same in the 3MT_French dataset, therefore, for each clip there are three raters to annotate it. All three raters always are different. For more details see [@Biancardi2024]. To switch to the clips you may use a clip with options: full, beg, mid, end. By default system will use "full".

## Feature extraction

Code with feature extraction is located in the file preprocessing/TextProcessor_MT.py. To execute it you can use the command:

```bash
python3 TextProcessor_MT.py --dataset MT "
```

If you want to use the code on the other datasets then you should prepare input data so that:
- Transcripts are contained in the folder: '../data/{dataset}/transcripts/' where dataset is the name of the folder containing the transcripts of your dataset.
- Each transcript is located in the separated .txt file with the name corresponding to the ID of this sample in the dataset.


## Training and Testing

We used several classical classification models:
| Model                          | Parameters                                                                 |
|--------------------------------|-----------------------------------------------------------------------------|
| Support Vector Machine (SVM)   | <ul><li><code>kernel</code>: ['rbf', 'linear']</li><li><code>probability</code>: [True]</li><li><code>gamma</code>: [0.001, 0.01, 0.1, 1, 'auto']</li><li><code>C</code>: [1, 10, 20]</li></ul> |
| Random Forest Classifier (RFC) | <ul><li><code>max_depth</code>: [None]</li><li><code>max_features</code>: [10, 20, 'sqrt']</li><li><code>min_samples_split</code>: [2, 3, 10]</li><li><code>min_samples_leaf</code>: [1, 3, 10]</li><li><code>bootstrap</code>: [True]</li><li><code>n_estimators</code>: [300, 400]</li><li><code>criterion</code>: ['gini']</li></ul> |
| Logistic Regression (LR)       | <ul><li><code>penalty</code>: ['l1', 'l2']</li><li><code>C</code>: [0.001, 0.01, 0.1, 1, 10, 20]</li><li><code>multi_class</code>: ['ovr']</li><li><code>solver</code>: ['liblinear']</li><li><code>max_iter</code>: [1000]</li></ul> |
| Naive Bayes (NB)               | None                                                                       |
| K-Nearest Neighbors (KNN)      | <ul><li><code>n_neighbors</code>: [3, 5, 7]</li><li><code>weights</code>: ['uniform', 'distance']</li><li><code>algorithm</code>: ['auto', 'ball_tree', 'kd_tree', 'brute']</li></ul> |


To test classification and obtain results one may use the following line executed from the root (default parameters: all, MT, persuasiveness, full):

```bash
python3 test_MT.py --model SVM --dataset MT --dimension dimension_of_interest --clip full
```

| Parameter | Options |
|----------|----------|
| model  | all, SVM, RFC, LR, NB, KNN |
| dataset | MT, POM  |
| dimension | <ul><li><code>MT</code>: persuasiveness, engagement, confidence, global</li><li><code>POM</code>: persuasive, confident, etc. </li></ul> |
| clip | full, beg, mid, end |

When the model is provided with the "all" option, the code will test all the classification models from this list: [SVM, RFC, LR, NB, KNN]. To add new models, please, add them to the file "./Models/ML_Model.py" and then to the list contained in the variable models_list in the test_MT.py.

## Feature Importance: SHAP values

SHAP value analysis is implemented in the same file with training and testing of the model, for more details refer to feedback/SHAP.py.

## Results

Finally, results  will be saved to the folder: "results/{dataset}/classification/{rate_type} depending on which rate_type you specified above. 

## Contacts

With any questions, you can contact me via alisa.barkar@talecom-paris.fr or alisa.george.barkar@gmail.com


