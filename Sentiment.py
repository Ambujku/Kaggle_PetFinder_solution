# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# taken help from https://www.kaggle.com/skooch/corrected-catboostregressor kernal
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Importing the Training Dataset

df = pd.read_csv('/train/train.csv')

#df = df.drop(["Name", "VideoAmt", "Description", "PetID", "PhotoAmt", "RescuerID"], axis = 1)

#X = df.iloc[:, :17]
Y = df["AdoptionSpeed"]

# trying with json metadata and sentiment

train_id = df['PetID']
df = df.drop(["PetID", "AdoptionSpeed"], axis = 1)
df.info()

doc_sent_mag = []
doc_sent_score = []

nf_count = 0

for pet in train_id:
    try:
        with open('../input/train_sentiment/' + pet + '.json', 'r', encoding="utf8") as f:
            sentiment = json.load(f)
            # print(sentiment['documentSentiment']['magnitude'])
            doc_sent_mag.append(sentiment['documentSentiment']['magnitude'])
            doc_sent_score.append(sentiment['documentSentiment']['score'])
    except FileNotFoundError:
        nf_count += 1
        doc_sent_mag.append(-1)
        doc_sent_score.append(-1)

df.loc[:, 'doc_sent_mag'] = doc_sent_mag
df.loc[:, 'doc_sent_score'] = doc_sent_score


vertex_xs = []
vertex_ys = []
bounding_confidences = []
bounding_importance_fracs = []
dominant_blues = []
dominant_greens = []
dominant_reds = []
dominant_pixel_fracs = []
dominant_scores = []
label_descriptions = []
label_scores = []
nf_count = 0
nl_count = 0
for pet in train_id:
    try:
        with open('../input/train_metadata/' + pet + '-1.json', 'r', encoding="utf8") as f:
            data = json.load(f)
            vertex_x = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['x']
            vertex_xs.append(vertex_x)
            vertex_y = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['y']
            vertex_ys.append(vertex_y)
            bounding_confidence = data['cropHintsAnnotation']['cropHints'][0]['confidence']
            bounding_confidences.append(bounding_confidence)
            bounding_importance_frac = data['cropHintsAnnotation']['cropHints'][0].get('importanceFraction', -1)
            bounding_importance_fracs.append(bounding_importance_frac)
            dominant_blue = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['blue']
            dominant_blues.append(dominant_blue)
            dominant_green = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['green']
            dominant_greens.append(dominant_green)
            dominant_red = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['red']
            dominant_reds.append(dominant_red)
            dominant_pixel_frac = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['pixelFraction']
            dominant_pixel_fracs.append(dominant_pixel_frac)
            dominant_score = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['score']
            dominant_scores.append(dominant_score)
            if data.get('labelAnnotations'):
                label_description = data['labelAnnotations'][0]['description']
                label_descriptions.append(label_description)
                label_score = data['labelAnnotations'][0]['score']
                label_scores.append(label_score)
            else:
                nl_count += 1
                label_descriptions.append('nothing')
                label_scores.append(-1)
    except FileNotFoundError:
        nf_count += 1
        vertex_xs.append(-1)
        vertex_ys.append(-1)
        bounding_confidences.append(-1)
        bounding_importance_fracs.append(-1)
        dominant_blues.append(-1)
        dominant_greens.append(-1)
        dominant_reds.append(-1)
        dominant_pixel_fracs.append(-1)
        dominant_scores.append(-1)
        label_descriptions.append('nothing')
        label_scores.append(-1)

print(nf_count)
print(nl_count)
df.loc[:, 'vertex_x'] = vertex_xs
df.loc[:, 'vertex_y'] = vertex_ys
df.loc[:, 'bounding_confidence'] = bounding_confidences
df.loc[:, 'bounding_importance'] = bounding_importance_fracs
df.loc[:, 'dominant_blue'] = dominant_blues
df.loc[:, 'dominant_green'] = dominant_greens
df.loc[:, 'dominant_red'] = dominant_reds
df.loc[:, 'dominant_pixel_frac'] = dominant_pixel_fracs
df.loc[:, 'dominant_score'] = dominant_scores
df.loc[:, 'label_description'] = label_descriptions
df.loc[:, 'label_score'] = label_scores


from sklearn.feature_extraction.text import TfidfVectorizer

train_desc = df.Description.fillna("none").values

tfv = TfidfVectorizer(min_df=3, max_features=10000,
strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1,
stop_words = 'english')

tfv.fit(list(train_desc))

desc = tfv.transform(train_desc)

from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=120)
svd.fit(desc)
desc = svd.transform(desc)

desc = pd.DataFrame(desc, columns=['svd_{}'.format(i) for i in range(120)])

df = pd.concat((df, desc), axis=1)

df = df.drop(["Name", "Description", "RescuerID"], axis =1 )

# removing label desc since categorical and other values are provided

df = df.drop("label_description", axis=1)


# Split train and test data set

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(df, Y, test_size = 0.25)

Y_test = Y_test.tolist()

'''
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 10, random_state = 42)
rf.fit(X_train, Y_train)


Y_pred = rf.predict(X_test)

#convert to list both Y_pred and tointeger

Y_pred = Y_pred.tolist()

for i in range(len(Y_pred)):
Y_pred[i] = int(round(Y_pred[i]))



cm_rf = confusion_matrix(Y_test, Y_pred)

sum = 0
for i in range(len(cm_rf)):
sum = sum + cm_rf[i][i]

acc_rf = sum/len(Y_test)


from xgboost import XGBClassifier

# parameter tuning with learning_rate and estimator:
#lr = [0.1, 0.2,0.3, 0.4]
#estimator = [10, 100, 300, 400, 500, 1000, 2000]
#accuracy = []
#acc_estimator =[]

#for j in estimator:
xgb = XGBClassifier(learning_rate=0.2,n_estimators=400, booster='dart')
xgb.fit(X_train, Y_train)

Y_pred_xgb = xgb.predict(X_test)
Y_pred_xgb = Y_pred_xgb.tolist()
cm_xgb = confusion_matrix(Y_test, Y_pred_xgb)
sum = 0
for i in range(len(cm_xgb)):
sum = sum + cm_xgb[i][i]
acc_xgb = sum/len(Y_test)
'''
# LGBM classifier training
from lightgbm import LGBMClassifier

clf = LGBMClassifier(
n_estimators=400,
learning_rate=0.03,
num_leaves=30,
colsample_bytree=.8,
subsample=.9,
max_depth=7,
reg_alpha=.1,
reg_lambda=.1,
min_split_gain=.01,
min_child_weight=2,
silent=-1,
verbose=-1,
)

clf.fit(X_train, Y_train)

Y_pred_lgbm = clf.predict(X_test)

Y_pred_lgbm = Y_pred_lgbm.tolist()

cm_lgbm = confusion_matrix(Y_test, Y_pred_lgbm)
sum = 0
for i in range(len(cm_lgbm)):
    sum = sum + cm_lgbm[i][i]
acc_lgbm = sum/len(Y_test)


df_test = pd.read_csv('test/test.csv')

test_id = df_test['PetID']


test_doc_sent_mag = []
test_doc_sent_score = []

nf_count = 0

for pet in test_id:
    try:
        with open('../input/test_sentiment/' + pet + '.json', 'r', encoding="utf8") as f:
        sentiment = json.load(f)
        # print(sentiment['documentSentiment']['magnitude'])
        test_doc_sent_mag.append(sentiment['documentSentiment']['magnitude'])
        test_doc_sent_score.append(sentiment['documentSentiment']['score'])
    except FileNotFoundError:
        nf_count += 1
        test_doc_sent_mag.append(-1)
        test_doc_sent_score.append(-1)
        
        df_test.loc[:, 'doc_sent_mag'] = test_doc_sent_mag
        df_test.loc[:, 'doc_sent_score'] = test_doc_sent_score


vertex_xs = []
vertex_ys = []
bounding_confidences = []
bounding_importance_fracs = []
dominant_blues = []
dominant_greens = []
dominant_reds = []
dominant_pixel_fracs = []
dominant_scores = []
label_descriptions = []
label_scores = []
nf_count = 0
nl_count = 0
for pet in test_id:
    try:
        with open('../input/test_metadata/' + pet + '-1.json', 'r', encoding="utf8") as f:
            data = json.load(f)
            vertex_x = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['x']
            vertex_xs.append(vertex_x)
            vertex_y = data['cropHintsAnnotation']['cropHints'][0]['boundingPoly']['vertices'][2]['y']
            vertex_ys.append(vertex_y)
            bounding_confidence = data['cropHintsAnnotation']['cropHints'][0]['confidence']
            bounding_confidences.append(bounding_confidence)
            bounding_importance_frac = data['cropHintsAnnotation']['cropHints'][0].get('importanceFraction', -1)
            bounding_importance_fracs.append(bounding_importance_frac)
            dominant_blue = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['blue']
            dominant_blues.append(dominant_blue)
            dominant_green = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['green']
            dominant_greens.append(dominant_green)
            dominant_red = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['color']['red']
            dominant_reds.append(dominant_red)
            dominant_pixel_frac = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['pixelFraction']
            dominant_pixel_fracs.append(dominant_pixel_frac)
            dominant_score = data['imagePropertiesAnnotation']['dominantColors']['colors'][0]['score']
            dominant_scores.append(dominant_score)
            if data.get('labelAnnotations'):
                label_description = data['labelAnnotations'][0]['description']
                label_descriptions.append(label_description)
                label_score = data['labelAnnotations'][0]['score']
                label_scores.append(label_score)
            else:
                nl_count += 1
                label_descriptions.append('nothing')
                label_scores.append(-1)
    except FileNotFoundError:
        nf_count += 1
        vertex_xs.append(-1)
        vertex_ys.append(-1)
        bounding_confidences.append(-1)
        bounding_importance_fracs.append(-1)
        dominant_blues.append(-1)
        dominant_greens.append(-1)
        dominant_reds.append(-1)
        dominant_pixel_fracs.append(-1)
        dominant_scores.append(-1)
        label_descriptions.append('nothing')
        label_scores.append(-1)

print(nf_count)
print(nl_count)
df_test.loc[:, 'vertex_x'] = vertex_xs
df_test.loc[:, 'vertex_y'] = vertex_ys
df_test.loc[:, 'bounding_confidence'] = bounding_confidences
df_test.loc[:, 'bounding_importance'] = bounding_importance_fracs
df_test.loc[:, 'dominant_blue'] = dominant_blues
df_test.loc[:, 'dominant_green'] = dominant_greens
df_test.loc[:, 'dominant_red'] = dominant_reds
df_test.loc[:, 'dominant_pixel_frac'] = dominant_pixel_fracs
df_test.loc[:, 'dominant_score'] = dominant_scores
df_test.loc[:, 'label_description'] = label_descriptions
df_test.loc[:, 'label_score'] = label_scores

test_desc = df_test.Description.fillna("none").values

#tfv = TfidfVectorizer(min_df=3, max_features=10000,
# strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
# ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1,
# stop_words = 'english')

tfv.fit(list(test_desc))

desc_test = tfv.transform(test_desc)

from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=120)
svd.fit(desc_test)
desc_test = svd.transform(desc_test)

desc_test = pd.DataFrame(desc_test, columns=['svd_{}'.format(i) for i in range(120)])

df_test = pd.concat((df_test, desc_test), axis=1)


df_test = df_test.drop(["Name", "RescuerID", "Description", "PetID"], axis = 1)

df_test = df_test.drop("label_description", axis=1)

test_pred = clf.predict(df_test)
test_pred = test_pred.tolist()

submit = pd.DataFrame({"AdoptionSpeed":test_pred})

submit.insert(loc=0, column='PetID', value=df_test["PetID"])

submit.to_csv('submission.csv', index=False)