import pandas as pd
import numpy as np
import tensorflow
import keras
from tensorflow.keras.optimizers import Adam, Adagrad, SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Dropout
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.callbacks import EarlyStopping
import myFunctions as do
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler


x = pd.read_csv('data/data.csv')
# y = pd.read_csv('octoberFightsUpdated.csv')
y = pd.read_csv('data/NovemFights.csv')
# print(x.columns)
'''
['EVENTDATE', 'COUNTRY', 'ROUNDS', 'FIGHTER1', 'FIGHTERa', 'STANCE',
       'DOB', 'COUNTRYOFEVENT', 'HEIGHT', 'WEIGHT', 'REACH', 'STRIKES_LANDED',
       'STRIKING_ACCURACY', 'STRIKES_ABSORBED', 'STRIKING_DEFENCE',
       'TAKEDOWNS_LANDED', 'TAKEDOWN_ACCURACY', 'TAKEDOWN_DEFENCE',
       'SUBMISSION_AVERAGE', 'KNOCKDOWN_RATIO', 'AVERAGE_FIGHTTIME',
       'STRIKES_STANDING', 'CLINCH_STRIKES', 'GROUND_STRIKES', 'HEAD_STRIKES',
       'BODY_STRIKES', 'LEG_STRIKES', 'FIGHTER1RECENT1', 'FIGHTER1RECENT2',
       'FIGHTER1RECENT3', 'FIGHTER1RECENT4', 'FIGHTER1NUMUFCFIGHTS',
       'FIGHTER1RECENTBONUSES', 'FIGHTER1WINRECORD', 'FIGHTER1LOSSRECORD',
       'FIGHTER1LAYOFFTIME', 'FIGHTER2', 'FIGHTERb', 'STANCE2', 'DOB2',
       'COUNTRY2', 'HEIGHT2', 'WEIGHT2', 'REACH2', 'STRIKES_LANDED2',
       'STRIKING_ACCURACY2', 'STRIKES_ABSORBED2', 'STRIKING_DEFENCE2',
       'TAKEDOWNS_LANDED2', 'TAKEDOWN_ACCURACY2', 'TAKEDOWN_DEFENCE2',
       'SUBMISSION_AVERAGE2', 'KNOCKDOWN_RATIO2', 'AVERAGE_FIGHTTIME2',
       'STRIKES_STANDING2', 'CLINCH_STRIKES2', 'GROUND_STRIKES2',
       'HEAD_STRIKES2', 'BODY_STRIKES2', 'LEG_STRIKES2', 'FIGHTER2RECENT1',
       'FIGHTER2RECENT2', 'FIGHTER2RECENT3', 'FIGHTER2RECENT4',
       'FIGHTER2NUMUFCFIGHTS', 'FIGHTER2RECENTBONUSES', 'FIGHTER2WINRECORD',
       'FIGHTER2LOSSRECORD', 'FIGHTER2LAYOFFTIME', 'FIGHTER1OUTCOME',
       'FIGHTER1WIN'],
'''
from sklearn.preprocessing import LabelEncoder



def encode_age(dobs):
    encoded_ages =[]
    for dob in dobs:
        age = 2020-dob
        if age <= 25:
            encoded_ages.append([1])
        elif age <=34:
            encoded_ages.append([2])
        elif age <=39:
            encoded_ages.append([3])
        else:
            encoded_ages.append([4])
    return encoded_ages


def encode_both_fighter_features(fighter1_feature, fighter2_feature, data_set,prediction_set):
    x_feature1 = data_set[fighter1_feature].values
    x_feature2 = data_set[fighter2_feature].values
    y_feature1 = prediction_set[fighter1_feature].values
    y_feature2 = prediction_set[fighter2_feature].values
    x_enc_feature1, y_enc_feature1 = encode_feature_sets(x_feature1,y_feature1)
    x_enc_feature2, y_enc_feature2 = encode_feature_sets(x_feature2, y_feature2)
    return [[x_enc_feature1, x_enc_feature2], [y_enc_feature1, y_enc_feature2]]

def encode_feature(feature):
    encoder = LabelEncoder()
    encoded_feature = encoder.fit_transform(feature)
    return encoded_feature

#fits training set and transforms test set
def encode_feature_sets(feature,prediction_set_feature):
    encoder = LabelEncoder()
    encoded_feature = encoder.fit_transform(feature)
    encoded_transformed_feature = encoder.transform(prediction_set_feature)
    return encoded_feature, encoded_transformed_feature


def encode_home_advantages(data_frame):
    home_advantage = []
    red = data_frame['COUNTRY'].values
    blue = data_frame['COUNTRY2'].values
    event = data_frame['COUNTRYOFEVENT'].values
    for i in range(len(event)):
        if red[i] == blue[i]:
            home_advantage.append([0])
        elif red[i] == event[i]:
            home_advantage.append([1])
        elif blue[i] == event[i]:
            home_advantage.append([2])
        else:
            home_advantage.append([0])
    home_advantage = np.asarray(home_advantage)
    return home_advantage


def calc_height_advantage(data_set):
    mms = MinMaxScaler(feature_range=(0,10))
    fighter1 = data_set['HEIGHT'].values
    fighter2 = data_set['HEIGHT2'].values
    height_advantage= []
    for i in range(len(fighter1)):
        height_advantage.append([fighter1[i]-fighter2[i]])
    mms.fit_transform(height_advantage)
    return np.asarray(height_advantage)


def calc_reach_advantage(data_set):
    mms = MinMaxScaler(feature_range=(0, 10))
    fighter1 = data_set['REACH'].values
    fighter2 = data_set['REACH2'].values
    reach_advantage = []
    for i in range(len(fighter1)):
        reach_advantage.append([fighter1[i] - fighter2[i]])
    mms.fit_transform(reach_advantage)
    return np.asarray(reach_advantage)

def calc_age_advantage(data_set):
    mms = MinMaxScaler(feature_range=(0, 5))
    fighter1 = data_set['DOB'].values
    fighter2 = data_set['DOB2'].values
    age_advantage =[]
    for i in range(len(fighter1)):
        age_advantage.append([fighter1[i]-fighter2[i]])
        mms.fit_transform(age_advantage)
    return np.asarray(age_advantage)


def encode_weight_classes(data_frame):
    weight = data_frame['WEIGHT'].values
    weight_class = []
    i = 0
    for i in weight :
        if i>206:
            weight_class.append([1])#HV
        elif i>186:
            weight_class.append([2])#LHW
        elif i>170:
            weight_class.append([3])#middleweight
        elif i>156:
            weight_class.append([4])#ww
        elif i>146:
            weight_class.append([5])#LW
        elif i>136:
            weight_class.append([6])#FW
        elif i>126:
             weight_class.append([7])#bantam
        else:
             weight_class.append([8])  # bantam
    return np.asarray(weight_class)

def encode_past_outcomes(past_outcome):
    encoded_outcomes=[]
    for i in past_outcome:
        if i == "KO/TKOWIN":
           encoded_outcomes.append([1])
        elif i == "SUBWIN":
           encoded_outcomes.append([2])
        elif i=="U-DECWIN" or i=="M-DECWIN":
           encoded_outcomes.append([3])
        elif i=="S-DECWIN":
           encoded_outcomes.append([4])
        elif i=="S-DECLOSS":
            encoded_outcomes.append([5])
        elif i=="U-DECLOSS" or i =="M-DECLOSS":
            encoded_outcomes.append([6])
        elif i == "SUBLOSS":
            encoded_outcomes.append([7])
        elif i == "KO/TKOLOSS":
            encoded_outcomes.append([8])
        else:
            encoded_outcomes.append([0])
    return np.asarray(encoded_outcomes)


def data_formating_pipeline(data, prediction_data):
    outcomes = data['FIGHTER1OUTCOME'].values.astype('str')
    encoder = LabelEncoder()
    labels = encoder.fit_transform(outcomes)
    print(encoder.classes_)

    x_ages = encode_age(data['DOB'].values.astype('float64'))
    x_reach_advantage = calc_reach_advantage(data)
    x_height_advantage = calc_height_advantage(data)
    x_weight_class = encode_weight_classes(data)
    x_age_advantages = calc_age_advantage(data)
    x_home_advantages = encode_home_advantages(data)

    y_ages = encode_age(prediction_data['DOB'].values.astype('float64'))
    y_reach_advantage = calc_reach_advantage(prediction_data)
    y_height_advantage = calc_height_advantage(prediction_data)
    y_weight_class = encode_weight_classes(prediction_data)
    y_age_advantages = calc_age_advantage(prediction_data)
    y_home_advantages = encode_home_advantages(prediction_data)

    x_stances, y_stances = encode_both_fighter_features('STANCE', 'STANCE2', data, prediction_data)

    x_rounds = encoder.fit_transform(data['ROUNDS'].values.astype('float64'))
    y_rounds = encoder.transform(prediction_data['ROUNDS'].values.astype('float64'))



    x_features_to_scale = data[[
       'STRIKES_LANDED', 'STRIKING_ACCURACY', 'STRIKES_ABSORBED', 'STRIKING_DEFENCE',
       'TAKEDOWNS_LANDED', 'TAKEDOWN_ACCURACY', 'TAKEDOWN_DEFENCE', 'SUBMISSION_AVERAGE',
       'AVERAGE_FIGHTTIME', 'STRIKES_STANDING', 'CLINCH_STRIKES', 'GROUND_STRIKES',
       'HEAD_STRIKES','BODY_STRIKES', 'LEG_STRIKES','FIGHTER1NUMUFCFIGHTS',
       'FIGHTER1RECENTBONUSES','FIGHTER1WINRECORD', 'FIGHTER1LOSSRECORD', 'FIGHTER1LAYOFFTIME',
       'STRIKES_LANDED2', 'STRIKING_ACCURACY2', 'STRIKES_ABSORBED2', 'STRIKING_DEFENCE2',
       'TAKEDOWNS_LANDED2', 'TAKEDOWN_ACCURACY2', 'TAKEDOWN_DEFENCE2','SUBMISSION_AVERAGE2',
       'AVERAGE_FIGHTTIME2','STRIKES_STANDING2', 'CLINCH_STRIKES2', 'GROUND_STRIKES2',
       'HEAD_STRIKES2', 'BODY_STRIKES2', 'LEG_STRIKES2','FIGHTER2NUMUFCFIGHTS',
       'FIGHTER2RECENTBONUSES', 'FIGHTER2WINRECORD','FIGHTER2LOSSRECORD', 'FIGHTER2LAYOFFTIME']].values.astype('float64')
    y_features_to_scale = prediction_data[[
       'STRIKES_LANDED', 'STRIKING_ACCURACY', 'STRIKES_ABSORBED', 'STRIKING_DEFENCE',
       'TAKEDOWNS_LANDED', 'TAKEDOWN_ACCURACY', 'TAKEDOWN_DEFENCE', 'SUBMISSION_AVERAGE',
       'AVERAGE_FIGHTTIME', 'STRIKES_STANDING', 'CLINCH_STRIKES', 'GROUND_STRIKES',
       'HEAD_STRIKES','BODY_STRIKES', 'LEG_STRIKES','FIGHTER1NUMUFCFIGHTS',
       'FIGHTER1RECENTBONUSES','FIGHTER1WINRECORD', 'FIGHTER1LOSSRECORD', 'FIGHTER1LAYOFFTIME',
       'STRIKES_LANDED2', 'STRIKING_ACCURACY2', 'STRIKES_ABSORBED2', 'STRIKING_DEFENCE2',
       'TAKEDOWNS_LANDED2', 'TAKEDOWN_ACCURACY2', 'TAKEDOWN_DEFENCE2','SUBMISSION_AVERAGE2',
       'AVERAGE_FIGHTTIME2','STRIKES_STANDING2', 'CLINCH_STRIKES2', 'GROUND_STRIKES2',
       'HEAD_STRIKES2', 'BODY_STRIKES2', 'LEG_STRIKES2','FIGHTER2NUMUFCFIGHTS',
       'FIGHTER2RECENTBONUSES', 'FIGHTER2WINRECORD','FIGHTER2LOSSRECORD', 'FIGHTER2LAYOFFTIME']].values.astype('float64')

    scaler = StandardScaler()

    x_scaled_features = scaler.fit_transform(x_features_to_scale)
    y_scaled_features = scaler.fit_transform(y_features_to_scale)
    data_set = np.concatenate((x_scaled_features, x_ages, x_age_advantages, x_reach_advantage, x_height_advantage,
                               x_weight_class,x_stances[0], x_home_advantages), axis=1)
    prediction_dataset = np.concatenate((y_scaled_features, y_ages, y_age_advantages, y_reach_advantage, y_height_advantage,
                                         y_weight_class, y_home_advantages), axis=1)
    return data_set, prediction_dataset


data, pred_data = data_formating_pipeline(x,y)

np.save('data/data.npy',data)
np.save('data/pred_data.npy',pred_data)
