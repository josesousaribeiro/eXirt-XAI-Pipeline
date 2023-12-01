#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 10:40:10 2023

@author: josedesousa
"""

import time
import os
import importlib.util
import platform
import wget

import openml
import pandas as pd
import statistics
import numpy as np
import random
import copy
import warnings


from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics

from collections import OrderedDict
from scipy import linalg
from adjustText import adjust_text
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

os.system("pip install --upgrade --upgrade-strategy only-if-needed scikit-learn==0.22")

os.system("pip install skater==1.0.2")

package_name = 'lofo-importance'
spec = importlib.util.find_spec(package_name)
if spec is None:
    os.system("pip install lofo-importance")
    
package_name = 'shap'
spec = importlib.util.find_spec(package_name)
if spec is None:
    os.system("pip install shap")

package_name = 'eli5'
spec = importlib.util.find_spec(package_name)
if spec is None:
    os.system("pip install eli5")
    
package_name = 'dalex'
spec = importlib.util.find_spec(package_name)
if spec is None:
    os.system("pip install dalex")
    
package_name = 'py-ciu'
spec = importlib.util.find_spec(package_name)
if spec is None:
    os.system("pip install py-ciu")
    
package_name = 'plotly'
spec = importlib.util.find_spec(package_name)
if spec is None:
    os.system("pip install plotly")
    
package_name = 'adjustText'
spec = importlib.util.find_spec(package_name)
if spec is None:
    os.system("pip install adjustText")
    
package_name = 'scikit_posthocs'
spec = importlib.util.find_spec(package_name)
if spec is None:
    os.system("pip install scikit_posthocs")
    
package_name = 'lightgbm'
spec = importlib.util.find_spec(package_name)
if spec is None:
    os.system("pip install lightgbm")
    
package_name = 'catboost'
spec = importlib.util.find_spec(package_name)
if spec is None:
    os.system("pip install catboost")
    
package_name = 'openml'
spec = importlib.util.find_spec(package_name)
if spec is None:
    os.system("pip install openml")
    
package_name = 'rpy2'
spec = importlib.util.find_spec(package_name)
if spec is None:
    os.system("pip install rpy2")

package_name = 'wget'
spec = importlib.util.find_spec(package_name)
if spec is None:
    os.system("pip install wget")

package_name = 'catsim'
spec = importlib.util.find_spec(package_name)
if spec is None:
    os.system("pip install catsim")

#INSTALL EXIRT

package_name = 'eXirt'
spec = importlib.util.find_spec(package_name)
if spec is None:
    os.system("pip install eXirt")

if os.path.isfile(os.path.join(os.getcwd(),'decodIRT_MLtIRT.py')) == False:
                  wget.download('https://raw.githubusercontent.com/josesousaribeiro/eXirt/main/pyexirt/decodIRT_MLtIRT.py')

if os.path.isfile(os.path.join(os.getcwd(),'decodIRT_analysis.py')) == False:
                  wget.download('https://raw.githubusercontent.com/josesousaribeiro/eXirt/main/pyexirt/decodIRT_analysis.py')



import eli5
import shap
import dalex as dx
import numpy as np
import pandas as pd
import openml
import pickle
import matplotlib.pyplot as plt
import statistics
import os
import gc
import math
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import seaborn as sns
import scikit_posthocs as sp
import copy
import lightgbm as lgb
import random


from sklearn.ensemble import AdaBoostClassifier
from ciu import determine_ciu
from lofo import LOFOImportance, FLOFOImportance, Dataset
from skater.core.explanations import Interpretation
from skater.model import InMemoryModel
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from scipy.stats import friedmanchisquare
from scipy.stats import spearmanr
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import silhouette_samples, silhouette_score

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from matplotlib import cm
from matplotlib.colors import ListedColormap
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn import preprocessing
from matplotlib.pyplot import figure
from catboost import CatBoostClassifier
from eli5.sklearn import PermutationImportance



from sklearn.datasets import load_wine
from sklearn.datasets import load_iris


from re import A
from scipy.special import expit
from numpy import percentile, concatenate, array, linspace, append

from pyexirt.eXirt import Explainer

do_download_workspace = True
do_download = False

bar = ''

if platform.system() == 'Windows':
    bar = '\\'
else:
    bar =  '/'

path_content = '.'+bar+'content_eXirt'
path_fig = bar+'out_fig'
path_csv = bar+'out_csv'
path_model = bar+'out_model'
path_irt = bar+'out_irt'
path_dataset = bar+'*'

os.system('rm -r '+path_content+path_fig)
os.system('rm -r '+path_content+path_csv)
os.system('rm -r '+path_content+path_model)
os.system('rm -r '+path_content+path_irt)


os.system('mkdir '+path_content+path_fig)
os.system('mkdir '+path_content+path_csv)
os.system('mkdir '+path_content+path_model)
os.system('mkdir '+path_content+path_irt)

def dirByDataset(datasetName):
  os.system('mkdir '+path_content+path_fig+datasetName)
  os.system('mkdir '+path_content+path_csv+datasetName)
  os.system('mkdir '+path_content+path_model+datasetName)
  os.system('mkdir '+path_content+path_irt+datasetName)

print("===========================================================================")
print("============================ XAI TOOLS ====================================")
print("===========================================================================")
print("")

#EXirt
explainer = Explainer()


class ExplainableTools():
  def explainRankByLofo(model,X,Y,names_x_attributes):
    df = X.copy()
    df['class'] = Y.to_list()
    dataset = Dataset(df=df, target="class", features=names_x_attributes)
    fi = LOFOImportance(dataset, scoring='accuracy', model=model)
    importances = fi.get_importance()
    importances = importances.sort_values(by=['importance_mean','feature'],ascending=False) #fix problem of equals values of explaination
    return importances['feature'].to_list()

  def explainRankByEli5(model, X, Y):
      perm = PermutationImportance(model, random_state=42).fit(X, Y)
      rank = eli5.show_weights(perm, feature_names = X.columns.tolist(),top=-1)
      rank = pd.read_html(rank.data)[0]
      rank = rank.sort_values(by=['Weight','Feature'], ascending=False) #fix problem of equals values of explaination
      return rank['Feature'].to_list()


  def explainRankByKernelShap(model,x_features_names, X,is_gradient=False): # shap.sample(data, K) or shap.kmeans(data, K)
    np.random.seed(0)
    explainer = shap.KernelExplainer(model.predict_proba, X[:],nsamples=len(x_features_names))
    shap_values = explainer.shap_values(X[:])
    if is_gradient == False:
        vals= np.abs(shap_values).sum(1)
    else:
        vals= np.abs([shap_values]).sum(1) #correction []
    temp_df = pd.DataFrame(list(zip(x_features_names, sum(vals))), columns=['feat_name','shap_value'])
    temp_df = temp_df.sort_values(by=['shap_value','feat_name'], ascending=False) #fix problem of equals values of explaination
    return list(temp_df['feat_name'])


  def explainRankByTreeShap(model, x_features_names, X, is_gradient=False):
      np.random.seed(0)
      shap_values = shap.TreeExplainer(model, feature_perturbation='interventional').shap_values(X)
      if is_gradient == False:
        vals= np.abs(shap_values).mean(0)
      else:
        vals= np.abs([shap_values]).mean(0) #correction []
      temp_df = pd.DataFrame(list(zip(x_features_names, sum(vals))), columns=['feat_name','shap_value'])
      temp_df = temp_df.sort_values(by=['shap_value','feat_name'], ascending=False) #fix problem of equals values of explaination

      return temp_df['feat_name'].to_list()

  def explainRankByCiu(model, x_test, X, feature_names,context_dic,rank):

    def _makeRankByCu(ciu):
      df_cu = pd.DataFrame(list(ciu.cu.items()), columns=['attribute', 'cu'])
      df_cu = df_cu.sort_values(by='cu', ascending=False)
      #ciu.plot_cu()
      return df_cu['attribute'].to_list()

    def _makeRankByCi(ciu):
      df_ci = pd.DataFrame(list(ciu.ci.items()), columns=['attribute', 'ci'])
      df_ci = df_ci.sort_values(by=['ci','attribute'], ascending=False) #fix problem of equals values of explaination
      return df_ci['attribute'].to_list()

    case = x_test.values[0]
    example_prediction = model.predict([x_test.values[0]])
    example_prediction_probs = model.predict_proba([x_test.values[0]])
    prediction_index = list(example_prediction_probs[0]).index(max(example_prediction_probs[0]))




    ciu = determine_ciu(
        x_test.iloc[[1]],
        model.predict_proba,
        X.to_dict('list'),
        samples = 1000,
        prediction_index = 1)

    if rank == 'ci':
      result = _makeRankByCi(ciu)
    else:
      if rank == 'cu':
        result = _makeRankByCu(ciu)
      else:
        result = {}

    #ciu
    return result


  def explainRankSkater(model, X):
    interpreter = Interpretation(X.to_numpy(), feature_names=X.columns.to_list())

    model_new = InMemoryModel(model.predict_proba,
                              examples=X.to_numpy(),
                              unique_values=model.classes_)

    rank = interpreter.feature_importance.feature_importance(model_new,
                                                             ascending=False,
                                                             progressbar=False
                                                             #n_jobs=1
                                                             )
    rank = rank.to_frame(name='values')
    rank = rank.reset_index()
    rank = rank.rename(columns={'index':'variable','values':'values'})
    rank = rank.sort_values(by=['values','variable'], ascending=False) #fix problem of equals values of explaination
    return rank['variable'].to_list()

  def explainRankDalex(model, X_train, y_train):
    explainer = dx.Explainer(model, X_train, y_train,verbose=False)
    explanation = explainer.model_parts()
    rank = explanation.result
    rank = rank[rank.variable != '_baseline_']
    rank = rank[rank.variable != '_full_model_']
    rank = rank.sort_values(by=['dropout_loss','variable'], ascending=False) #fix problem of equals values of explaination
    return rank['variable'].tolist()

  #Calculando ganho de informacao
  def explainRankByInfoGain(X,Y, attribute_X_names):
    def InfoGain(data,attribute_X_name,target_name):
      """
      Calculate the information gain of a dataset. This function takes three parameters:
      1. data = The dataset for whose feature the IG should be calculated
      2. attribute_X_name = the name of the feature for which the information gain should be calculated
      3. target_name = the name of the target feature. The default for this example is "class"
      """

      def entropy(target_col):
        """
        Calculate the entropy of a dataset.
        The only parameter of this function is the target_col parameter which specifies the target column
        """
        elements,counts = np.unique(target_col,return_counts = True)
        entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
        return entropy



      #Calculate the entropy of the total dataset
      total_entropy = entropy(data[target_name])

      ##Calculate the entropy of the dataset

      #Calculate the values and the corresponding counts for the split attribute
      vals,counts= np.unique(data[attribute_X_name],return_counts=True)

      #Calculate the weighted entropy
      Weighted_Entropy = np.sum([(counts[i]/np.sum(counts))*entropy(data.where(data[attribute_X_name]==vals[i]).dropna()[target_name]) for i in range(len(vals))])

      #Calculate the information gain
      result = total_entropy - Weighted_Entropy

      return result

    data = X.copy()
    target_name = 'class'
    data[target_name] = Y.to_list()

    rank = pd.Series([InfoGain(data,feature,target_name) for feature in attribute_X_names], index=attribute_X_names).sort_values(ascending=False)
    rank = rank.to_frame(name='values')
    rank = rank.reset_index()
    rank = rank.rename(columns={'index':'variable','values':'values'})
    rank = rank.sort_values(by=['values','variable'],ascending=False) #fix problem of equals values of explaination
    return rank['variable'].to_list()

print("===========================================================================")
print("============================ PRE-PROCESS ==================================")
print("===========================================================================")
print("")

class PreprocessDefault():

  def z_score_serie(s):
    # copy the dataframe
    s_std = s.copy()
    s_std = (s_std - s_std.mean()) / s_std.std()
    return s_std

  def z_score(df):
    # copy the dataframe
    df_std = df.copy()
    # apply the z-score method
    for column in df_std.columns:
        df_std[column] = (df_std[column] - df_std[column].mean()) / df_std[column].std()

    return df_std

  def normalize(df):
    # copy the dataframe
    df_norm = df.copy()
    # apply min-max scaling
    for column in df_norm.columns:
      if df[column].dtype != 'category':
        if(len(df_norm[column].unique()) > 1): #fix NaN generation
          df_norm[column] = (df_norm[column] - df_norm[column].min()) / (df_norm[column].max() - df_norm[column].min())
        else:
          df_norm[column] = 0
    return df_norm


  def PreprocessXY(X, Y):
    X = X.astype(float)
    Y = Y.astype(float)

    return X, Y

  def PreprocessXByDataset(X,datasetName):
          flagThereArePreprocess = False

          if datasetName == "credit-g":
              flagThereArePreprocess = True

              cleanup_nums = {
                      "checking_status": {"no checking":394, "<0": 274, "0<=X<200":269, ">=200":63},
                      "credit_history": {"existing paid":530, "critical/other existing credit": 293, "delayed previously":88, "all paid":49, "no credits/all paid":40},
                      "purpose": {"radio/tv":280, "new car":234, "furniture/equipment":181, "used car":103, "business": 97, "education":50, "repairs":  22, "domestic appliance":12, "other":12, "retraining":9},
                      "savings_status": {"<100": 603, "no known savings":183, "100<=X<500":103, "500<=X<1000":63, ">=1000":48},
                      "employment": {"1<=X<4":339, ">=7":253, "4<=X<7":174, "<1":172, "unemployed":62},
                      "personal_status": {"male single":548, "female div/dep/mar":310, "male mar/wid":92, "male div/sep":50},
                      "other_parties": {"none":907, "guarantor":52, "co applicant":41},
                      "property_magnitude": {"car":332, "real estate":282, "life insurance":232, "no known property":154},
                      "other_payment_plans": {"none":814, "bank":139, "stores":47},
                      "housing": {"own":713, "rent":179, "for free":108},
                      "job": {"skilled":630, "unskilled resident":200, "high qualif/self emp/mgmt":148, "unemp/unskilled non res":22},
                      "own_telephone": {"none":0, "yes":1},
                      "foreign_worker": {"yes":1, "no":0}
                      }
              obj_df = X.select_dtypes(include=['object']).copy()
              obj_df[obj_df.isnull().any(axis=1)]
              obj_df.head()
              X.replace(cleanup_nums, inplace=True)
              X.head()

          if datasetName == "blood-transfusion-service-center":
              flagThereArePreprocess = True
              print("All numeric")

          if datasetName == "monks-problems-2":
              flagThereArePreprocess = True
              X.attr1 = X.attr1.astype(int);
              X.attr2 = X.attr2.astype(int);
              X.attr3 = X.attr3.astype(int);
              X.attr4 = X.attr4.astype(int);
              X.attr5 = X.attr5.astype(int);
              X.attr6 = X.attr6.astype(int);

          if datasetName == "tic-tac-toe":
              flagThereArePreprocess = True

              cleanup_nums = {
                      "top-left-square": {"x": 418, "o": 335, "b":205},
                      "top-middle-square": {"x": 378, "o": 330, "b": 250},
                      "top-right-square": {"x": 418, "o": 335, "b":205},
                      "middle-left-square": {"x": 378, "o": 330,"b":250},
                      "middle-middle-square": {"x": 458, "o": 340, "b":160},
                      "middle-right-square": {"x": 378, "o": 330, "b":250},
                      "bottom-left-square": {"x": 418, "o": 335, "b": 205},
                      "bottom-middle-square": {"x": 378, "o": 330,"b": 250},
                      "bottom-right-square": {"x": 418, "o": 335,"b":205}
                      }
              obj_df = X.select_dtypes(include=['object']).copy()
              obj_df[obj_df.isnull().any(axis=1)]
              obj_df.head()
              X.replace(cleanup_nums, inplace=True)
              X.head()

          if datasetName == "monks-problems-1":
              flagThereArePreprocess = True
              X.attr1 = X.attr1.astype(int);
              X.attr2 = X.attr2.astype(int);
              X.attr3 = X.attr3.astype(int);
              X.attr4 = X.attr4.astype(int);
              X.attr5 = X.attr5.astype(int);
              X.attr6 = X.attr6.astype(int);


          if datasetName == "steel-plates-fault":
              flagThereArePreprocess = True
              X.columns = [
                            "X_Minimum","X_Maximum","Y_Minimum","Y_Maximum","Pixels_Areas",
                            "X_Perimeter","Y_Perimeter","Sum_of_Luminosity","Minimum_of_Luminosity",
                            "Maximum_of_Luminosity","Length_of_Conveyer","TypeOfSteel_A300","TypeOfSteel_A400",
                            "Steel_Plate_Thickness","Edges_Index","Empty_Index","Square_Index","Outside_X_Index",
                            "Edges_X_Index","Edges_Y_Index","Outside_Global_Index","LogOfAreas",
                            "Log_X_Index","Log_Y_Index","Orientation_Index","Luminosity_Index","SigmoidOfAreas",
                            "Pastry","Z_Scratch","K_Scatch","Stains","Dirtiness","Bumps"
                            ]
              print("All numeric")
              X.head()

          if datasetName == "kr-vs-kp":
              flagThereArePreprocess = True
              cleanup_nums = {
                      "bkblk":{ "f":2839,"t":357},
                      "bknwy":{ "f":2971,"t":225},
                      "bkon8":{ "f":3076,"t":120},
                      "bkona":{ "f":2874,"t":322},
                      "bkspr":{ "f":2129,"t":1067},
                      "bkxbq":{ "f":1722,"t":1474},
                      "bkxcr":{ "f":2026,"t":1170},
                      "bkxwp":{ "f":2500,"t":696},
                      "blxwp":{ "f":1980,"t":1216},
                      "bxqsq":{ "f":2225,"t":971},
                      "cntxt":{ "f":1817,"t":1379},
                      "dsopp":{ "f":2860,"t":336},
                      "dwipd":{ "g":991,"l":2205},
                      "hdchk":{ "f":3181,"t":15},
                      "katri":{ "b":224,"n":2526,"w":446},
                      "mulch":{ "f":3040,"t":156},
                      "qxmsq":{ "f":3099,"t":97},
                      "r2ar8":{ "f":1000,"t":2196},
                      "reskd":{ "f":3170,"t":26},
                      "reskr":{ "f":2714,"t":482},
                      "rimmx":{ "f":2612,"t":584},
                      "rkxwp":{ "f":2556,"t":640},
                      "rxmsq":{ "f":3013,"t":183},
                      "simpl":{ "f":1975,"t":1221},
                      "skach":{ "f":3185,"t":11},
                      "skewr":{ "f":980,"t":2216},
                      "skrxp":{ "f":3021,"t":175},
                      "spcop":{ "f":3195,"t":1},
                      "stlmt":{ "f":3149,"t":47},
                      "thrsk":{ "f":3060,"t":136},
                      "wkcti":{ "f":2631,"t":565},
                      "wkna8":{ "f":3021,"t":175},
                      "wknck":{ "f":1984,"t":1212},
                      "wkovl":{ "f":1189,"t":2007},
                      "wkpos":{ "f":851,"t":2345},
                      "wtoeg":{ "n":2407,"t":789}
                      }
              obj_df = X.select_dtypes(include=['object']).copy()
              obj_df[obj_df.isnull().any(axis=1)]
              obj_df.head()
              X.replace(cleanup_nums, inplace=True)
              X.head()

          if datasetName == "qsar-biodeg":
              flagThereArePreprocess = True
              X = X.astype(float)
              print("All numeric")

          if datasetName == "wdbc":
              flagThereArePreprocess = True
              print("All numeric")

          if datasetName == "phoneme":
              flagThereArePreprocess = True
              print("All numeric")

          if datasetName == "diabetes":
              flagThereArePreprocess = True
              print("All numeric")

          if datasetName == "ozone-level-8hr":
              flagThereArePreprocess = True
              print("All numeric")

          if datasetName == "kc2":
              flagThereArePreprocess = True
              print("All numeric")

          if datasetName == "eeg-eye-state":
              flagThereArePreprocess = True
              print("All numeric")

          if datasetName == "climate-model-simulation-crashes":
              flagThereArePreprocess = True
              print("All numeric")

          if datasetName == "spambase":
              flagThereArePreprocess = True
              print("All numeric")

          if datasetName == "kc1":
              flagThereArePreprocess = True
              print("All numeric")

          if datasetName == "ilpd":
              flagThereArePreprocess = True
              X.V2 = X.V2.map({"Female":1, "Male":0})
              X.V2 = X.V2.astype(int);

          if datasetName == "pc1":
              flagThereArePreprocess = True
              print("All numeric")

          if datasetName == "pc3":
              flagThereArePreprocess = True
              print("All numeric")

          if datasetName == "banknote-authentication":
              flagThereArePreprocess = True
              print("All numeric")

          if datasetName == "pc4":
              flagThereArePreprocess = True
              print("All numeric")

          if datasetName == "mozilla4":
              flagThereArePreprocess = True
              print("All numeric")

          if datasetName == "monks-problems-3":
              flagThereArePreprocess = True

              cleanup_nums = {
                  "attr1":{ "1":192,"2":184,"3":178},
                  "attr2":{ "1":183,"2":186,"3":185},
                  "attr3":{ "1":281,"2":273},
                  "attr4":{ "1":184,"2":182,"3":188},
                  "attr5":{ "1":140,"2":139,"3":136,"4":139},
                  "attr6":{ "1":275,"2":279}
                      }
              obj_df = X.select_dtypes(include=['object']).copy()
              obj_df[obj_df.isnull().any(axis=1)]
              obj_df.head()
              X.replace(cleanup_nums, inplace=True)
              X.head()


          if datasetName == "PhishingWebsites":
              flagThereArePreprocess = True
              cleanup_nums = {
                            "having_IP_Address":{ "-1":3793,"1":7262},
                            "URL_Length":{ "-1":8960,"0":135,"1":1960},
                            "Shortining_Service":{ "-1":1444,"1":9611},
                            "having_At_Symbol":{ "-1":1655,"1":9400},
                            "double_slash_redirecting":{ "-1":1429,"1":9626},
                            "Prefix_Suffix":{ "-1":9590,"1":1465},
                            "having_Sub_Domain":{ "-1":3363,"0":3622,"1":4070},
                            "SSLfinal_State":{ "-1":3557,"0":1167,"1":6331},
                            "Domain_registeration_length":{ "-1":7389,"1":3666},
                            "Favicon":{ "-1":2053,"1":9002},
                            "port":{ "-1":1502,"1":9553},
                            "HTTPS_token":{ "-1":1796,"1":9259},
                            "Request_URL":{ "-1":4495,"1":6560},
                            "URL_of_Anchor":{ "-1":3282,"0":5337,"1":2436},
                            "Links_in_tags":{ "-1":3956,"0":4449,"1":2650},
                            "SFH":{ "-1":8440,"0":761,"1":1854},
                            "Submitting_to_email":{ "-1":2014,"1":9041},
                            "Abnormal_URL":{ "-1":1629,"1":9426},
                            "Redirect":{ "0":9776,"1":1279},
                            "on_mouseover":{ "-1":1315,"1":9740},
                            "RightClick":{ "-1":476,"1":10579},
                            "popUpWidnow":{ "-1":2137,"1":8918},
                            "Iframe":{ "-1":1012,"1":10043},
                            "age_of_domain":{ "-1":5189,"1":5866},
                            "DNSRecord":{ "-1":3443,"1":7612},
                            "web_traffic":{ "-1":2655,"0":2569,"1":5831},
                            "Page_Rank":{ "-1":8201,"1":2854},
                            "Google_Index":{ "-1":1539,"1":9516},
                            "Links_pointing_to_page":{ "-1":548,"0":6156,"1":4351},
                            "Statistical_report":{ "-1":1550,"1":9505}
                        }
              obj_df = X.select_dtypes(include=['object']).copy()
              obj_df[obj_df.isnull().any(axis=1)]
              obj_df.head()
              X.replace(cleanup_nums, inplace=True)
              X.head()


          if datasetName == "churn":
              flagThereArePreprocess = True
              cleanup_nums = {
                  "area_code":{ "408":1259,"415":2495,"510":1246},
                  "international_plan":{ "0":4527,"1":473},
                  "voice_mail_plan":{ "0":3677,"1":1323},
                  "number_customer_service_calls":{ "0":1023,"1":1786,"2":1127,"3":665,"4":252,"5":96,"6":34,"7":13,"8":2,"9":2}
                      }
              obj_df = X.select_dtypes(include=['object']).copy()
              obj_df[obj_df.isnull().any(axis=1)]
              obj_df.head()
              X.replace(cleanup_nums, inplace=True)
              X.head()

          if datasetName == "Australian":
              flagThereArePreprocess = True
              cleanup_nums = {
                  "A1":{ "0":222,"1":468},
                  "A4":{ "1":163,"2":525,"3":2},
                  "A5":{ "1":53,"2":30,"3":59,"4":51,"5":10,"6":54,"7":38,"8":146,"9":64,"10":25,"11":78,"12":3,"13":41,"14":38},
                  "A6":{ "1":57,"2":6,"3":8,"4":408,"5":59,"7":6,"8":138,"9":8},
                  "A8":{ "0":329,"1":361},
                  "A9":{ "0":395,"1":295},
                  "A11":{ "0":374,"1":316},
                  "A12":{ "1":57,"2":625,"3":8}
                      }
              obj_df = X.select_dtypes(include=['object']).copy()
              obj_df[obj_df.isnull().any(axis=1)]
              obj_df.head()
              X.replace(cleanup_nums, inplace=True)
              X.head()

          if datasetName == "autoUniv-au1-1000":
                flagThereArePreprocess = True
                print("All numeric")

          if datasetName == "haberman":
                flagThereArePreprocess = True
                cleanup_nums = {
                    "Patients_year_of_operation":{ "58":36,"59":27,"60":28,"61":26,"62":23,"63":30,"64":31,"65":28,"66":28,"67":25,"68":13,"69":11}
                        }
                obj_df = X.select_dtypes(include=['object']).copy()
                obj_df[obj_df.isnull().any(axis=1)]
                obj_df.head()
                X.replace(cleanup_nums, inplace=True)
                X.head()

          if datasetName == "heart-statlog":
                flagThereArePreprocess = True
                print("All numeric")

          if datasetName == "ionosphere":
                flagThereArePreprocess = True
                print("All numeric")

          if datasetName == "sonar":
                flagThereArePreprocess = True
                print("All numeric")

          if datasetName == "Satellite":
                flagThereArePreprocess = True
                print("All numeric")

          if datasetName == "SPECT":
                flagThereArePreprocess = True
                cleanup_nums = {
                    "F1":{ "0":148,"1":119},
                    "F2":{ "0":201,"1":66},
                    "F3":{ "0":162,"1":105},
                    "F4":{ "0":191,"1":76},
                    "F5":{ "0":159,"1":108},
                    "F6":{ "0":204,"1":63},
                    "F7":{ "0":191,"1":76},
                    "F8":{ "0":153,"1":114},
                    "F9":{ "0":184,"1":83},
                    "F10":{ "0":166,"1":101},
                    "F11":{ "0":202,"1":65},
                    "F12":{ "0":188,"1":79},
                    "F13":{ "0":135,"1":132},
                    "F14":{ "0":186,"1":81},
                    "F15":{ "0":220,"1":47},
                    "F16":{ "0":184,"1":83},
                    "F17":{ "0":229,"1":38},
                    "F18":{ "0":232,"1":35},
                    "F19":{ "0":201,"1":66},
                    "F20":{ "0":181,"1":86},
                    "F21":{ "0":170,"1":97},
                    "F22":{ "0":157,"1":110}
                        }
                obj_df = X.select_dtypes(include=['object']).copy()
                obj_df[obj_df.isnull().any(axis=1)]
                obj_df.head()
                X.replace(cleanup_nums, inplace=True)
                X.head()

          if datasetName == "analcatdata_boxing1":
                flagThereArePreprocess = True
                cleanup_nums = {
                      "Judge":{ "Associated_Press":12,"Boxing_Monthly-Leach":12,"Boxing_Times":12,"E._Williams":12,"ESPN":12,"HBO-Lederman":12,"L._OConnell":12,"S._Christodoulu":12,"Sportsline":12,"Sportsticker":12},
                      "Official":{ "No":84,"Yes":36},
                      "Round":{ "1":10,"2":10,"3":10,"4":10,"5":10,"6":10,"7":10,"8":10,"9":10,"10":10,"11":10,"12":10}
                        }
                obj_df = X.select_dtypes(include=['object']).copy()
                obj_df[obj_df.isnull().any(axis=1)]
                obj_df.head()
                X.replace(cleanup_nums, inplace=True)
                X.head()

          if datasetName == "aids":
                flagThereArePreprocess = True
                cleanup_nums = {
                      "Age":{ "15-24":10,"25-34":10,"35-44":10,"45-54":10,"55_&_older":10},
                      "Race":{ "American_Indian_&_Alaska_Native":10,"Asian_&_Pacific_Islander":10,"Black_not_Hispanic":10,"Hispanic":10,"White_not_Hispanic":10},
                      }
                obj_df = X.select_dtypes(include=['object']).copy()
                obj_df[obj_df.isnull().any(axis=1)]
                obj_df.head()
                X.replace(cleanup_nums, inplace=True)
                X.head()

          if datasetName == "servo":
              flagThereArePreprocess = True
              cleanup_nums = {
                    "motor":{ "A":36,"B":36,"C":40,"D":22,"E":33},
                    "screw":{ "A":42,"B":35,"C":31,"D":30,"E":29},
                    "pgain":{ "3":50,"4":66,"5":26,"6":25},
                    "vgain":{ "1":47,"2":49,"3":27,"4":22,"5":22}
                      }
              obj_df = X.select_dtypes(include=['object']).copy()
              obj_df[obj_df.isnull().any(axis=1)]
              obj_df.head()
              X.replace(cleanup_nums, inplace=True)
              X.head()

          if datasetName == "analcatdata_creditscore":
              flagThereArePreprocess = True
              cleanup_nums = {
                    "Own.home":{ "0":64,"1":36},
                    "Self.employed":{ "0":95,"1":5},
                    "Derogatory.reports":{ "0":82,"1":10,"2":3,"3":3,"4":1,"7":1}
                      }
              obj_df = X.select_dtypes(include=['object']).copy()
              obj_df[obj_df.isnull().any(axis=1)]
              obj_df.head()
              X.replace(cleanup_nums, inplace=True)
              X.head()

          if datasetName == "analcatdata_boxing2":
              flagThereArePreprocess = True
              cleanup_nums = {
                    "Judge":{ "Associated_Press":12,"B._Logist":12,"G._Hamada":12,"HBO-Lederman":12,"J._Roth":12,"Las_Vegas_Review-Journal":12,"Los_Angeles_Times-Kawakami":12,"Los_Angeles_Times-Springer":12,"Sportsticker":12,"USA_Today":12,"van_de_Wiele":12},
                    "Official":{ "No":96,"Yes":36},
                    "Round":{ "1":11,"2":11,"3":11,"4":11,"5":11,"6":11,"7":11,"8":11,"9":11,"10":11,"11":11,"12":11}
                      }
              obj_df = X.select_dtypes(include=['object']).copy()
              obj_df[obj_df.isnull().any(axis=1)]
              obj_df.head()
              X.replace(cleanup_nums, inplace=True)
              X.head()

          if datasetName == "datatrieve":
              flagThereArePreprocess = True
              print("All numeric")

          if datasetName == "analcatdata_lawsuit":
              flagThereArePreprocess = True
              cleanup_nums = {
                    "Minority":{ "0":167,"1":97}
                      }
              obj_df = X.select_dtypes(include=['object']).copy()
              obj_df[obj_df.isnull().any(axis=1)]
              obj_df.head()
              X.replace(cleanup_nums, inplace=True)
              X.head()

          if datasetName == "pc2":
              flagThereArePreprocess = True
              print("All numeric")

          if datasetName == "delta_ailerons":
              flagThereArePreprocess = True
              print("All numeric")

          if datasetName == "mc1":
              flagThereArePreprocess = True
              print("All numeric")

          if datasetName == "ar4":
              flagThereArePreprocess = True
              print("All numeric")

          if datasetName == "ar6":
              flagThereArePreprocess = True
              print("All numeric")

          if datasetName == "kc3":
              flagThereArePreprocess = True
              print("All numeric")

          if datasetName == "mc2":
              flagThereArePreprocess = True
              print("All numeric")

          if datasetName == "mw1":
              flagThereArePreprocess = True
              print("All numeric")

          if datasetName == "jEdit_4.0_4.2":
              flagThereArePreprocess = True
              print("All numeric")

          if datasetName == "prnn_crabs":
              flagThereArePreprocess = True
              cleanup_nums = {
                      "sex":{ "Female":1,"Male":0}
                    }
              obj_df = X.select_dtypes(include=['object']).copy()
              obj_df[obj_df.isnull().any(axis=1)]
              obj_df.head()
              X.replace(cleanup_nums, inplace=True)
              X.head()

          if datasetName == 'analcatdata_lawsuit':
              flagThereArePreprocess = True
              print("All numeric")

          if datasetName == 'hill-valley':
              flagThereArePreprocess = True
              print("All numeric")


          if flagThereArePreprocess == False:
            print("@@@@ NOTE>: There arent preprocess in X dataset ", datasetName," @@@@")


          X[X.select_dtypes(['category']).columns]= X[X.select_dtypes(['category']).columns].apply(lambda x: pd.factorize(x)[0])

          return X

  def PreprocessYByDataset(Y,datasetName):
          flagThereArePreprocess = False
          if datasetName == "hill-valley":
              flagThereArePreprocess = True
              Y = Y.astype(int)

          if datasetName == "analcatdata_lawsuit":
              flagThereArePreprocess = True
              Y = Y.astype(int)

          if datasetName == "credit-g":
              flagThereArePreprocess = True
              Y = Y.map({"good":1, "bad":0})

          if datasetName == "blood-transfusion-service-center":
              flagThereArePreprocess = True
              Y = Y.astype(int)

          if datasetName == "monks-problems-2":
              flagThereArePreprocess = True
              Y = Y.astype(int);

          if datasetName == "tic-tac-toe":
              flagThereArePreprocess = True
              Y = Y.map({"positive":1, "negative":0})

          if datasetName == "monks-problems-1":
              flagThereArePreprocess = True
              Y = Y.astype(int);

          if datasetName == "steel-plates-fault":
              flagThereArePreprocess = True
              Y = Y.astype(int);

          if datasetName == "kr-vs-kp":
              flagThereArePreprocess = True
              Y = Y.map({"won":1, "nowin":0})

          if datasetName == "qsar-biodeg":
              flagThereArePreprocess = True
              Y = Y.astype(int);

          if datasetName == "wdbc":
              flagThereArePreprocess = True
              Y = Y.astype(int);

          if datasetName == "phoneme":
              flagThereArePreprocess = True
              Y = Y.astype(int);

          if datasetName == "diabetes":
              flagThereArePreprocess = True
              Y = Y.map({"tested_positive":1, "tested_negative":0})

          if datasetName == "ozone-level-8hr":
              flagThereArePreprocess = True
              Y = Y.astype(int);

          if datasetName == "kc2":
              flagThereArePreprocess = True
              Y = Y.map({"yes":1, "no":0})

          if datasetName == "eeg-eye-state":
              flagThereArePreprocess = True
              Y = Y.astype(int);

          if datasetName == "climate-model-simulation-crashes":
              flagThereArePreprocess = True
              Y = Y.astype(int);

          if datasetName == "spambase":
              flagThereArePreprocess = True
              Y = Y.astype(int);

          if datasetName == "kc1":
              flagThereArePreprocess = True
              Y = Y.astype(int);

          if datasetName == "ilpd":
              flagThereArePreprocess = True
              Y = Y.astype(int);

          if datasetName == "pc1":
              flagThereArePreprocess = True
              Y = Y.astype(int);

          if datasetName == "pc3":
              flagThereArePreprocess = True
              Y = Y.astype(int);

          if datasetName == "banknote-authentication":
              flagThereArePreprocess = True
              Y = Y.astype(int);

          if datasetName == "pc4":
              flagThereArePreprocess = True
              Y = Y.astype(int);

          if datasetName == "mozilla4":
              flagThereArePreprocess = True
              Y = Y.astype(int);

          if datasetName == "monks-problems-3":
              flagThereArePreprocess = True
              Y = Y.astype(int);

          if datasetName == "PhishingWebsites":
              flagThereArePreprocess = True
              Y = Y.map({"1":1, "-1":0})

          if datasetName == "churn":
              flagThereArePreprocess = True
              Y = Y.astype(int);

          if datasetName == "Australian":
              flagThereArePreprocess = True
              Y = Y.astype(int);

          if datasetName == "autoUniv-au1-1000":
              flagThereArePreprocess = True
              Y = Y.map({"class1":1, "class2":0})

          if datasetName == "haberman":
              flagThereArePreprocess = True
              Y = Y.astype(int);

          if datasetName == "heart-statlog":
              flagThereArePreprocess = True
              Y = Y.map({"absent":1, "present":0})

          if datasetName == "ionosphere":
              flagThereArePreprocess = True
              Y = Y.map({"b":1, "g":0});

          if datasetName == "sonar":
              flagThereArePreprocess = True
              Y = Y.map({"Rock":1, "Mine":0});

          if datasetName == "Satellite":
              flagThereArePreprocess = True
              Y = Y.map({"Normal":1, "Anomaly":0});

          if datasetName == "SPECT":
              flagThereArePreprocess = True
              Y = Y.astype(int);

          if datasetName == "analcatdata_boxing1":
              flagThereArePreprocess = True
              Y = Y.map({"Holyfield":1, "Lewis":0});

          if datasetName == "aids":
              flagThereArePreprocess = True
              Y = Y.map({"Female":1, "Male":0});

          if datasetName == "servo":
              flagThereArePreprocess = True
              Y = Y.map({"P":1, "N":0});

          if datasetName == "analcatdata_creditscore":
              flagThereArePreprocess = True
              Y = Y.astype(int);

          if datasetName == "analcatdata_boxing2":
              flagThereArePreprocess = True
              Y = Y.map({"de_la_Hoya":1, "Trinidad":0});

          if datasetName == "datatrieve":
              flagThereArePreprocess = True
              Y = Y.astype(int);

          if datasetName == "analcatdata_lawsuit":
              flagThereArePreprocess = True
              Y = Y.astype(int);

          if datasetName == "pc2":
              flagThereArePreprocess = True
              Y = Y.map({False:1, True:0});

          if datasetName == "delta_ailerons":
              flagThereArePreprocess = True
              Y = Y.map({"P":1, "N":0});

          if datasetName == "mc1":
              flagThereArePreprocess = True
              Y = Y.map({False:1, True:0});

          if datasetName == "ar4":
              flagThereArePreprocess = True
              Y = Y.map({False:1, True:0});

          if datasetName == "ar6":
              flagThereArePreprocess = True
              Y = Y.map({False:1, True:0});

          if datasetName == "kc3":
              flagThereArePreprocess = True
              Y = Y.map({False:1, True:0});

          if datasetName == "mc2":
              flagThereArePreprocess = True
              Y = Y.map({ False: 1, True: 0});

          if datasetName == "mw1":
              flagThereArePreprocess = True
              Y = Y.map({False:1, True:0});

          if datasetName == "jEdit_4.0_4.2":
              flagThereArePreprocess = True
              Y = Y.map({True:1, False:0});

          if datasetName == "prnn_crabs":
              flagThereArePreprocess = True
              Y = Y.map({"blue_form":1, "orange_form":0});

          if flagThereArePreprocess == False:
            print("@@@@ NOTE>: There arent preprocess in Y dataset ", datasetName," @@@@")

          return Y

print("===========================================================================")
print("============================== ANALYSIS ===================================")
print("===========================================================================")
print("")
class AnalysisDefault():

  do_download_files = do_download

  def calcAccuracyPrecisionRecallByModel(y_test,y_pred):
      ac = accuracy_score(y_test, y_pred)
      pr = precision_score(y_test, y_pred)
      re = recall_score(y_test, y_pred)
      au = roc_auc_score(y_test, y_pred)
      return round(ac,2), round(pr,2), round(re,2), round(au,2)


  def calcConfusionMatrix(y_test, y_pred, name_model):
      m = confusion_matrix(y_test, y_pred,normalize='true')

      print("## confusion matrix ",name_model," ##")
      print("  Correct True: ",m[0][0]," Uncorrect True: ",m[0][1])
      print("Uncorrect False: ",m[1][0],"   Correct False: ",m[1][1])
      return round(m[0][0],2), round(m[1][1],2)



  def calcCrossValidation(model, X_train, y_train, cv, plotGraph):
      hit_rates = cross_val_score(model, X_train, y_train, cv=cv)
      if plotGraph == True:
          sns.kdeplot(data=hit_rates,legend=True,bw_method=0.7)

      return hit_rates

  def plotComparationCrossValidation(hr_m1,hr_m2,hr_m3,hr_m4, file_name):
      dt_tmp = {
            'm1':hr_m1,
            'm2':hr_m2,
            'm3':hr_m3,
            'm4':hr_m4
            }
      dt_tmp = pd.DataFrame(dt_tmp)

      fig, ax = plt.subplots(figsize=(8,6))
      sns.kdeplot(data=dt_tmp,legend=True,bw_method=0.7)
      plt.savefig(file_name,bbox_inches='tight')
      plt.show()

  def plotFriedmanTest(hr_rf,hr_gb,hr_dt,hr_ad, file_name):
      a = [hr_rf.tolist(),hr_gb.tolist(), hr_dt.tolist(),hr_ad.tolist()]

      #print('Friedman Test')
      #print(friedmanchisquare(*a))
      posthoc = sp.posthoc_nemenyi_friedman(list(map(list, zip(*a))))
      models_names = ['m1','m2', 'm3','m4']
      sns.heatmap(posthoc, vmin=0, vmax=1, xticklabels=models_names,yticklabels=models_names,cmap="Greens",linewidths=.5,annot=True)
      plt.savefig(file_name,bbox_inches='tight')
      plt.show()
      
  def calcFriedmanTestByDf(df,mlist):
      
      a = []
      
      for i in mlist:
          a.append(df[i].to_list())
      
      print('Friedman Test')
      print(friedmanchisquare(*a))
      posthoc = sp.posthoc_nemenyi_friedman(list(map(list, zip(*a))))

      return posthoc
    
  def calcFriedmanTest(hr_m1,hr_m2,hr_m3,hr_m4, models_names):
      a = [hr_m1.tolist(),hr_m2.tolist(), hr_m3.tolist(),hr_m4.tolist()]
      
      #print('Friedman Test')
      #print(friedmanchisquare(*a))
      posthoc = sp.posthoc_nemenyi_friedman(list(map(list, zip(*a))))


      #the sequency before is sensivity
      ret_for_df = [
                    #posthoc.iloc[0,0],
                    posthoc.iloc[0,1],
                    posthoc.iloc[0,2],
                    posthoc.iloc[0,3],
                    #posthoc.iloc[1,1],
                    posthoc.iloc[1,2],
                    posthoc.iloc[1,3],
                    #posthoc.iloc[2,2],
                    posthoc.iloc[2,3],
                    #posthoc.iloc[3,3]
                    ]

      return posthoc, ret_for_df

  

  def calcSpearmanCoef(d1,d2):
    coef, p = spearmanr(d1, d2)
    return coef, p

print("===========================================================================")
print("============================== MCA MODULES ================================")
print("===========================================================================")
print("")


GRAY = OrderedDict([
    ('light', '#bababa'),
    ('dark', '#404040')
])


def stylize_axis(ax, grid=True):

    if grid:
        ax.grid()

    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    ax.axhline(y=0, linestyle='-', linewidth=1.2, color=GRAY['dark'], alpha=0.6)
    ax.axvline(x=0, linestyle='-', linewidth=1.2, color=GRAY['dark'], alpha=0.6)

    return ax


def build_ellipse(X, Y):
    """Construct ellipse coordinates from two arrays of numbers.
    Args:
        X (1D array_like)
        Y (1D array_like)
    Returns:
        float: The mean of `X`.
        float: The mean of `Y`.
        float: The width of the ellipse.
        float: The height of the ellipse.
        float: The angle of orientation of the ellipse.
    """
    x_mean = np.mean(X)
    y_mean = np.mean(Y)

    cov_matrix = np.cov(np.vstack((X, Y)))
    U, s, V = linalg.svd(cov_matrix, full_matrices=False)

    chi_95 = np.sqrt(4.61)  # 90% quantile of the chi-square distribution
    width = np.sqrt(cov_matrix[0][0]) * chi_95 * 2
    height = np.sqrt(cov_matrix[1][1]) * chi_95 * 2

    eigenvector = V.T[0]
    angle = np.arctan(eigenvector[1] / eigenvector[0])

    return x_mean, y_mean, width, height, angle

def make_labels_and_names(X):

    if isinstance(X, pd.DataFrame):
        row_label = X.index.name if X.index.name else 'Rows'
        row_names = X.index.tolist()
        col_label = X.columns.name if X.columns.name else 'Columns'
        col_names = X.columns.tolist()
    else:
        row_label = 'Rows'
        row_names = list(range(X.shape[0]))
        col_label = 'Columns'
        col_names = list(range(X.shape[1]))

    return row_label, row_names, col_label, col_names

"""Singular Value Decomposition (SVD)"""
try:
    import fbpca
    FBPCA_INSTALLED = True
except ImportError:
    FBPCA_INSTALLED = False
from sklearn.utils import extmath


def compute_svd(X, n_components, n_iter, random_state, engine):
    """Computes an SVD with k components."""

    # Determine what SVD engine to use
    if engine == 'auto':
        engine = 'sklearn'

    # Compute the SVD
    if engine == 'fbpca':
        if FBPCA_INSTALLED:
            U, s, V = fbpca.pca(X, k=n_components, n_iter=n_iter)
        else:
            raise ValueError('fbpca is not installed; please install it if you want to use it')
    elif engine == 'sklearn':
        U, s, V = extmath.randomized_svd(
            X,
            n_components=n_components,
            n_iter=n_iter,
            random_state=random_state
        )
    else:
        raise ValueError("engine has to be one of ('auto', 'fbpca', 'sklearn')")

    U, V = extmath.svd_flip(U, V)

    return U, s, V


"""Correspondence Analysis (CA)"""
from scipy import sparse
from sklearn import base
from sklearn import utils

# from . import plot
# from . import util
# from . import svd


class CA(base.BaseEstimator, base.TransformerMixin):

    def __init__(self, n_components=2, n_iter=10, copy=True, check_input=True, benzecri=False,
                 random_state=None, engine='auto'):
        self.n_components = n_components
        self.n_iter = n_iter
        self.copy = copy
        self.check_input = check_input
        self.random_state = random_state
        self.benzecri = benzecri
        self.engine = engine

    def fit(self, X, y=None):

        # Check input
        if self.check_input:
            utils.check_array(X)

        # Check all values are positive
        if (X < 0).any().any():
            raise ValueError("All values in X should be positive")

        _, row_names, _, col_names = make_labels_and_names(X)

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        if self.copy:
            X = np.copy(X)

        # Compute the correspondence matrix which contains the relative frequencies
        X = X.astype(float) / np.sum(X)

        # Compute row and column masses
        self.row_masses_ = pd.Series(X.sum(axis=1), index=row_names)
        self.col_masses_ = pd.Series(X.sum(axis=0), index=col_names)

        # Compute standardised residuals
        r = self.row_masses_.to_numpy()
        c = self.col_masses_.to_numpy()
        S = sparse.diags(r ** -.5) @ (X - np.outer(r, c)) @ sparse.diags(c ** -.5)

        # Compute SVD on the standardised residuals
        self.U_, self.s_, self.V_ = compute_svd(
            X=S,
            n_components=self.n_components,
            n_iter=self.n_iter,
            random_state=self.random_state,
            engine=self.engine
        )

        # Compute total inertia
        self.total_inertia_ = np.einsum('ij,ji->', S, S.T)

        return self

    def _check_is_fitted(self):
        utils.validation.check_is_fitted(self, 'total_inertia_')

    def transform(self, X):
        """Computes the row principal coordinates of a dataset.
        Same as calling `row_coordinates`. In most cases you should be using the same
        dataset as you did when calling the `fit` method. You might however also want to included
        supplementary data.
        """
        self._check_is_fitted()
        if self.check_input:
            utils.check_array(X)
        return self.row_coordinates(X)

    @property
    def eigenvalues_(self):
        """The eigenvalues associated with each principal component.
        Benzecri correction is applied if specified.
        """
        self._check_is_fitted()

        K = len(self.col_masses_)

        if self.benzecri:
            return np.array([
                (K / (K - 1.) * (s - 1. / K)) ** 2
                if s > 1. / K else 0
                for s in np.square(self.s_)
            ])

        return np.square(self.s_).tolist()

    @property
    def explained_inertia_(self):
        """The percentage of explained inertia per principal component."""
        self._check_is_fitted()
        return [eig / self.total_inertia_ for eig in self.eigenvalues_]

    def row_coordinates(self, X):
        """The row principal coordinates."""
        self._check_is_fitted()

        _, row_names, _, _ = make_labels_and_names(X)

        if isinstance(X, pd.DataFrame):
            try:
                X = X.sparse.to_coo().astype(float)
            except AttributeError:
                X = X.to_numpy()

        if self.copy:
            X = X.copy()

        # Normalise the rows so that they sum up to 1
        if isinstance(X, np.ndarray):
            X = X / X.sum(axis=1)[:, None]
        else:
            X = X / X.sum(axis=1)

        return pd.DataFrame(
            data=X @ sparse.diags(self.col_masses_.to_numpy() ** -0.5) @ self.V_.T,
            index=row_names
        )

    def column_coordinates(self, X):
        """The column principal coordinates."""
        self._check_is_fitted()

        _, _, _, col_names = make_labels_and_names(X)

        if isinstance(X, pd.DataFrame):
            is_sparse = X.dtypes.apply(pd.api.types.is_sparse).all()
            if is_sparse:
                X = X.sparse.to_coo()
            else:
                X = X.to_numpy()

        if self.copy:
            X = X.copy()

        # Transpose and make sure the rows sum up to 1
        if isinstance(X, np.ndarray):
            X = X.T / X.T.sum(axis=1)[:, None]
        else:
            X = X.T / X.T.sum(axis=1)

        return pd.DataFrame(
            data=X @ sparse.diags(self.row_masses_.to_numpy() ** -0.5) @ self.U_,
            index=col_names
        )

    def plot_coordinates(self, X, ax=None, figsize=(6, 6), x_component=0, y_component=1,
                                   show_row_labels=True, show_col_labels=True, **kwargs):
        """Plot the principal coordinates."""

        self._check_is_fitted()

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        # Add style
        ax = stylize_axis(ax)

        # Get labels and names
        row_label, row_names, col_label, col_names = make_labels_and_names(X)

        # Plot row principal coordinates
        row_coords = self.row_coordinates(X)
        ax.scatter(
            row_coords[x_component],
            row_coords[y_component],
            **kwargs,
            label=row_label
        )

        # Plot column principal coordinates
        col_coords = self.column_coordinates(X)
        ax.scatter(
            col_coords[x_component],
            col_coords[y_component],
            **kwargs,
            label=col_label
        )

        # Add row labels
        if show_row_labels:
            x = row_coords[x_component]
            y = row_coords[y_component]
            for xi, yi, label in zip(x, y, row_names):
                ax.annotate(label, (xi, yi))

        # Add column labels
        if show_col_labels:
            x = col_coords[x_component]
            y = col_coords[y_component]
            for xi, yi, label in zip(x, y, col_names):
                ax.annotate(label, (xi, yi))

        # Legend
        ax.legend()

        # Text
        ax.set_title('Principal coordinates')
        ei = self.explained_inertia_
        ax.set_xlabel('Component {} ({:.2f}% inertia)'.format(x_component, 100 * ei[x_component]),fontsize=14)
        ax.set_ylabel('Component {} ({:.2f}% inertia)'.format(y_component, 100 * ei[y_component]),fontsize=14)

        return ax



# from . import ca
# from . import plot


class MCA(CA):

    def fit(self, X, y=None):

        if self.check_input:
            utils.check_array(X, dtype=[str, np.number])

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        n_initial_columns = X.shape[1]

        # One-hot encode the data
        one_hot = pd.get_dummies(X)

        # Apply CA to the indicator matrix
        super().fit(one_hot)

        # Compute the total inertia
        n_new_columns = one_hot.shape[1]
        self.total_inertia_ = (n_new_columns - n_initial_columns) / n_initial_columns

        return self

    def row_coordinates(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return super().row_coordinates(pd.get_dummies(X))

    def column_coordinates(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return super().column_coordinates(pd.get_dummies(X))

    def transform(self, X):
        """Computes the row principal coordinates of a dataset."""
        self._check_is_fitted()
        if self.check_input:
            utils.check_array(X, dtype=[str, np.number])
        return self.row_coordinates(X)

    def plot_coordinates(self, X, ax=None, figsize=(6, 6), x_component=0, y_component=1,
                         show_row_points=True, row_points_size=10,
                         row_points_alpha=1, show_row_labels=False,
                         show_column_points=True, column_points_size=30, show_column_labels=False,
                         legend_n_cols=1,
                         xlim=(-2,2),
                         ylim=(-2,2),c=[],colors = ['#5496fa',
                                                    '#aec7e8',
                                                    '#ff7f0e',
                                                    '#ffbb78',
                                                    '#539c8e',
                                                    '#98df8a',
                                                    '#FF6347',
                                                    '#ff9896',
                                                    '#9467bd',
                                                    '#c5b0d5',
                                                    '#8c564b',
                                                    '#c49c94',
                                                    '#e377c2',
                                                    '#f7b6d2',
                                                    '#7f7f7f',
                                                    '#c7c7c7',
                                                    '#bcbd22',
                                                    '#dbdb8d',
                                                    '#17becf',
                                                    '#9edae5'],
                         df_dataset_properties=None,
                         k_cluster_selected=''):
        """Plot row and column principal coordinates.
        Parameters:
            ax (matplotlib.Axis): A fresh one will be created and returned if not provided.
            figsize ((float, float)): The desired figure size if `ax` is not provided.
            x_component (int): Number of the component used for the x-axis.
            y_component (int): Number of the component used for the y-axis.
            show_row_points (bool): Whether to show row principal components or not.
            row_points_size (float): Row principal components point size.
            row_points_alpha (float): Alpha for the row principal component.
            show_row_labels (bool): Whether to show row labels or not.
            show_column_points (bool): Whether to show column principal components or not.
            column_points_size (float): Column principal components point size.
            show_column_labels (bool): Whether to show column labels or not.
            legend_n_cols (int): Number of columns used for the legend.
        Returns:
            matplotlib.Axis
        """

        color_cluster = ['blue', 'red', 'orange', 'green']

        def jitter_dots(dots):
          offsets = dots.get_offsets()
          jittered_offsets = offsets
          # only jitter in the x-direction
          jittered_offsets[:, 0] += np.random.uniform(-0.1, 0.1, offsets.shape[0])
          # only jitter in the y-direction
          jittered_offsets[:, 1] += np.random.uniform(-0.1, 0.1, offsets.shape[0])


        self._check_is_fitted()

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        # Add style
        ax = stylize_axis(ax)

        # Plot row principal coordinates
        if show_row_points or show_row_labels:

            row_coords = self.row_coordinates(X)

            if show_row_points:
                id_c0 = df_dataset_properties[df_dataset_properties.loc[:][k_cluster_selected] == 0].index
                id_c1 = df_dataset_properties[df_dataset_properties.loc[:][k_cluster_selected] == 1].index
                id_c2 = df_dataset_properties[df_dataset_properties.loc[:][k_cluster_selected] == 2].index
                id_c3 = df_dataset_properties[df_dataset_properties.loc[:][k_cluster_selected] == 3].index
                if c[0]==True:
                  dots = ax.scatter(
                      row_coords.loc[id_c0, x_component],
                      row_coords.loc[id_c0, y_component],
                      s=row_points_size,
                      color=color_cluster[0],
                      alpha=row_points_alpha,
                      marker=r'^',
                      label='Dataset from cluster 0'
                  )
                  if show_row_labels == False:
                    jitter_dots(dots)

                if c[1]==True:
                  dots = ax.scatter(
                      row_coords.loc[id_c1, x_component],
                      row_coords.loc[id_c1, y_component],
                      s=row_points_size,
                      color=color_cluster[1],
                      alpha=row_points_alpha,
                      marker=r'^',
                      label='Dataset from cluster 1'
                  )
                  if show_row_labels == False:
                    jitter_dots(dots)

                if c[2]==True:
                  dots = ax.scatter(
                      row_coords.loc[id_c2, x_component],
                      row_coords.loc[id_c2, y_component],
                      s=row_points_size,
                      color=color_cluster[2],
                      alpha=row_points_alpha,
                      marker=r'^',
                      label='Dataset from cluster 2'
                  )
                  if show_row_labels == False:
                    jitter_dots(dots)

                if c[3]==True:
                  dots = ax.scatter(
                      row_coords.loc[id_c3, x_component],
                      row_coords.loc[id_c3, y_component],
                      s=row_points_size,
                      color=color_cluster[3],
                      alpha=row_points_alpha,
                      marker=r'^',
                      label='Dataset from cluster 3'
                  )
                  if show_row_labels == False:
                    jitter_dots(dots)
            texts = []

            if show_row_labels:
                for _, row in row_coords.iterrows():
                    if (df_dataset_properties.loc[row.name][k_cluster_selected] == 0) and c[0]==True:
                      texts.append(ax.text(row[x_component], row[y_component], s=row.name, **dict(color=color_cluster[0], alpha=0.6,fontsize='11')))
                    else:
                      if (df_dataset_properties.loc[row.name][k_cluster_selected] == 1) and c[1]==True:
                        texts.append(ax.text(row[x_component], row[y_component], s=row.name, **dict(color=color_cluster[1], alpha=0.6,fontsize='11')))
                      else:
                        if (df_dataset_properties.loc[row.name][k_cluster_selected] == 2) and c[2]==True:
                          texts.append(ax.text(row[x_component], row[y_component], s=row.name, **dict(color=color_cluster[2],fontsize='11')))
                        else:
                          if (df_dataset_properties.loc[row.name][k_cluster_selected] == 3) and c[3]==True:
                            texts.append(ax.text(row[x_component], row[y_component], s=row.name, **dict(color=color_cluster[3],fontsize='11')))
            #adjust_text(texts)


        # Plot column principal coordinates
        if show_column_points or show_column_labels:

            col_coords = self.column_coordinates(X)
            x = col_coords[x_component]
            y = col_coords[y_component]

            prefixes = col_coords.index.str.split('_').map(lambda x: x[0])



            index = 0
            for prefix in prefixes.unique():
                mask = prefixes == prefix

                if show_column_points:
                    ax.scatter(x[mask], y[mask], s=column_points_size, label=prefix,color=colors[index])

                if show_column_labels:
                    for i, label in enumerate(col_coords[mask].index):
                       texts.append(ax.text(x[mask][i], y[mask][i],s=label,**dict(fontsize='12')))
                index += 1


            adjust_text(texts,arrowprops=dict(arrowstyle='->', color='silver'),expand_text=(1,1.4))
            ax.legend(ncol=legend_n_cols,loc='upper left',fontsize='12')


        # Text
        #ax.set_title('Row and column principal coordinates')
        ei = self.explained_inertia_
        ax.set_xlabel('Component {} ({:.2f}%)'.format(x_component, 100 * ei[x_component]),fontsize=20)
        ax.set_ylabel('Component {} ({:.2f}%)'.format(y_component, 100 * ei[y_component]),fontsize=20)
        plt.xlim(xlim)
        plt.ylim(ylim)
        return ax


print("===========================================================================")
print("=========================== MAIN PIPELINE MODULE ==========================")
print("===========================================================================")
print("")

class Util():
  def is_int(n):
    for i in range(len(n)):
      if isinstance(n[i],float):
        return False
    return True

  def identifyCatFeatures(X):
    categorical_features = []
    for i, c in enumerate(X.columns.to_list()):
      if X[c].dtype == "category":
        categorical_features.append(c)
    return categorical_features

  def powerSetLimited(s,l):

    def powerset(s):
      x = len(s)
      masks = [1 << i for i in range(x)]
      for i in range(1 << x):
          yield [ss for mask, ss in zip(masks, s) if i & mask]

    psl = []
    ps = list(powerset(s))
    for _,i in enumerate(ps):
      if len(i) <= l and len(i)>0:
        psl.append(i)
    return psl

  def auc_score(y, y_pred):

    fpr, tpr, thresholds = metrics.roc_curve(y, y_pred, pos_label=2)

    return metrics.auc(fpr, tpr)


def main_pipeline (code_datasets, df_dataset_properties_cluster,cluster_name,apply_smote=False):

  global path_dataset

  global X

  

  models_name = ['m1','m2','m3','m4']

  #This dataset will be complete (with values) only in final of notebook
  dt = {  'm1':[],
          'm2':[],
          'm3':[],
          'm4':[]
        }
  df_models_accuracy_analisys = pd.DataFrame(dt)

  dt = {  'm1':[],
          'm2':[],
          'm3':[],
          'm4':[]
        }
  df_models_precision_analisys = pd.DataFrame(dt)

  dt = {  'm1':[],
          'm2':[],
          'm3':[],
          'm4':[]
        }
  df_models_recall_analisys = pd.DataFrame(dt)

  dt = {  'dataset_name'
          'm1_n_nodes':[],
          'm2_n_nodes':[],
          'm3_n_nodes':[],
          'm4_n_nodes':[]
        }
  df_models_nodes = pd.DataFrame(dt)

  dt = {  #'M1xM1':[],
          'm1xm2':[],
          'm1xm3':[],
          'm1xm4':[],
          #'M2xM2':[],
          'm2xm3':[],
          'm2xm4':[],
          #'M3xM3':[],
          'm3xm4':[]
          #'M4xM4':[]
        }
  df_models_friedman_analisys = pd.DataFrame(dt)

  dt = {'dataset_name':[],
          'model_name':[],
          'model_params':[],
          'accuracy':[],
          'precision':[],
          'recall':[],
          'correct_true':[],
          'correct_false':[]}
  df_models_info = pd.DataFrame(dt)

  dt = {'dataset_name':[],
          'model_name':[],
          'xai_vs_xai':[],
          'correlation':[]}
  df_resume_boxplot = pd.DataFrame(dt)


  dt = {  'model':[],
          'shap_vs_shap':[],
          'shap_vs_eli5':[],
          'shap_vs_dalex':[],
          'shap_vs_ci':[],
          'shap_vs_skater':[],
          'shap_vs_lofo':[],
          'shap_vs_exirt':[],
          'eli5_vs_eli5':[],
          'eli5_vs_dalex':[],
          'eli5_vs_ci':[],
          'eli5_vs_skater':[],
          'eli5_vs_lofo':[],
          'eli5_vs_exirt':[],
          'dalex_vs_dalex':[],
          'dalex_vs_ci':[],
          'dalex_vs_skater':[],
          'dalex_vs_lofo':[],
          'dalex_vs_exirt':[],
          'ci_vs_ci':[],
          'ci_vs_skater':[],
          'ci_vs_lofo':[],
          'ci_vs_exirt':[],
          'skater_vs_skater':[],
          'skater_vs_lofo':[],
          'lofo_vs_lofo':[],
          'lofo_vs_exirt':[],
          'exirt_vs_exirt':[]}
  df_resume_final_boxplot = pd.DataFrame(dt)
  df_resume_final_boxplot_p = pd.DataFrame(dt)







  #if run_train_test_model_* is False the upload of model is required
  run_train_test_model_m1 = True
  run_train_test_model_m2 = True
  run_train_test_model_m3 = True
  run_train_test_model_m4 = True

  crossvalidation = 12

  save_split_train_test_data = True



  print('## Execution of pipeline to: ')
  print(code_datasets)

  ind = 0
  indexf = -1

  for i in range(len(code_datasets)):

          indexf = indexf + 1

          df_models_nodes.loc[i,'dataset_name'] = str(code_datasets[i])

          df_models_info.loc[ind,'dataset_name'] = str(code_datasets[i])
          df_models_info.loc[ind+1,'dataset_name'] = str(code_datasets[i])
          df_models_info.loc[ind+2,'dataset_name'] = str(code_datasets[i])
          df_models_info.loc[ind+3,'dataset_name'] = str(code_datasets[i])

          #genarate diretocry
          path_dataset = bar+code_datasets[i]
          dirByDataset(path_dataset)

          dataset = openml.datasets.get_dataset(code_datasets[i])

          print(dataset.get_data)

          X, Y, categorical_indicator, attribute_names = dataset.get_data(
                  dataset_format="dataframe", target=dataset.default_target_attribute)




          dt = {'att_original_names':[],
                'shap_m1':[],
                'shap_m2':[],
                'shap_m3':[],
                'shap_m4':[],
                'eli5_m1':[],
                'eli5_m2':[],
                'eli5_m3':[],
                'eli5_m4':[],
                'dalex_m1':[],
                'dalex_m2':[],
                'dalex_m3':[],
                'dalex_m4':[],
                'ci_m1':[],
                'ci_m2':[],
                'ci_m3':[],
                'ci_m4':[],
                'skater_m1':[],
                'skater_m2':[],
                'skater_m3':[],
                'skater_m4':[],
                'lofo_m1':[],
                'lofo_m2':[],
                'lofo_m3':[],
                'lofo_m4':[],
                'exirt_m1':[],
                'exirt_m2':[],
                'exirt_m3':[],
                'exirt_m4':[]}
          df_feature_rank = pd.DataFrame(dt)
          df_feature_rank['att_original_names'] = attribute_names


          #pre-process
          X = PreprocessDefault.PreprocessXByDataset(X,code_datasets[i])
          Y = PreprocessDefault.PreprocessYByDataset(Y,code_datasets[i])

          #X, Y = PreprocessDefault.PreprocessXY(X, Y)

          attribute_names = X.columns.to_list()



          print(X)

          #normalize data by min-max
          X = PreprocessDefault.normalize(X)

          #fix NaN (incorporar este trecho ao eXirt)
          X['class'] = Y
          X.dropna(inplace=True)
          Y = X['class']
          X = X.drop(columns=['class'])

          #print("pos processado")
          print(X.head(n=100))

          #create context dictionary (necessary to CIU)
          context_dic = {}
          for k in range(len(attribute_names)):
            context_dic[attribute_names[k]] = [min(X[attribute_names[k]]), max(X[attribute_names[k]]), Util.is_int(X[attribute_names[k]])]



          #split data
          X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, stratify=Y, random_state=41) # 95% training and 5% test


          if save_split_train_test_data == True:
              name_dataset = code_datasets[i]

              X_train.to_csv(path_content+path_csv+path_dataset+bar+"X_train_"+name_dataset+".csv")
              X_test.to_csv(path_content+path_csv+path_dataset+bar+"X_test_"+name_dataset+".csv")
              y_train.to_csv(path_content+path_csv+path_dataset+bar+"y_train_"+name_dataset+".csv")
              y_test.to_csv(path_content+path_csv+path_dataset+bar+"y_test_"+name_dataset+".csv")



          #execution of model m1

          file_name = path_content+path_model+path_dataset+bar+'model_m1_'+code_datasets[i]+'.sav'

          df_models_info.loc[ind,'model_name'] = 'm1'
          
          hit_rates_m1 = []
          if run_train_test_model_m1 == True:
            
            # Create a based model

            model_m1 = lgb.LGBMClassifier(verbosity=-1)
            

            hit_rates = AnalysisDefault.calcCrossValidation(model_m1,X, Y, crossvalidation, False)
            hit_rates_m1 = hit_rates
            
            model_m1.fit(X_train, y_train)
            
            

            pickle.dump(model_m1, open(file_name, 'wb'))
          else:
            model_m1 = pickle.load(open(file_name, 'rb'))

          y_pred_m1=model_m1.predict(X_test)
          print(model_m1)
          df_models_info.loc[ind,'model_params'] = str(model_m1.get_params())

          #analysis precision, accuracy and recall
          ac, pr, re, au = AnalysisDefault.calcAccuracyPrecisionRecallByModel(y_test,y_pred_m1)

          df_models_accuracy_analisys.loc[i,'m1'] = ac
          df_models_precision_analisys.loc[i,'m1'] = pr
          df_models_recall_analisys.loc[i,'m1'] = re

          df_models_info.loc[ind,'accuracy'] = ac
          df_models_info.loc[ind,'precision'] = pr
          df_models_info.loc[ind,'recall'] = re
          df_models_info.loc[ind,'auc'] = au
          

          #confusion matrix
          tp, tn = AnalysisDefault.calcConfusionMatrix(y_test,y_pred_m1,"m1")
          df_models_info.loc[ind,'correct_true'] = tp
          df_models_info.loc[ind,'correct_false'] = tn


          

          



          #execution of model m2

          file_name = path_content+path_model+path_dataset+bar+'model_m2_'+code_datasets[i]+'.sav'

          df_models_info.loc[ind+1,'model_name'] = 'm2'
          
          hit_rates_m2 = []
          
          if run_train_test_model_m2 == True:

            #Create a based model
            model_m2 = CatBoostClassifier(silent=True, cat_features=Util.identifyCatFeatures(X))
            
            hit_rates = AnalysisDefault.calcCrossValidation(model_m2,X, Y, crossvalidation, False)
            hit_rates_m2 = hit_rates
            
            model_m2.fit(X_train, y_train)
            
            pickle.dump(model_m2, open(file_name, 'wb'))
          else:
            model_m2 = pickle.load(open(file_name, 'rb'))

          y_pred_m2=model_m2.predict(X_test)
          print(model_m2)
          df_models_info.loc[ind+1,'model_params'] = str(model_m2.get_params())

          #analysis precision, accuracy and recall
          ac, pr, re, au = AnalysisDefault.calcAccuracyPrecisionRecallByModel(y_test,y_pred_m2)

          df_models_accuracy_analisys.loc[i,'m2'] = ac
          df_models_precision_analisys.loc[i,'m2'] = pr
          df_models_recall_analisys.loc[i,'m2'] = re

          df_models_info.loc[ind+1,'accuracy'] = ac
          df_models_info.loc[ind+1,'precision'] = pr
          df_models_info.loc[ind+1,'recall'] = re
          df_models_info.loc[ind+1,'auc'] = au

          #confusion matrix
          tp, tn = AnalysisDefault.calcConfusionMatrix(y_test,y_pred_m2,"m2")
          df_models_info.loc[ind+1,'correct_true'] = tp
          df_models_info.loc[ind+1,'correct_false'] = tn


          #statistical analysis
          hit_rates = AnalysisDefault.calcCrossValidation(model_m2,X, Y, crossvalidation, False)
          hit_rates_m2 = hit_rates

          #execution of model m3

          file_name = path_content+path_model+path_dataset+bar+'model_m3_'+code_datasets[i]+'.sav'

          df_models_info.loc[ind+2,'model_name'] = 'm3'
          
          hit_rates_m3 = []
          if run_train_test_model_m3 == True:

            # Create a based model
            model_m3 = RandomForestClassifier(verbose=0)
            
            hit_rates = AnalysisDefault.calcCrossValidation(model_m3,X, Y, crossvalidation, False)
            hit_rates_m3 = hit_rates
            
            model_m3.fit(X_train, y_train)
            
            pickle.dump(model_m3, open(file_name, 'wb'))
          else:
            model_m3 = pickle.load(open(file_name, 'rb'))

          y_pred_m3=model_m3.predict(X_test)
          print(model_m3)
          df_models_info.loc[ind+2,'model_params'] = str(model_m3.get_params())

          #analysis precision, accuracy and recall
          ac, pr, re, au = AnalysisDefault.calcAccuracyPrecisionRecallByModel(y_test,y_pred_m3)

          df_models_accuracy_analisys.loc[i,'m3'] = ac
          df_models_precision_analisys.loc[i,'m3'] = pr
          df_models_recall_analisys.loc[i,'m3'] = re

          df_models_info.loc[ind+2,'accuracy'] = ac
          df_models_info.loc[ind+2,'precision'] = pr
          df_models_info.loc[ind+2,'recall'] = re
          df_models_info.loc[ind+2,'auc'] = au

          #confusion matrix
          tp, tn = AnalysisDefault.calcConfusionMatrix(y_test,y_pred_m3,"m3")
          df_models_info.loc[ind+2,'correct_true'] = tp
          df_models_info.loc[ind+2,'correct_false'] = tn


          #statistical analysis
          hit_rates = AnalysisDefault.calcCrossValidation(model_m3,X, Y, crossvalidation, False)
          hit_rates_m3 = hit_rates


          	  #execution of model m4

          file_name = path_content+path_model+path_dataset+bar+'model_m4_'+code_datasets[i]+'.sav'

          df_models_info.loc[ind+3,'model_name'] = 'm4'
          
          hit_rates_m4 = []
          if run_train_test_model_m4 == True:

            
            # Create a based model
            model_m4 = GradientBoostingClassifier(verbose=0)
            
            
            hit_rates = AnalysisDefault.calcCrossValidation(model_m4,X, Y, crossvalidation, False)
            hit_rates_m4 = hit_rates
            
            model_m4.fit(X_train, y_train)
            
            #save model
            pickle.dump(model_m4, open(file_name, 'wb'))
          else:
            model_m4 = pickle.load(open(file_name, 'rb'))

          y_pred_m4=model_m4.predict(X_test)
          print(model_m4)
          df_models_info.loc[ind+3,'model_params'] = str(model_m4.get_params())

          #analysis precision, accuracy and recall
          ac, pr, re, au = AnalysisDefault.calcAccuracyPrecisionRecallByModel(y_test,y_pred_m4)

          df_models_accuracy_analisys.loc[i,'m4'] = ac
          df_models_precision_analisys.loc[i,'m4'] = pr
          df_models_recall_analisys.loc[i,'m4'] = re

          df_models_info.loc[ind+3,'accuracy'] = ac
          df_models_info.loc[ind+3,'precision'] = pr
          df_models_info.loc[ind+3,'recall'] = re
          df_models_info.loc[ind+3,'auc'] = au

          #confusion matrix
          tp, tn = AnalysisDefault.calcConfusionMatrix(y_test,y_pred_m4,"m4")
          df_models_info.loc[ind+3,'correct_true'] = tp
          df_models_info.loc[ind+3,'correct_false'] = tn

          
          #statistical analysis
          hit_rates = AnalysisDefault.calcCrossValidation(model_m4,X, Y, crossvalidation, False)
          hit_rates_m4 = hit_rates




          #statisticals comparations
          name = str(path_content+path_fig+path_dataset+bar+"fig_crossval_dataset_"+code_datasets[i]+".pdf")
          AnalysisDefault.plotComparationCrossValidation(hit_rates_m1,hit_rates_m2,hit_rates_m3,hit_rates_m4,name)
          name = str(path_content+path_fig+path_dataset+bar+"fig_crossval_dataset_"+code_datasets[i]+".png")
          AnalysisDefault.plotComparationCrossValidation(hit_rates_m1,hit_rates_m2,hit_rates_m3,hit_rates_m4,name)
          


          name = str(path_content+path_fig+path_dataset+bar+"fig_friedman_dataset_"+code_datasets[i]+".pdf")
          AnalysisDefault.plotFriedmanTest(hit_rates_m1,hit_rates_m2,hit_rates_m3,hit_rates_m4,name)
          name = str(path_content+path_fig+path_dataset+bar+"fig_friedman_dataset_"+code_datasets[i]+".png")
          AnalysisDefault.plotFriedmanTest(hit_rates_m1,hit_rates_m2,hit_rates_m3,hit_rates_m4,name)
          
          friedCoef, line_df_models_friedman = AnalysisDefault.calcFriedmanTest(hit_rates_m1,hit_rates_m2,hit_rates_m3,hit_rates_m4,models_names=models_name)


          df_models_friedman_analisys.loc[len(df_models_friedman_analisys)] = line_df_models_friedman

          X_data_train = X_train.copy(deep=True)
          X_data_test = X_test.copy(deep=True)
          y_data_train = y_train.copy(deep=True)
          y_data_test = y_test.copy(deep=True)

          #explanation by exirt
          print()
          print('eXirt explaning...')
          print('Explaining M1...')
          df_feature_rank['exirt_m1'], temp = explainer.explainRankByEXirt(model_m1, X_data_train, X_data_test, y_data_train, y_data_test,code_datasets[i],model_name='m1')
          print('Explaining M2...')
          df_feature_rank['exirt_m2'], temp = explainer.explainRankByEXirt(model_m2, X_data_train, X_data_test, y_data_train, y_data_test,code_datasets[i],model_name='m2')
          print('Explaining M3...')
          df_feature_rank['exirt_m3'], temp = explainer.explainRankByEXirt(model_m3, X_data_train, X_data_test, y_data_train, y_data_test,code_datasets[i],model_name='m3')
          print('Explaining M4...')
          df_feature_rank['exirt_m4'], temp = explainer.explainRankByEXirt(model_m4, X_data_train, X_data_test, y_data_train, y_data_test,code_datasets[i],model_name='m4')

          X_data = X_test.copy(deep=True)

          #explanation by skater
          print()
          print('Skater explaning...')
          print('Explaining M1...')
          df_feature_rank['skater_m1'] = ExplainableTools.explainRankSkater(model_m1, X_data)
          print('Explaining M2...')
          df_feature_rank['skater_m2'] = ExplainableTools.explainRankSkater(model_m2, X_data)
          #print('Explaining M3...')
          df_feature_rank['skater_m3'] = ExplainableTools.explainRankSkater(model_m3, X_data)
          print('Explaining M4...')
          df_feature_rank['skater_m4'] = ExplainableTools.explainRankSkater(model_m4, X_data)

          del X_data
          X_data = X_test.copy(deep=True)
          y_data = y_test.copy(deep=True)

          #explanation by eli5
          print()
          print('Eli5 explaning...')
          print('Explaining M1...')
          df_feature_rank['eli5_m1'] = ExplainableTools.explainRankByEli5(model_m1, X_data, y_data)
          print('Explaining M2...')
          df_feature_rank['eli5_m2'] = ExplainableTools.explainRankByEli5(model_m2, X_data, y_data)
          print('Explaining M3...')
          df_feature_rank['eli5_m3'] = ExplainableTools.explainRankByEli5(model_m3, X_data, y_data)
          print('Explaining M4...')
          df_feature_rank['eli5_m4'] = ExplainableTools.explainRankByEli5(model_m4, X_data, y_data)

          del X_data
          del y_data
          X_data = X_test.copy(deep=True)

          #explanation by tree shap
          print()
          print('Shap explaning...')
          print('Explaining M1...')
          df_feature_rank['shap_m1'] = ExplainableTools.explainRankByTreeShap(model_m1, attribute_names, X_data)
          print('Explaining M2...')
          df_feature_rank['shap_m2'] = ExplainableTools.explainRankByTreeShap(model_m2,attribute_names, X_data,is_gradient=True)
          print('Explaining M3...')
          df_feature_rank['shap_m3'] = ExplainableTools.explainRankByTreeShap(model_m3,attribute_names, X_data)
          print('Explaining M4...')
          df_feature_rank['shap_m4'] = ExplainableTools.explainRankByTreeShap(model_m4,attribute_names, X_data,is_gradient=True)

          del X_data
          X_data = X_test.copy(deep=True)
          y_data = y_test.copy(deep=True)


          #explanation by dalex
          print()
          print('Dalex explaning...')
          print('Explaining M1...')
          df_feature_rank['dalex_m1'] = ExplainableTools.explainRankDalex(model_m1,X_data, y_data)
          print('Explaining M2...')
          df_feature_rank['dalex_m2'] = ExplainableTools.explainRankDalex(model_m2,X_data, y_data)
          print('Explaining M3...')
          df_feature_rank['dalex_m3'] = ExplainableTools.explainRankDalex(model_m3,X_data, y_data)
          print('Explaining M4...')
          df_feature_rank['dalex_m4'] = ExplainableTools.explainRankDalex(model_m4,X_data, y_data)

          del X_data
          X_data = X_test.copy(deep=True)

          #explanation by ci
          print()
          print('CI explaning...')
          print('Explaining M1...')
          df_feature_rank['ci_m1'] = ExplainableTools.explainRankByCiu(model_m1, X_data, X, attribute_names, context_dic,rank='ci')
          print('Explaining M2...')
          df_feature_rank['ci_m2'] = ExplainableTools.explainRankByCiu(model_m2, X_data, X, attribute_names, context_dic,rank='ci')
          print('Explaining M3...')
          df_feature_rank['ci_m3'] = ExplainableTools.explainRankByCiu(model_m3, X_data, X, attribute_names, context_dic,rank='ci')
          print('Explaining M4...')
          df_feature_rank['ci_m4'] = ExplainableTools.explainRankByCiu(model_m4, X_data, X, attribute_names, context_dic,rank='ci')

          del X_data
          X_data = X_test.copy(deep=True)
          y_data = y_test.copy(deep=True)

          #explanation by lofo
          print()
          print('Lofo explaning...')
          print('Explaining M1...')
          df_feature_rank['lofo_m1'] = ExplainableTools.explainRankByLofo(model_m1, X_data, y_data, attribute_names)
          print('Explaining M2...')
          df_feature_rank['lofo_m2'] = ExplainableTools.explainRankByLofo(model_m2, X_data, y_data, attribute_names)
          print('Explaining M3...')
          df_feature_rank['lofo_m3'] = ExplainableTools.explainRankByLofo(model_m3, X_data, y_data, attribute_names)
          print('Explaining M4...')
          df_feature_rank['lofo_m4'] = ExplainableTools.explainRankByLofo(model_m4, X_data, y_data, attribute_names)

          del X_data
          del y_data
          

          




          #spearman correlation m1

          dt = {'tool':['shap_m1','eli5_m1','dalex_m1','ci_m1','skater_m1','lofo_m1', 'exirt_m1'],
                'shap_m1':[0,0,0,0,0,0,0],
                'eli5_m1':[0,0,0,0,0,0,0],
                'dalex_m1':[0,0,0,0,0,0,0],
                'ci_m1':[0,0,0,0,0,0,0],
                'skater_m1':[0,0,0,0,0,0,0],
                'lofo_m1':[0,0,0,0,0,0,0],
                'exirt_m1':[0,0,0,0,0,0,0],
                }
          df_spearman_correlations_m1 = pd.DataFrame(dt)
          df_spearman_correlations_m1 = df_spearman_correlations_m1.set_index('tool')

          df_spearman_p_m1 = pd.DataFrame(dt)
          df_spearman_p_m1 = df_spearman_p_m1.set_index('tool')


          indexf = indexf + 1

          df_resume_final_boxplot.loc[indexf,'model'] = 'm1'
          df_resume_final_boxplot_p.loc[indexf,'model'] = 'm1'
          for index1, d1 in enumerate(df_spearman_correlations_m1.columns.to_list()):
            for index2, d2 in enumerate(df_spearman_correlations_m1.columns.to_list()):
              if index1 >= index2:
                if d1 != 'tool' and d2 != 'tool':
                  df_spearman_correlations_m1.loc[d1,d2], df_spearman_p_m1.loc[d1,d2] =  AnalysisDefault.calcSpearmanCoef(df_feature_rank[d1],df_feature_rank[d2])
                  df_resume_final_boxplot.loc[indexf,str(d1.replace('_m1','')+'_vs_'+d2.replace('_m1',''))] = df_spearman_correlations_m1.loc[d1,d2]
                  df_resume_final_boxplot_p.loc[indexf,str(d1.replace('_m1','')+'_vs_'+d2.replace('_m1',''))] = df_spearman_p_m1.loc[d1,d2]



          df_spearman_correlations_m1 = df_spearman_correlations_m1.reset_index()
          minSp = -0.7
          maxSp = 0.7
          sns.heatmap(df_spearman_correlations_m1.drop(columns='tool').values,xticklabels=df_spearman_correlations_m1['tool'].values,yticklabels=df_spearman_correlations_m1['tool'].values, vmin=minSp, vmax=maxSp,cmap="RdYlGn",linewidths=.5,annot=True)

          name_fig = path_content+path_fig+path_dataset+bar+"fig_spearman_coeficients_matrix_"+code_datasets[i]+"_m1.pdf"
          plt.savefig(name_fig,bbox_inches='tight')
          name_fig = path_content+path_fig+path_dataset+bar+"fig_spearman_coeficients_matrix_"+code_datasets[i]+"_m1.png"
          plt.savefig(name_fig,bbox_inches='tight')
          
          plt.show()

          df_spearman_p_m1 = df_spearman_p_m1.reset_index()
          minSp = -0.7
          maxSp = 0.7
          sns.heatmap(df_spearman_p_m1.drop(columns='tool').values,xticklabels=df_spearman_p_m1['tool'].values,yticklabels=df_spearman_p_m1['tool'].values, vmin=minSp, vmax=maxSp,cmap="hot",linewidths=.5,annot=True)

          name_fig = path_content+path_fig+path_dataset+bar+"fig_spearman_coeficients_matrix_"+code_datasets[i]+"_m1.pdf"
          plt.savefig(name_fig,bbox_inches='tight')
          name_fig = path_content+path_fig+path_dataset+bar+"fig_spearman_coeficients_matrix_"+code_datasets[i]+"_m1.png"
          plt.savefig(name_fig,bbox_inches='tight')
          plt.show()


          name_df = path_content+path_csv+path_dataset+bar+"df_spearman_coeficients_matrix_"+code_datasets[i]+"_m1.csv"
          df_spearman_correlations_m1.to_csv(name_df)


          name_df = path_content+path_csv+path_dataset+bar+"df_spearman_p_matrix_"+code_datasets[i]+"_m1.csv"
          df_spearman_p_m1.to_csv(name_df)



          #spearman correlation m2

          dt = {'tool':['shap_m2','eli5_m2','dalex_m2','ci_m2','skater_m2','lofo_m2', 'exirt_m2'],
                'shap_m2':  [0,0,0,0,0,0,0],
                'eli5_m2':  [0,0,0,0,0,0,0],
                'dalex_m2': [0,0,0,0,0,0,0],
                'ci_m2':    [0,0,0,0,0,0,0],
                'skater_m2':[0,0,0,0,0,0,0],
                'lofo_m2':  [0,0,0,0,0,0,0],
                'exirt_m2': [0,0,0,0,0,0,0],
                }
          df_spearman_correlations_m2 = pd.DataFrame(dt)
          df_spearman_correlations_m2 = df_spearman_correlations_m2.set_index('tool')

          df_spearman_p_m2 = pd.DataFrame(dt)
          df_spearman_p_m2 = df_spearman_p_m2.set_index('tool')


          indexf = indexf + 1

          df_resume_final_boxplot.loc[indexf,'model'] = 'm2'
          df_resume_final_boxplot_p.loc[indexf,'model'] = 'm2'
          for index1, d1 in enumerate(df_spearman_correlations_m2.columns.to_list()):
            for index2, d2 in enumerate(df_spearman_correlations_m2.columns.to_list()):
              if index1 >= index2:
                if d1 != 'tool' and d2 != 'tool':
                  df_spearman_correlations_m2.loc[d1,d2], df_spearman_p_m2.loc[d1,d2] =  AnalysisDefault.calcSpearmanCoef(df_feature_rank[d1],df_feature_rank[d2])
                  df_resume_final_boxplot.loc[indexf,str(d1.replace('_m2','')+'_vs_'+d2.replace('_m2',''))] = df_spearman_correlations_m2.loc[d1,d2]
                  df_resume_final_boxplot_p.loc[indexf,str(d1.replace('_m2','')+'_vs_'+d2.replace('_m2',''))] = df_spearman_p_m2.loc[d1,d2]



          df_spearman_correlations_m2 = df_spearman_correlations_m2.reset_index()
          minSp = -0.7
          maxSp = 0.7
          sns.heatmap(df_spearman_correlations_m2.drop(columns='tool').values,xticklabels=df_spearman_correlations_m2['tool'].values,yticklabels=df_spearman_correlations_m2['tool'].values, vmin=minSp, vmax=maxSp,cmap="RdYlGn",linewidths=.5,annot=True)

          name_fig = path_content+path_fig+path_dataset+bar+"fig_spearman_coeficients_matrix_"+code_datasets[i]+"_m2.pdf"
          plt.savefig(name_fig,bbox_inches='tight')
          name_fig = path_content+path_fig+path_dataset+bar+"fig_spearman_coeficients_matrix_"+code_datasets[i]+"_m2.png"
          plt.savefig(name_fig,bbox_inches='tight')
          plt.show()

          df_spearman_p_m2 = df_spearman_p_m2.reset_index()
          minSp = -0.7
          maxSp = 0.7
          sns.heatmap(df_spearman_p_m2.drop(columns='tool').values,xticklabels=df_spearman_p_m2['tool'].values,yticklabels=df_spearman_p_m2['tool'].values, vmin=minSp, vmax=maxSp,cmap="hot",linewidths=.5,annot=True)

          name_fig = path_content+path_fig+path_dataset+bar+"fig_spearman_coeficients_matrix_"+code_datasets[i]+"_m2.pdf"
          plt.savefig(name_fig,bbox_inches='tight')
          name_fig = path_content+path_fig+path_dataset+bar+"fig_spearman_coeficients_matrix_"+code_datasets[i]+"_m2.png"
          plt.savefig(name_fig,bbox_inches='tight')
          plt.show()

          

          name_df = path_content+path_csv+path_dataset+bar+"df_spearman_coeficients_matrix_"+code_datasets[i]+"_m2.csv"
          df_spearman_correlations_m2.to_csv(name_df)


          name_df = path_content+path_csv+path_dataset+bar+"df_spearman_p_matrix_"+code_datasets[i]+"_m2.csv"
          df_spearman_p_m2.to_csv(name_df)


          #spearman correlation m3

          dt = {'tool':['shap_m3','eli5_m3','dalex_m3','ci_m3','skater_m3','lofo_m3', 'exirt_m3'],
                'shap_m3':  [0,0,0,0,0,0,0],
                'eli5_m3':  [0,0,0,0,0,0,0],
                'dalex_m3': [0,0,0,0,0,0,0],
                'ci_m3':    [0,0,0,0,0,0,0],
                'skater_m3':[0,0,0,0,0,0,0],
                'lofo_m3':  [0,0,0,0,0,0,0],
                'exirt_m3': [0,0,0,0,0,0,0],
                }
          df_spearman_correlations_m3 = pd.DataFrame(dt)
          df_spearman_correlations_m3 = df_spearman_correlations_m3.set_index('tool')

          df_spearman_p_m3 = pd.DataFrame(dt)
          df_spearman_p_m3 = df_spearman_p_m3.set_index('tool')


          indexf = indexf + 1

          df_resume_final_boxplot.loc[indexf,'model'] = 'm3'
          df_resume_final_boxplot_p.loc[indexf,'model'] = 'm3'
          for index1, d1 in enumerate(df_spearman_correlations_m3.columns.to_list()):
            for index2, d2 in enumerate(df_spearman_correlations_m3.columns.to_list()):
              if index1 >= index2:
                if d1 != 'tool' and d2 != 'tool':
                  df_spearman_correlations_m3.loc[d1,d2], df_spearman_p_m3.loc[d1,d2] =  AnalysisDefault.calcSpearmanCoef(df_feature_rank[d1],df_feature_rank[d2])
                  df_resume_final_boxplot.loc[indexf,str(d1.replace('_m3','')+'_vs_'+d2.replace('_m3',''))] = df_spearman_correlations_m3.loc[d1,d2]
                  df_resume_final_boxplot_p.loc[indexf,str(d1.replace('_m3','')+'_vs_'+d2.replace('_m3',''))] = df_spearman_p_m3.loc[d1,d2]



          df_spearman_correlations_m3 = df_spearman_correlations_m3.reset_index()
          minSp = -0.7
          maxSp = 0.7
          sns.heatmap(df_spearman_correlations_m3.drop(columns='tool').values,xticklabels=df_spearman_correlations_m3['tool'].values,yticklabels=df_spearman_correlations_m3['tool'].values, vmin=minSp, vmax=maxSp,cmap="RdYlGn",linewidths=.5,annot=True)

          name_fig = path_content+path_fig+path_dataset+bar+"fig_spearman_coeficients_matrix_"+code_datasets[i]+"_m2.pdf"
          plt.savefig(name_fig,bbox_inches='tight')
          name_fig = path_content+path_fig+path_dataset+bar+"fig_spearman_coeficients_matrix_"+code_datasets[i]+"_m2.png"
          plt.savefig(name_fig,bbox_inches='tight')
          plt.show()

          df_spearman_p_m3 = df_spearman_p_m3.reset_index()
          minSp = -0.7
          maxSp = 0.7
          sns.heatmap(df_spearman_p_m3.drop(columns='tool').values,xticklabels=df_spearman_p_m3['tool'].values,yticklabels=df_spearman_p_m3['tool'].values, vmin=minSp, vmax=maxSp,cmap="hot",linewidths=.5,annot=True)

          name_fig = path_content+path_fig+path_dataset+bar+"fig_spearman_coeficients_matrix_"+code_datasets[i]+"_m3.pdf"
          plt.savefig(name_fig,bbox_inches='tight')
          name_fig = path_content+path_fig+path_dataset+bar+"fig_spearman_coeficients_matrix_"+code_datasets[i]+"_m3.png"
          plt.savefig(name_fig,bbox_inches='tight')
          plt.show()


          name_df = path_content+path_csv+path_dataset+bar+"df_spearman_coeficients_matrix_"+code_datasets[i]+"_m3.csv"
          df_spearman_correlations_m3.to_csv(name_df)


          name_df = path_content+path_csv+path_dataset+bar+"df_spearman_p_matrix_"+code_datasets[i]+"_m3.csv"
          df_spearman_p_m3.to_csv(name_df)


          #spearman correlation m4

          dt = {'tool':['shap_m4','eli5_m4','dalex_m4','ci_m4','skater_m4','lofo_m4','exirt_m4'],
                'shap_m4':[0,0,0,0,0,0,0],
                'eli5_m4':[0,0,0,0,0,0,0],
                'dalex_m4':[0,0,0,0,0,0,0],
                'ci_m4':[0,0,0,0,0,0,0],
                'skater_m4':[0,0,0,0,0,0,0],
                'lofo_m4':[0,0,0,0,0,0,0],
                'exirt_m4':[0,0,0,0,0,0,0],
                }
          df_spearman_correlations_m4 = pd.DataFrame(dt)
          df_spearman_correlations_m4 = df_spearman_correlations_m4.set_index('tool')

          df_spearman_p_m4 = pd.DataFrame(dt)
          df_spearman_p_m4 = df_spearman_p_m4.set_index('tool')


          indexf = indexf + 1

          df_resume_final_boxplot.loc[indexf,'model'] = 'm4'
          df_resume_final_boxplot_p.loc[indexf,'model'] = 'm4'
          for index1, d1 in enumerate(df_spearman_correlations_m4.columns.to_list()):
            for index2, d2 in enumerate(df_spearman_correlations_m4.columns.to_list()):
              if index1 >= index2:
                if d1 != 'tool' and d2 != 'tool':
                  df_spearman_correlations_m4.loc[d1,d2], df_spearman_p_m4.loc[d1,d2] =  AnalysisDefault.calcSpearmanCoef(df_feature_rank[d1],df_feature_rank[d2])
                  df_resume_final_boxplot.loc[indexf,str(d1.replace('_m4','')+'_vs_'+d2.replace('_m4',''))] = df_spearman_correlations_m4.loc[d1,d2]
                  df_resume_final_boxplot_p.loc[indexf,str(d1.replace('_m4','')+'_vs_'+d2.replace('_m4',''))] = df_spearman_p_m4.loc[d1,d2]



          df_spearman_correlations_m4 = df_spearman_correlations_m4.reset_index()
          minSp = -0.7
          maxSp = 0.7
          sns.heatmap(df_spearman_correlations_m4.drop(columns='tool').values,xticklabels=df_spearman_correlations_m4['tool'].values,yticklabels=df_spearman_correlations_m4['tool'].values, vmin=minSp, vmax=maxSp,cmap="RdYlGn",linewidths=.5,annot=True)

          name_fig = path_content+path_fig+path_dataset+bar+"fig_spearman_coeficients_matrix_"+code_datasets[i]+"_m4.pdf"
          plt.savefig(name_fig,bbox_inches='tight')
          name_fig = path_content+path_fig+path_dataset+bar+"fig_spearman_coeficients_matrix_"+code_datasets[i]+"_m4.png"
          plt.savefig(name_fig,bbox_inches='tight')
          plt.show()

          df_spearman_p_m4 = df_spearman_p_m4.reset_index()
          minSp = -0.7
          maxSp = 0.7
          sns.heatmap(df_spearman_p_m4.drop(columns='tool').values,xticklabels=df_spearman_p_m4['tool'].values,yticklabels=df_spearman_p_m4['tool'].values, vmin=minSp, vmax=maxSp,cmap="hot",linewidths=.5,annot=True)

          name_fig = path_content+path_fig+path_dataset+bar+"fig_spearman_coeficients_matrix_"+code_datasets[i]+"_m4.pdf"
          plt.savefig(name_fig,bbox_inches='tight')
          name_fig = path_content+path_fig+path_dataset+bar+"fig_spearman_coeficients_matrix_"+code_datasets[i]+"_m4.png"
          plt.savefig(name_fig,bbox_inches='tight')
          plt.show()


          name_df = path_content+path_csv+path_dataset+bar+"df_spearman_coeficients_matrix_"+code_datasets[i]+"_m4.csv"
          df_spearman_correlations_m4.to_csv(name_df)


          name_df = path_content+path_csv+path_dataset+bar+"df_spearman_p_matrix_"+code_datasets[i]+"_m4.csv"
          df_spearman_p_m4.to_csv(name_df)


          ind = ind + 4

          name = path_content+path_csv+bar+'df_models_info.csv'
          df_models_info.to_csv(name)

          name_df = path_content+path_csv+path_dataset+bar+"df_features_ranks_by_"+code_datasets[i]+".csv"
          df_feature_rank.to_csv(name_df)



  return df_resume_final_boxplot, df_resume_final_boxplot_p, df_dataset_properties_cluster, models_name, df_models_friedman_analisys, df_models_accuracy_analisys, df_models_precision_analisys, df_models_recall_analisys

def plot_resume_full(dfs_ret, df_dataset_properties_cluster, models_name, cluster, min, max,step_precision,ylabel,tag):

  #fix plot after remaker
  dfs_ret = dfs_ret.dropna(axis=1)
  dfs_ret = dfs_ret.drop(columns=['shap_vs_shap',
                                  'eli5_vs_eli5',
                                  'lofo_vs_lofo',
                                  'skater_vs_skater',
                                  'exirt_vs_exirt',
                                  'ci_vs_ci',
                                  'dalex_vs_dalex'])

  dfs_ret = [dfs_ret[dfs_ret['model'] == 'm1'],
             dfs_ret[dfs_ret['model'] == 'm2'],
             dfs_ret[dfs_ret['model'] == 'm3'],
             dfs_ret[dfs_ret['model'] == 'm4']]



  for ii, df_boxplot in enumerate(dfs_ret):


      #df_boxplot.boxplot()
      figure(figsize=(6, 4), dpi=200)
      meds = df_boxplot.median()
      meds.sort_values(ascending=True, inplace=True)
      df2 = df_boxplot[meds.index]

      axes = df2.boxplot(rot=90,return_type='axes')
      plt.title('C'+str(cluster)+' and '+models_name[ii].upper())
      axes.set_ylim(min, max)
      plt.ylabel(ylabel)
      major_ticks = np.arange(min, max, step_precision)


      axes.set_yticks(major_ticks)

      plt.savefig(path_content+path_fig+bar+'resume_pipeline_'+models_name[ii]+'_'+tag+'_full_cluster_'+str(cluster)+'.pdf',bbox_inches='tight')
      plt.savefig(path_content+path_fig+bar+'resume_pipeline_'+models_name[ii]+'_'+tag+'_full_cluster_'+str(cluster)+'.png',bbox_inches='tight')
      
      plt.show()


  dfs_ret_ultimate = dfs_ret.copy()


  for i, df_comparation in enumerate(dfs_ret_ultimate):
    list_lofo_medians = []
    list_exirt_medians = []
    list_xai_medians = []
    for ii, col in enumerate(df_comparation.columns):
       if (col == 'eli5_vs_shap'   or
           col == 'skater_vs_ci'   or
           col == 'dalex_vs_shap'  or
           col == 'dalex_vs_eli5'  or
           col == 'ci_vs_eli5'     or
           col == 'ci_vs_dalex'    or
           col == 'skater_vs_shap' or
           col == 'skater_vs_eli5' or
           col == 'skater_vs_dalex'or
           col == 'ci_vs_shap'):
         list_xai_medians.append(df_comparation[col].median())

       if (col == 'exirt_vs_shap'   or
           col == 'exirt_vs_eli5'   or
           col == 'exirt_vs_dalex'  or
           col == 'exirt_vs_ci'     or
           col == 'exirt_vs_skater'):
          list_exirt_medians.append(df_comparation[col].median())

       if (col == 'lofo_vs_shap'   or
           col == 'lofo_vs_eli5'   or
           col == 'lofo_vs_dalex'  or
           col == 'lofo_vs_ci'     or
           col == 'lofo_vs_skater'):
          list_lofo_medians.append(df_comparation[col].median())

  #   list_lofo_means_of_medians.append(sum(list_lofo_medians)/len(list_lofo_medians))
  #   list_exirt_means_of_medians.append(sum(list_exirt_medians)/len(list_exirt_medians))
  #   list_xai_means_of_medians.append(sum(list_xai_medians)/len(list_xai_medians))

  # df = pd.DataFrame({
  #   'MOM eXirt': list_exirt_means_of_medians,
  #   'MOM XAI': list_xai_means_of_medians,
  #   'MOM Lofo': list_lofo_means_of_medians}, index=[1, 2, 3, 4])
  # df.plot.line(grid=True,figsize=(7,6),xticks=[1,2,3,4],yticks=np.arange(0, 1.1, step=0.05),ylim=(0,1.1),style='-o')
  # return df

def plot_resume_light(dfs_ret, df_dataset_properties_cluster, models_name, cluster, min, max,step_precision,ylabel,tag):

  #fix plot after remaker
  dfs_ret = dfs_ret.dropna(axis=1)
  dfs_ret = dfs_ret.drop(columns=['dalex_vs_eli5',
                                  'lofo_vs_dalex',
                                  'skater_vs_eli5',
                                  'ci_vs_dalex',
                                  'lofo_vs_shap',
                                  'dalex_vs_shap',
                                  'skater_vs_ci',
                                  'ci_vs_shap',
                                  'lofo_vs_skater',
                                  'ci_vs_eli5',
                                  'eli5_vs_shap',
                                  'lofo_vs_ci',
                                  'lofo_vs_eli5',
                                  'skater_vs_shap',
                                  'skater_vs_dalex',
                                  'shap_vs_shap',
                                  'eli5_vs_eli5',
                                  'lofo_vs_lofo',
                                  'skater_vs_skater',
                                  'exirt_vs_exirt',
                                  'ci_vs_ci',
                                  'dalex_vs_dalex'])

  dfs_ret = [dfs_ret[dfs_ret['model'] == 'm1'],
             dfs_ret[dfs_ret['model'] == 'm2'],
             dfs_ret[dfs_ret['model'] == 'm3'],
             dfs_ret[dfs_ret['model'] == 'm4']]



  for ii, df_boxplot in enumerate(dfs_ret):


      #df_boxplot.boxplot()
      figure(figsize=(2, 4), dpi=200)
      meds = df_boxplot.median()
      meds.sort_values(ascending=True, inplace=True)
      df2 = df_boxplot[meds.index]

      axes = df2.boxplot(rot=90,return_type='axes')
      plt.title('C'+str(cluster)+' and '+models_name[ii].upper())
      axes.set_ylim(min, max)
      plt.ylabel(ylabel)
      major_ticks = np.arange(min, max, step_precision)


      axes.set_yticks(major_ticks)

      plt.savefig(path_content+path_fig+bar+'resume_pipeline_'+models_name[ii]+'_'+tag+'_light_cluster_'+str(cluster)+'.pdf',bbox_inches='tight')
      plt.savefig(path_content+path_fig+bar+'resume_pipeline_'+models_name[ii]+'_'+tag+'_light_cluster_'+str(cluster)+'.png',bbox_inches='tight')
      
      plt.show()


  dfs_ret_ultimate = dfs_ret.copy()

  #list_lofo_means_of_medians = []
  #list_exirt_means_of_medians = []
  #list_xai_means_of_medians = []

  for i, df_comparation in enumerate(dfs_ret_ultimate):
    list_lofo_medians = []
    list_exirt_medians = []
    list_xai_medians = []
    for ii, col in enumerate(df_comparation.columns):
       if (col == 'eli5_vs_shap'   or
           col == 'skater_vs_ci'   or
           col == 'dalex_vs_shap'  or
           col == 'dalex_vs_eli5'  or
           col == 'ci_vs_eli5'     or
           col == 'ci_vs_dalex'    or
           col == 'skater_vs_shap' or
           col == 'skater_vs_eli5' or
           col == 'skater_vs_dalex'or
           col == 'ci_vs_shap'):
         list_xai_medians.append(df_comparation[col].median())

       if (col == 'exirt_vs_shap'   or
           col == 'exirt_vs_eli5'   or
           col == 'exirt_vs_dalex'  or
           col == 'exirt_vs_ci'     or
           col == 'exirt_vs_skater'):
          list_exirt_medians.append(df_comparation[col].median())

       if (col == 'lofo_vs_shap'   or
           col == 'lofo_vs_eli5'   or
           col == 'lofo_vs_dalex'  or
           col == 'lofo_vs_ci'     or
           col == 'lofo_vs_skater'):
          list_lofo_medians.append(df_comparation[col].median())



def main():
    print("===========================================================================")
    print("============================== DATA ORGANIZER =============================")
    print("===========================================================================")
    print("")
    
    code_datasets = ['ozone-level-8hr',# 73
                  'sonar',# 61
                  'spambase',# 58
                  'qsar-biodeg',# 42
                  'kc3',# 40
                  'mc1',# 39
                  'pc3',# 38
                  'mw1',# 38
                  'pc4',# 38
                  'kr-vs-kp',# 37 CLUSTER 3
                  'Satellite',# 37
                  'pc2',# 37
                  'ionosphere',# 35
                  'steel-plates-fault',# 34
                  'PhishingWebsites',# 31 CLUSTER 3
                  'wdbc',# 31 voltou
                  'SPECT',# 23 CLUSTER 3
                  'kc2',# 22
                  'pc1',# 22
                  'kc1',# 22
                  'credit-g',# 21
                  'churn',# 21
                  'climate-model-simulation-crashes',# 21
                  'Australian',# 15
                  'eeg-eye-state',# 15 voltou
                  'heart-statlog',# 14 voltou
                  'ilpd',# 11
                  'tic-tac-toe',# 10
                  'jEdit_4.0_4.2',# 9
                  'diabetes',# 9
                  'prnn_crabs',# 8
                  'monks-problems-1',# 7
                  'monks-problems-3',# 7
                  'monks-problems-2',# 7
                  'delta_ailerons',# 6
                  'mozilla4',# 6
                  'phoneme',# 6
                  'blood-transfusion-service-center',# 5
                  'banknote-authentication',# 5
                  'haberman', # 4,
                  'analcatdata_lawsuit'# 5
                 ]
    
    #rapid test
    _code_datasets = ['climate-model-simulation-crashes',# 21
                  'Australian',# 15
                  'eeg-eye-state',# 15 voltou
                  'heart-statlog',# 14 voltou
                  'ilpd',# 11
                  'tic-tac-toe',# 10
                  'jEdit_4.0_4.2',# 9
                  'diabetes',# 9
                  'prnn_crabs',# 8
                  'monks-problems-1',# 7
                  'monks-problems-3',# 7
                  'monks-problems-2',# 7
                  'delta_ailerons',# 6
                  'mozilla4',# 6
                  'phoneme',# 6
                  'blood-transfusion-service-center',# 5
                  'banknote-authentication',# 5
                  'haberman', # 4,
                  'analcatdata_lawsuit'# 5
                 ]
    
    
    print("===========================================================================")
    print("========================== EXTRACT DATASET PROPERTIES =====================")
    print("===========================================================================")
    print("")
    
    print("--> Extraindo propriedades dos dados...")
    #Lista de ID's ou nome dos datasets que se deseja resgatar os metadados

    list_datasets = code_datasets
    
    #Cria a pasta cache para salvar os dados do OpenML
    openml.config.cache_directory = os.path.expanduser(os.getcwd()+'/cache')
    
    #Lista de todos os metadados existentes no OpenML
    metadados = ['NumberOfInstances', 'NumberOfFeatures', 'NumberOfClasses', 'NumberOfMissingValues', 'NumberOfInstancesWithMissingValues',
                 'NumberOfNumericFeatures', 'NumberOfSymbolicFeatures', 'NumberOfBinaryFeatures', 'StdvNominalAttDistinctValues',
                 'MeanNominalAttDistinctValues', 'MeanSkewnessOfNumericAtts', 'MajorityClassPercentage', 'MeanStdDevOfNumericAtts', 'ClassEntropy',
                 'MajorityClassSize', 'MinAttributeEntropy', 'MaxAttributeEntropy', 'MinMeansOfNumericAtts', 'MaxMeansOfNumericAtts', 'Dimensionality',
                 'PercentageOfBinaryFeatures', 'MinNominalAttDistinctValues', 'EquivalentNumberOfAtts', 'MaxNominalAttDistinctValues', 'PercentageOfInstancesWithMissingValues',
                 'MaxSkewnessOfNumericAtts', 'MinStdDevOfNumericAtts', 'PercentageOfMissingValues', 'AutoCorrelation', 'MaxStdDevOfNumericAtts', 'MinorityClassPercentage',
                 'PercentageOfNumericFeatures', 'MeanAttributeEntropy', 'MinorityClassSize', 'PercentageOfSymbolicFeatures','MeanMutualInformation']
    
    metadados = ['NumberOfFeatures','NumberOfInstances','Dimensionality',
                 'PercentageOfBinaryFeatures','StdvNominalAttDistinctValues',
                 'MeanNominalAttDistinctValues','ClassEntropy','AutoCorrelation',
                 'NumberOfNumericFeatures', 'NumberOfSymbolicFeatures',
                 'NumberOfBinaryFeatures','PercentageOfSymbolicFeatures',
                 'PercentageOfNumericFeatures', 'MajorityClassPercentage',
                 'MinorityClassPercentage']
    
    #metadados = ['NumberOfFeatures','NumberOfInstances']
    
    
    listDataQltsOnly = {}
    list_name = {}
    for i in tqdm(range(len(list_datasets))):
        dataset = openml.datasets.get_dataset(list_datasets[i],download_data = False)
        #Original
        #listValues = list(dataset.qualities.values())
        #listKeys = list(dataset.qualities.keys())
        list_name[dataset.dataset_id] = dataset.name
        dataDict = {}
        key = 0
        for j in metadados:
            try:
                if not(math.isnan(dataset.qualities[j])):
                    dataDict[j] = dataset.qualities[j]
            except:
                key = 1
                break
    #    dataDict["did"] = dataset.dataset_id
    #    listDataQlts[i] = dataDict
        if key ==0:
            listDataQltsOnly[dataset.name] = dataDict
        else:
            continue
    
        gc.collect()
    
    
    df_dataset_properties = pd.DataFrame.from_dict(listDataQltsOnly, orient='index')
    df_dataset_properties.to_csv(path_content+path_csv+bar+'df_dataset_properties.csv')
    print("Salvando arquivo: df_dataset_properties.csv")
    
    print("===========================================================================")
    print("============================== CLUSTERING =================================")
    print("===========================================================================")
    print("")
    min_max_scaler = preprocessing.MinMaxScaler()
    #mStandardScaler = preprocessing.StandardScaler()
    data_norm = min_max_scaler.fit_transform(df_dataset_properties)
    #data_norm = PreprocessDefault.z_score(df_dataset_properties)
    
    print("--> Elbow test...")
    Sum_of_squared_distances = []
    n = 12
    K = range(1,n)
    for k in K:
        km = KMeans(n_clusters=k,random_state=0,init='k-means++',n_init = 1000, max_iter = 5000)
        km = km.fit(data_norm)
        Sum_of_squared_distances.append(km.inertia_)
    
    plt.figure(figsize=(7,7))
    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xticks(np.arange(1,n,2))
    plt.grid(axis='x')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    diretorio = path_content+path_fig+bar+'elbow_'+str(n)+'.pdf'
    plt.savefig(diretorio,bbox_inches='tight')
    diretorio = path_content+path_fig+bar+'elbow_'+str(n)+'.png'
    plt.savefig(diretorio,bbox_inches='tight')
    print("--> Gráfico Elbow salvo em: "+diretorio)
    plt.show()
    
    print("--> Silhouette test...")
    #https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html

    k = 5
    range_n_clusters = range(2, k)
    
    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, ax1 = plt.subplots()
        fig.set_size_inches(6, 7)
    
        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(data_norm) + (n_clusters + 1) * 10])
    
        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        kmeans = KMeans(n_clusters=n_clusters, random_state=0,init='k-means++',n_init = 1000, max_iter = 5000)
        cluster_labels = kmeans.fit_predict(data_norm)
        df_dataset_properties['clusters_kmeans'+str(n_clusters)] = cluster_labels
    
        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(data_norm, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)
    
        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(data_norm, cluster_labels)
    
        y_lower = 10
    
        my_colors = ['blue', 'red', 'orange', 'green']
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]
    
            ith_cluster_silhouette_values.sort()
    
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
    
            #color = cm.nipy_spectral(float(i) / n_clusters)
            color = my_colors[i]
    
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.9)
    
            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    
            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples
    
        ax1.set_title("The silhouette plot for the "+str(n_clusters)+" clusters.",fontsize=15)
        ax1.set_xlabel("The silhouette coefficient values",fontsize=14)
        ax1.set_ylabel("Cluster label",fontsize=14)
    
        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="black", linestyle="--")
        ax1.text(float(silhouette_avg)+0.01, 1.5, str(round(silhouette_avg,3)),fontsize=14,color='black')
        ax1.legend(['Cluster 0','Cluster 1', 'Cluster 2', 'Cluster 3','Average\nSilhouette\nWidth'],fontsize=14)
        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    
        #2nd Plot showing the actual clusters formed
        #colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    
    
    
        #plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
        #              "with n_clusters = %d" % n_clusters),
        #             fontsize=14, fontweight='bold')
        diretorio = path_content+path_fig+bar+'silhouette_clusters_'+str(n_clusters)+'.pdf'
        plt.savefig(diretorio,bbox_inches='tight')
        diretorio = path_content+path_fig+bar+'silhouette_clusters_'+str(n_clusters)+'.png'
        plt.savefig(diretorio,bbox_inches='tight')
        print("--> Gráfico Silhouette salvo em: "+diretorio)
    plt.show()
    
    print("--> Alguns prints...")
    k_cluster_selected = 'clusters_kmeans4'
    #df_dataset_properties.plot(rot=90, subplots=True,layout=(8,4),figsize=(30,20),kind='bar',grid=True)
    print("Clusters...")
    print(df_dataset_properties[k_cluster_selected].value_counts())
    print()
    print('Datasets for cluster 0')
    print(df_dataset_properties.query(k_cluster_selected+' == 0').index.to_list())
    print()
    print('Datasets for cluster 1')
    print(df_dataset_properties.query(k_cluster_selected+' == 1').index.to_list())
    print()
    print('Datasets for cluster 2')
    print(df_dataset_properties.query(k_cluster_selected+' == 2').index.to_list())
    print()
    print('Datasets for cluster 3')
    print(df_dataset_properties.query(k_cluster_selected+' == 3').index.to_list())
    
    print("dataset properties...")
    print(df_dataset_properties.head(4))
    
    diretorio = path_content+path_csv+bar+'df_dataset_properties_roud2.csv'
    df_dataset_properties.round(2).to_csv(diretorio)
    print("Salvando arquivo csv: "+diretorio)
    


    print("===========================================================================")
    print("==================================== MCA ==================================")
    print("===========================================================================")
    print("")
    df_dataset_properties_copy = df_dataset_properties.copy()
    #df_dataset_properties_copy = df_dataset_properties_copy.drop(columns=k_cluster_selected)
    
    df_dataset_properties_copy.to_csv('df_properties.csv')
    #files.download('df_properties.csv')
    
    df_dataset_properties_copy = df_dataset_properties_copy.drop(columns='clusters_kmeans2')
    df_dataset_properties_copy = df_dataset_properties_copy.drop(columns='clusters_kmeans3')
    df_dataset_properties_copy = df_dataset_properties_copy.drop(columns='clusters_kmeans4')
    
    df_dataset_properties = df_dataset_properties.drop(columns='clusters_kmeans2')
    df_dataset_properties = df_dataset_properties.drop(columns='clusters_kmeans3')
    
    for i,c in enumerate(df_dataset_properties_copy.columns):
      if c != k_cluster_selected:
        #try:
        #  df_dataset_properties_copy[c] = pd.qcut(df_dataset_properties_copy[c], q=2,duplicates='drop',labels=['s','h'])
        #except:
        df_dataset_properties_copy[c] = pd.cut(df_dataset_properties_copy[c], bins=2,duplicates='drop',labels=['s','h'])
        
    k4 = df_dataset_properties['clusters_kmeans4']
    df_dataset_properties = df_dataset_properties.drop(columns='clusters_kmeans4')
    df_dataset_properties = PreprocessDefault.normalize(df_dataset_properties)
    df_dataset_properties['clusters_kmeans4'] = k4
    
    diretorio = path_content+path_csv+bar+'df_dataset_properties_round2_norm.csv'
    df_dataset_properties.round(2).to_csv(diretorio)
    print("Salvando arquivo csv: "+diretorio)
    
    print("--> Executando o MCA...")
    mca = MCA()
    mca = mca.fit(df_dataset_properties_copy)
    mca.plot_coordinates(
     X=df_dataset_properties_copy,
     ax=None,
     figsize=(15,20),
     show_row_points=True,
     row_points_size=200,
     show_row_labels=False,
     show_column_points=True,
     column_points_size=100,
     show_column_labels=True,
     legend_n_cols=1,
     xlim=(-1.3,2.7),
     ylim=(-0.8,2.3),
     c=[True,True,True,True],
     row_points_alpha=0.6,
     colors = ['#5496fa','#aec7e8','#ff7f0e','#ffbb78',
               '#539c8e','#556B2F','#E9967A','#F0E68C',
               '#9467bd','#c5b0d5','#8c564b','#c49c94',
               '#e377c2','#f7b6d2','#7f7f7f','#c7c7c7',
               '#bcbd22','#dbdb8d','#17becf','#9edae5'],
     df_dataset_properties=df_dataset_properties,
     k_cluster_selected=k_cluster_selected)
    diretorio = path_content+path_fig+bar+"mca_resume.pdf"
    plt.savefig(diretorio,bbox_inches='tight')
    diretorio = path_content+path_fig+bar+"mca_resume.png"
    plt.savefig(diretorio,bbox_inches='tight')
    print("Salvando figura em: "+diretorio)
    
    
    print("===========================================================================")
    print("==================================== MAIN ==================================")
    print("===========================================================================")
    print("")
    
    print("")
    print("--> Iniciando pipeline para datasets do cluster 0...")
    print("")
    
    
    cluster = 0

    df_prop_tmp = df_dataset_properties.query(k_cluster_selected + ' == '+str(cluster))
    list_dataset = list(df_dataset_properties.query(k_cluster_selected + ' == '+str(cluster)).index)
    dfs_ret_0, dfs_ret_0_p, df_dataset_properties_cluster_0, models_name_0, df_friedman_resume_0, df_accuracy_resume_0, df_precision_resume_0, df_recall_resume_0 = main_pipeline(list_dataset, df_prop_tmp,cluster_name=str(cluster))

    print("Gerando gráficos resumo de p-value:")
    plot_resume_full(dfs_ret_0_p, df_dataset_properties_cluster_0, models_name_0, cluster,0,1,0.05,"p-value",'p')
    plot_resume_light(dfs_ret_0_p, df_dataset_properties_cluster_0, models_name_0, cluster,0,1,0.05,"p-value",'p')    
    
    
    print("Gerando gráficos resumo de spearman:")
    plot_resume_full(dfs_ret_0, df_dataset_properties_cluster_0, models_name_0, cluster,-1.1,1.1,0.1,"Spearman Correlation",'coef')

    plot_resume_light(dfs_ret_0, df_dataset_properties_cluster_0, models_name_0, cluster,-1.1,1.1,0.1,"Spearman Correlation",'coef')
    
    
    
      
    figure(figsize=(2,3), dpi=200)
    axes = df_friedman_resume_0.boxplot(return_type = 'axes', rot=90)
    plt.title('Friedman Test\nCluster: '+ str(cluster))
    major_ticks = np.arange(0, 1, 0.05)
    axes.set_yticks(major_ticks)
    axes.set_ylabel('p-values')
    diretorio = path_content+path_fig+bar+'resume_friedman_test_cluster_'+str(cluster)+'.pdf'
    plt.savefig(diretorio,bbox_inches='tight')
    diretorio = path_content+path_fig+bar+'resume_friedman_test_cluster_'+str(cluster)+'.png'
    plt.savefig(diretorio,bbox_inches='tight')
    plt.show()    
    
    figure(figsize=(2,3), dpi=200)
    axes = df_accuracy_resume_0.boxplot(return_type = 'axes')
    axes.set_ylabel('Accuracy')
    plt.title('Accuracy of models\nby cluster: '+ str(cluster))
    #axes.set_ylim(0, 1)
    diretorio = path_content+path_fig+bar+'resume_accuracy_test_cluster_'+str(cluster)+'.pdf'
    plt.savefig(diretorio,bbox_inches='tight')
    diretorio = path_content+path_fig+bar+'resume_accuracy_test_cluster_'+str(cluster)+'.png'
    plt.savefig(diretorio,bbox_inches='tight')
    plt.show()
    
    
    figure(figsize=(2,3), dpi=200)
    axes = df_precision_resume_0.boxplot(return_type = 'axes')
    axes.set_ylabel('Precision')
    plt.title('Precision of models\nby cluster: '+ str(cluster))
    #axes.set_ylim(0, 1)
    diretorio = path_content+path_fig+bar+'resume_precision_test_cluster_'+str(cluster)+'.pdf'
    plt.savefig(diretorio,bbox_inches='tight')
    diretorio = path_content+path_fig+bar+'resume_precision_test_cluster_'+str(cluster)+'.png'
    plt.savefig(diretorio,bbox_inches='tight')
    plt.show()
    
    figure(figsize=(2,3), dpi=200)
    axes = df_recall_resume_0.boxplot(return_type = 'axes')
    axes.set_ylabel('Recall')
    plt.title('Recall of models\nby cluster: '+ str(cluster))
    diretorio = path_content+path_fig+bar+'resume_recall_test_cluster_'+str(cluster)+'.pdf'
    plt.savefig(diretorio,bbox_inches='tight')
    diretorio = path_content+path_fig+bar+'resume_recall_test_cluster_'+str(cluster)+'.png'
    plt.savefig(diretorio,bbox_inches='tight')
    plt.show()
    
    df_metrics = pd.concat([df_accuracy_resume_0, df_precision_resume_0, df_recall_resume_0])
    print("Df metrics")
    print(df_metrics)
    print("")
    matrix = AnalysisDefault.calcFriedmanTestByDf(df_metrics, models_name_0)
    print("Resume friedman based in accuracy")
    print(matrix)
    figure(figsize=(3,3), dpi=200)
    plt.title('Friedman test of models\nby cluster '+ str(cluster))
    axes.set_ylabel('p-values of models')
    axes.set_xlabel('p-values of models')
    sns.heatmap(matrix,xticklabels=models_name_0,yticklabels = models_name_0,vmin=0, vmax=1,cmap="crest",linewidths=.5,annot=True,fmt=".2f")
    diretorio = path_content+path_fig+bar+'resume_friedman_based_in_acc_pre_re_cluster_'+str(cluster)+'.pdf'
    plt.savefig(diretorio,bbox_inches='tight')
    diretorio = path_content+path_fig+bar+'resume_friedman_based_in_acc_pre_re_cluster_'+str(cluster)+'.png'
    plt.savefig(diretorio,bbox_inches='tight')
    plt.show()
    

if __name__ == "__main__":
    main()



