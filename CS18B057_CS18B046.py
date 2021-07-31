##############################
# PRML DATA CONTEST DEC 2020 #
# BY - CS18B057, CS18B046    #
##############################

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import normalize
from collections import defaultdict
import itertools
from datetime import datetime
import geopy

#Loading data files
#bikers = pd.read_csv("../../data/bikers.csv")
bikers_network = pd.read_csv("../../data/bikers_network.csv")
tour_convoy = pd.read_csv("../../data/tour_convoy.csv")
#tours = pd.read_csv("../../data/tours.csv")
train = pd.read_csv("../../data/train.csv")
test = pd.read_csv("../../data/test.csv")
locations = pd.read_csv("../../data/locations.csv")

#Merge biker and tour in train and test to extract unique values
data = pd.concat([train[['biker_id','tour_id']],test[['biker_id','tour_id']]],axis=0)

uniquebikers=set(data['biker_id'])
uniquetours=set(data['tour_id'])

#Build biker-tours and tour-bikers collection
data=data.reset_index(drop=True)
toursForbiker=defaultdict(set)
bikersFortour=defaultdict(set)
for i in range(len(data)):
    toursForbiker[data['biker_id'][i]].add(data['tour_id'][i])
    bikersFortour[data['tour_id'][i]].add(data['biker_id'][i])


# Import bikers data
df_bikers=pd.read_csv(r'../../data/bikers.csv')

#biker matrix preprocessing
le = LabelEncoder()

df_bikers['gender']=df_bikers['gender'].fillna('NaN')
df_bikers['gender']=le.fit_transform(df_bikers['gender'])
df_bikers['location_id']=le.fit_transform(df_bikers['location_id'])
df_bikers['language_id']=le.fit_transform(df_bikers['language_id'])

def bornInInt(bornIn):
    try:
        return np.nan if bornIn=='None' else int(bornIn)
    except:
        return np.nan

df_bikers['bornIn']=df_bikers['bornIn'].map(bornInInt)

def timezoneInt(timezone):
    try:
        return int(timezone)
    except:
        return np.nan

df_bikers['time_zone']=df_bikers['time_zone'].map(timezoneInt)

df_tours = pd.read_csv(r'../../data/tours.csv')

# we only consider the tours present in train/test files
df_tours1 = df_tours[df_tours.tour_id.isin(uniquetours)]
df_tours1 = df_tours1.reset_index(drop = True)

cluster_df = df_tours1[df_tours1.columns[9:109]]
word_counts = cluster_df

#clustering and labelling of tours based on description
from sklearn.cluster import KMeans

label = KMeans(n_clusters = 30, max_iter = 4000, random_state = 0).fit_predict(cluster_df)
dft = pd.DataFrame(label, columns = ['labels'])
df_tours2 = pd.concat([df_tours1, dft], axis = 1)

# tfidf for text description similarity
# (not used as a feature later)
from sklearn.feature_extraction.text import TfidfTransformer  

tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(word_counts)
tf_idf_vector=tfidf_transformer.transform(word_counts)
tfidf = pd.DataFrame(tf_idf_vector.todense())

# PCA on tfidf transformed word_counts
from sklearn.decomposition import PCA

pca = PCA(n_components = 10)
pca_words = pca.fit_transform(tfidf)
pca_w = pd.DataFrame(pca_words)
pca_w.columns = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10']
df_toursw = pd.concat([df_tours2, pca_w], axis = 1)

# Analysing tour_convoy
# only considering tours in train/test
tour_convoy1 = tour_convoy[tour_convoy.tour_id.isin(uniquetours)].reset_index(drop=True)

# numner of people yes/no/maybe/invited for a tour
ntours = len(uniquetours)
tourPopY = np.zeros(ntours)
tourPopN = np.zeros(ntours)
tourPopM = np.zeros(ntours)
tourPopI = np.zeros(ntours)

# Set of bikers for each category - going/ not going/ maybe/ invited
goingTour = defaultdict(set)
ngoingTour = defaultdict(set)
maybeTour = defaultdict(set)
inviteTour = defaultdict(set)

#tours a biker is going/not going/maybe/invited
bikerGoing = defaultdict(set)
bikerNGoing = defaultdict(set)
bikerMaybe = defaultdict(set)
bikerInvite = defaultdict(set)

# computing tour popularity and adding to features
for i in range(ntours):
    tourId=tour_convoy1['tour_id'][i]
    goingT = str(tour_convoy1['going'][i]).split(' ')
    notgoingT = str(tour_convoy1['not_going'][i]).split(' ')
    maybeT = str(tour_convoy1['maybe'][i]).split(' ')
    inviteT = str(tour_convoy1['invited'][i]).split(' ')
    goingTour[tourId] = set(goingT)
    ngoingTour[tourId] = set(notgoingT)
    maybeTour[tourId] = set(maybeT)
    inviteTour[tourId] = set(inviteT)
    if str(tour_convoy1['going'][i])=='nan':
        len_y=0
    else:
        len_y=len(tour_convoy1['going'][i].split(' '))
    if str(tour_convoy1['not_going'][i])=='nan':
        len_n=0
    else:
        len_n=len(tour_convoy1['not_going'][i].split(' '))
    if str(tour_convoy1['maybe'][i])=='nan':
        len_m=0
    else:
        len_m=len(tour_convoy1['maybe'][i].split(' '))
    if str(tour_convoy1['invited'][i])=='nan':
        len_i=0
    else:
        len_i=len(tour_convoy1['invited'][i].split(' '))
    tourPopY[i] = len_y
    tourPopN[i] = len_n
    tourPopM[i] = len_m
    tourPopI[i] = len_i
    for bikid in goingT:
        bikerGoing[bikid].add(tourId)
    for bikid in notgoingT:
        bikerNGoing[bikid].add(tourId)
    for bikid in maybeT:
        bikerMaybe[bikid].add(tourId)
    for bikid in inviteT:
        bikerInvite[bikid].add(tourId)

tour_convoy1['tourPopY'] = pd.Series(tourPopY)
tour_convoy1['tourPopN'] = pd.Series(tourPopN)
tour_convoy1['tourPopM'] = pd.Series(tourPopM)
tour_convoy1['tourPopI'] = pd.Series(tourPopI)

df_tours3 = df_toursw.merge(tour_convoy1[['tour_id', 'tourPopY', 'tourPopN', 'tourPopM', 'tourPopI']], on = 'tour_id')
df_tours3.rename(columns = {'biker_id': 'organiser'}, inplace = True)


# Analysing bikers_network features and adding number of friends, and set of a bikers friends
friends = defaultdict(set)

biker_net1 = bikers_network[bikers_network.biker_id.isin(uniquebikers)].reset_index(drop=True)
numfriends = np.zeros(len(biker_net1))

for i in range(len(biker_net1)):
    bikerId = biker_net1['biker_id'][i]
    friend_list = biker_net1['friends'][i].split(' ')
    friends[bikerId] = set(friend_list)
    numfriends[i] = len(friend_list)

nf = pd.DataFrame(numfriends, columns = ['num_friends'])
bikernet = pd.concat([biker_net1, nf], axis = 1)


#######################################################
# BUILDING THE FEATURE MATRIX FOR THE TRAIN/TEST FILES#
#######################################################
y_train = train[['like']]
x_train = train[['biker_id', 'tour_id', 'invited', 'timestamp']]
x_test = test[['biker_id', 'tour_id', 'invited', 'timestamp']]
x_all = pd.concat([x_train.assign(is_train=1),x_test.assign(is_train=0)],axis=0).reset_index(drop = True)

# label corresponding to cluster number, tour similarity
def addlabel(x):  
    labeller = df_tours3[['tour_id', 'labels']]
    return pd.merge(x, labeller, on = 'tour_id', how = 'left')

# tour popularity features - number going/ not going/ maybe/ invited
def addtourPop(x):
    tourpops = df_tours3[['tour_id', 'tourPopY', 'tourPopN', 'tourPopM', 'tourPopI']]
    return pd.merge(x, tourpops, on = 'tour_id', how = 'left')

# number of friends of a biker
def addnum_friends(x):
    frens = bikernet[['biker_id', 'num_friends']]
    return pd.merge(x, frens, on = 'biker_id', how = 'left')

# basic information about biker
def addbikerinfo(x):
    biker_info = df_bikers[['biker_id', 'language_id', 'location_id', 'gender', 'member_since', 'area','bornIn', 'time_zone']]
    return pd.merge(x, biker_info, on = 'biker_id', how = 'left')

# age of biker
def addage(x):
    age = []
    for i in range(len(x)):
        if np.isnan(x['bornIn'][i]):
            age.append(np.nan)
        else:
            yearb = x['bornIn'][i]
            age.append(2013 - yearb)
    x['age'] = pd.Series(age)
    return x

#info about tours
def addtourinfo(x):
    tour_info = df_tours3[['tour_id', 'organiser', 'tour_date', 'city', 'state', 'country', 'latitude', 'longitude',
                          'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10']]
    #tour_info = df_tours3[df_tours3.columns[0:109]]
    tour_info.rename(columns = {'latitude': 'tour_lat', 'longitude': 'tour_lng'}, inplace = True)
    return pd.merge(x, tour_info, on = 'tour_id', how = 'left')

# number of tours shown to a biker in train/test file
def addtourForbiker(x):
    ntoursbiker = []
    for i in range(len(x)):
        bikid = x['biker_id'][i]
        ntoursbiker.append(len(toursForbiker[bikid]))
    x['ntoursShown'] = pd.Series(ntoursbiker)
    return x

# number of tours a biker is going/not going/ maybe/ invited
def addbikerTours(x):
    bikGoing, bikNGoing, bikMaybe, bikInvite = [], [], [], []
    for i in range(len(x)):
        bikGoing.append(len(bikerGoing[x['biker_id'][i]]))
        bikNGoing.append(len(bikerNGoing[x['biker_id'][i]]))
        bikMaybe.append(len(bikerMaybe[x['biker_id'][i]]))
        bikInvite.append(len(bikerInvite[x['biker_id'][i]]))
    x['bik_G'] = pd.Series(bikGoing)
    x['bik_NG'] = pd.Series(bikNGoing)
    x['bik_M'] = pd.Series(bikMaybe)
    x['bik_I'] = pd.Series(bikInvite)
    return x

# number of days between (i) tour_date and member_since
                        #(ii) timestamp and member_since
                        #(iii) tour_date and timestamp (time to tour)
def adddateDiffs(x):
    diff1, diff2, diff3 = [], [], []
    for i in range(len(x)):
        t1 = datetime.strptime(x['member_since'][i], "%d-%m-%Y")
        t2 = datetime.strptime(x['tour_date'][i], "%d-%m-%Y")
        t3 = datetime.strptime(x['timestamp'][i], "%d-%m-%Y %H:%M:%S")
        diff1.append((t2-t1).days + 1/86400 * (t2-t1).seconds)
        diff2.append((t3-t1).days + 1/86400 * (t3-t1).seconds)
        diff3.append((t2-t3).days + 1/86400 * (t2-t3).seconds)
    x['diff1'] = pd.Series(diff1)
    x['diff2'] = pd.Series(diff2)
    x['diff3'] = pd.Series(diff3)
    return x

# is organiser a friend
def addorganiserFren(x):
    list1 = []
    for i in range(len(x)):
        organiserId = x['organiser'][i]
        bikId = x['biker_id'][i]
        if organiserId in friends[bikId]:
            list1.append(1)
        else:
            list1.append(0)
    x['isOrganiserFrnd'] = pd.Series(list1)
    return x

# number of friends of biker going/ not going/ maybe/ invited
def addfriendsInfl(x):
    freninfl1 = []
    freninfl2 = []
    freninfl3 = []
    freninfl4 = []
    for i in range(len(x)):
        bikid = x['biker_id'][i]
        tourid = x['tour_id'][i]
        t1 = len(friends[bikid].intersection(goingTour[tourid]))
        t2 = len(friends[bikid].intersection(ngoingTour[tourid]))
        t3 = len(friends[bikid].intersection(maybeTour[tourid]))
        t4 = len(friends[bikid].intersection(inviteTour[tourid]))
        t5 = len(friends[bikid])
        freninfl1.append(t1)
        freninfl2.append(t2)
        freninfl3.append(t3)
        freninfl4.append(t4)
    x['friendsInfl1'] = pd.Series(freninfl1)
    x['friendsInfl2'] = pd.Series(freninfl2)
    x['friendsInfl3'] = pd.Series(freninfl3)
    x['friendsInfl4'] = pd.Series(freninfl4)
    return x

# biker location from locations.csv
def addbikLocation(x):
    biklocation = locations[['biker_id', 'latitude', 'longitude']]
    biklocation.rename(columns = {'latitude': 'bik_lat', 'longitude': 'bik_lng'}, inplace = True)
    return pd.merge(x, biklocation, on = 'biker_id', how = 'left')
   
# Importing the geodesic module from the library to get tour-biker distance
from geopy.distance import geodesic as geodis

def adddistance(x):
    dist = []
    for i in range(len(x)):
        if np.isnan(x['bik_lat'][i]):
            dist.append(np.nan)
        elif np.isnan(x['tour_lat'][i]):
            dist.append(np.nan)
        else:
            bikloc = (x['bik_lat'][i], x['bik_lng'][i])
            tourloc = (x['tour_lat'][i], x['tour_lng'][i])
            dist.append(geodis(bikloc, tourloc).km)
    x['distance'] = pd.Series(dist)
    return x
            

x_all = addlabel(x_all)
x_all = addtourPop(x_all)
x_all = addnum_friends(x_all)
x_all = addbikerinfo(x_all)
x_all = addage(x_all)
x_all = addtourinfo(x_all)
x_all = addbikerTours(x_all)
x_all = addtourForbiker(x_all)
x_all = adddateDiffs(x_all)
x_all = addorganiserFren(x_all)
x_all = addfriendsInfl(x_all)
x_all = addbikLocation(x_all)
x_all = adddistance(x_all)

cols = ['biker_id', 'tour_id', 'invited', 'labels', 'tourPopY', 'tourPopN', 'tourPopM', 'tourPopI',
        'num_friends', 'language_id', 'location_id', 'gender','bornIn', 'age',
        'tour_lat', 'tour_lng', 'ntoursShown', 'diff1', 'diff2', 'diff3', 'isOrganiserFrnd', 
        'bik_G', 'bik_NG', 'bik_M', 'bik_I',
        'friendsInfl1', 'friendsInfl2', 'friendsInfl3', 'friendsInfl4', 'bik_lat', 'bik_lng', 'distance', 
        'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10',
        ]

# Separating back into train and test files
x_train = x_all[x_all['is_train']==1][cols]
x_test = x_all[x_all['is_train']==0][cols]
#print(x_train.info())
#print(y_train.info())
y_train1 = y_train


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

#Feature selection
ocols = ['invited', 'labels', 'tourPopY', 'tourPopN', 'tourPopM', 
         'tourPopI', 'num_friends','language_id', 'location_id','gender', 
         'bornIn', 'tour_lat', 'tour_lng','ntoursShown', 'diff1', 'diff2', 'diff3','isOrganiserFrnd', 
         'friendsInfl1', 'friendsInfl2', 'friendsInfl3', 'friendsInfl4',
         #'bik_G', 'bik_NG', 'bik_M', 'bik_I',
         #'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10',
         'bik_lat' , 'bik_lng', 'distance',
        ]  

X = x_train[ocols].to_numpy()
y = y_train1.to_numpy().ravel()

#Split training data
from sklearn.model_selection import train_test_split
Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size=0.3,random_state=0)
#import seaborn as sns
#import matplotlib.pyplot as plt
#corrf = pd.DataFrame(X).corr()
#f, ax = plt.subplots(figsize=(12, 9))
#sns.heatmap(corrf, vmax=.8, square=True);

#Optimal parameters
#print(opt_params)

import lightgbm as lgb

# defining the classifier and checking on train-test split 
lgc = lgb.LGBMClassifier(boosting_type='gbdt',  num_leaves = 58, 
                         max_depth=12, learning_rate=0.09,reg_lambda = 1.0, 
                         n_estimators=150, feature_fraction = 0.6740, seed=0 
                        )
clf_lg = lgc.fit(Xtrain,ytrain)

#print(clf_lg.score(Xtrain, ytrain))
#print(clf_lg.score(Xtest, ytest))
#print(clf_lg)
#print((clf_lg.predict_proba(Xtest)))
#lgb.plot_importance(clf_lg)

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold


# TRAINING the classifier on all of the data
def trainfun(clf):
    trainDf = x_train
    X = pd.DataFrame(trainDf, index=None, columns=ocols)
    y = np.array(y_train1).ravel()
    
    clf.fit(X, y)
    return clf
 
# k-fold validation
def validatefun(clf):   
    trainDf = x_train[ocols]
    X = np.matrix(pd.DataFrame(trainDf, index=None, columns=ocols))
    y = np.array(y_train1).ravel()
     
    nrows = len(trainDf)
    kfold = KFold(n_splits=10,shuffle=False)
    avgAccuracy = 0
    run = 0
    for train, test in kfold.split(X, y):
        Xtrain, Xtest, ytrain, ytest = X[train], X[test], y[train], y[test]
        
        clf.fit(Xtrain, ytrain)
        accuracy = 0
        ntest = len(ytest)
        for i in range(0, ntest):
            yt = clf.predict(Xtest[i, :])
            if yt == ytest[i]:
                accuracy += 1
                 
        accuracy = accuracy / ntest
        run += 1
        #print('accuracy(run %d) : %f' % (run, accuracy) )

clf1 = lgc
trainfun(clf1)
validatefun(clf1)

import warnings
warnings.filterwarnings('ignore')

#creating the submission file with the trained classifier
def testfun(clf, filename):
    testDf = np.matrix(pd.DataFrame(x_test, index=None, columns=ocols))
     
    nrows = len(testDf)
    donebik = []
    tlist = []
    for i in range(nrows):
        bikid = test['biker_id'][i]
        if bikid in donebik:
            continue
        list1 = []
        for j in range(i, nrows):
            if test['biker_id'][j] == bikid :
                tourid = test['tour_id'][j]
                pred = clf.predict_proba(testDf[j, :])
                list1.append([tourid, pred[0][0]])
        list1.sort(key = lambda x: x[1])
        donebik.append(bikid)
        tids = np.array(list1)
        #print(tids[0, -1])
        tids = " ".join(tids[:,0])
        tlist.append(tids)
        
    sample_submission =pd.DataFrame(columns=["biker_id","tour_id"])
    sample_submission["biker_id"] = donebik
    sample_submission["tour_id"] = tlist
    # Change name of submission if needed
    sample_submission.to_csv(filename,index=False)
    #print(sample_submission.shape)
    #print(sample_submission.head(4))

testfun(clf1, "CS18B057_CS18B046_1.csv")
##############################
# 1ST SUBMISSION FILE MADE   #
##############################

cols = ['biker_id', 'tour_id', 'invited', 'labels', 'tourPopY', 'tourPopN', 'tourPopM', 'tourPopI',
        'num_friends', 'language_id', 'location_id', 'gender','bornIn', 'age',
        'tour_lat', 'tour_lng', 'ntoursShown', 'diff1', 'diff2', 'diff3', 'isOrganiserFrnd', 
        'bik_G', 'bik_NG', 'bik_M', 'bik_I',
        'friendsInfl1', 'friendsInfl2', 'friendsInfl3', 'friendsInfl4', 'bik_lat', 'bik_lng', 'distance', 
        'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10',
        ]

x_train = x_all[x_all['is_train']==1][cols]
x_test = x_all[x_all['is_train']==0][cols]
#print(x_train.info())
#print(y_train.info())


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

#Feature selection
ocols = ['invited', 'labels', 'tourPopY', 'tourPopN', 'tourPopM', 
         'tourPopI', 'num_friends','language_id', 'location_id','gender', 
         'bornIn', 'tour_lat', 'tour_lng','ntoursShown', 'diff1', 'diff2', 'diff3','isOrganiserFrnd', 
         'friendsInfl1', 'friendsInfl2', 'friendsInfl3', 'friendsInfl4',
         'bik_G', 'bik_NG', 'bik_M', 'bik_I',
         #'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10',
         'bik_lat' , 'bik_lng', 'distance',
        ]  

X = x_train[ocols].to_numpy()
y = y_train1.to_numpy().ravel()

#Split training data
from sklearn.model_selection import train_test_split
Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size=0.3,random_state=0)

#print(opt_params)
lgc = lgb.LGBMClassifier(boosting_type='gbdt',  num_leaves = 43, 
                         max_depth=11, learning_rate=0.074,reg_lambda = 0.457,
                         n_estimators=100, feature_fraction = 0.844, seed=0 
                        )
clf_lg = lgc.fit(Xtrain,ytrain)

#print(clf_lg.score(Xtrain, ytrain))
#print(clf_lg.score(Xtest, ytest))
#print(clf_lg)
#lgb.plot_importance(clf_lg)

clf2 = lgc
trainfun(clf2)
validatefun(clf2)

testfun(clf2, "CS18B057_CS18B046_2.csv")
##############################
# 2ND SUBMISSION FILE MADE   #
##############################
####             #####             #########
#        ####              ######

##############################
#    Parameter Tuning code   #
##############################
'''from bayes_opt import BayesianOptimization
from skopt  import BayesSearchCV 
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
import time
import sys

#metrics 
from sklearn.metrics import roc_auc_score, roc_curve
import shap
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def bayes_parameter_opt_lgb(X, y, init_round=15, opt_round=25, n_folds=3, random_seed=6,n_estimators=200, output_process=False):
    # prepare data
    train_data = lgb.Dataset(data=X, label=y, free_raw_data=False)
    # parameters
    def lgb_eval(learning_rate,num_leaves, feature_fraction, #bagging_fraction, 
                 max_depth, reg_lambda):#, max_bin, min_data_in_leaf,min_sum_hessian_in_leaf,subsample):
        params = {'objective':'binary', 'metric':'auc'}
        params['learning_rate'] = max(min(learning_rate, 1), 0)
        params['num_leaves'] = int(round(num_leaves))
        params['feature_fraction'] = max(min(feature_fraction, 1), 0)
        #params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
        params['max_depth'] = int(round(max_depth))
        #params['max_bin'] = int(round(max_depth))
        #params['min_data_in_leaf'] = int(round(min_data_in_leaf))
        #params['min_sum_hessian_in_leaf'] = min_sum_hessian_in_leaf
        #params['subsample'] = max(min(subsample, 1), 0)
        params['reg_lambda'] = max(min(feature_fraction, 10), 0)
        
        cv_result = lgb.cv(params, train_data, nfold=n_folds, seed=random_seed, stratified=True, verbose_eval =200, metrics=['auc'])
        return max(cv_result['auc-mean'])
     
    lgbBO = BayesianOptimization(lgb_eval, {'learning_rate': (0.01, 0.09),
                                            'num_leaves': (16, 64),
                                            'feature_fraction': (0.3, 0.9),
                                            #'bagging_fraction': (0.8, 1),
                                            'max_depth': (5, 12),
                                            #'max_bin':(10,90),
                                            #'min_data_in_leaf': (10, 80),
                                            #'min_sum_hessian_in_leaf':(0,100),
                                           #'subsample': (0.01, 1.0)},
                                            'reg_lambda': (0.1, 1.0),
                                            
                                           },random_state=200)

    
    #n_iter: How many steps of bayesian optimization you want to perform. The more steps the more likely to find a good maximum you are.
    #init_points: How many steps of random exploration you want to perform. Random exploration can help by diversifying the exploration space.
    
    lgbBO.maximize(init_points=init_round, n_iter=opt_round)
    
    model_auc=[]
    for model in range(len( lgbBO.res)):
        model_auc.append(lgbBO.res[model]['target'])
    
    # return best parameters
    return lgbBO.res[pd.Series(model_auc).idxmax()]['target'],lgbBO.res[pd.Series(model_auc).idxmax()]['params']

#opt_params = bayes_parameter_opt_lgb(X, y, init_round=15, opt_round=25, n_folds=10, random_seed=6,n_estimators=150)
#print(opt_params)
'''