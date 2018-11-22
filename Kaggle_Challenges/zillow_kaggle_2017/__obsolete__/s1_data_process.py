# -*- coding: utf-8 -*-
"""
2017 Zillow Kaggle Challenge

https://www.kaggle.com/c/zillow-prize-1

@author: MP
"""

import sys, os
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import kendalltau
import warnings
import scipy.stats as st
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot
import pylab
matplotlib.style.use('ggplot')


# Set global parameters
data_dir = os.path.join(os.getcwd(), "./source_data")
propertcsv = "properties_2016.csv"
traincsv = "train_2016_v2.csv"
submissioncsv = "sample_submission.csv"


# Load Data and EDA
property_df = pd.read_csv(os.path.join(data_dir, propertcsv))
train_df = pd.read_csv(os.path.join(data_dir, traincsv))
submission_df = pd.read_csv(os.path.join(data_dir, submissioncsv))

train = pd.merge(train_df, property_df, on="parcelid", how="left")


if __name__ == "__main__":
    ################################################################
    print("Data EDA and Visualization")
    ################################################################
    # Data checks
    train_df.head()
    property_df.head()
    submission_df.head()
    print ("Shape Of Train: ",train_df.shape)
    print ("Shape Of Properties: ",property_df.shape)
    train.head(3).transpose()  # check the first few records in rotated view

    # Visualize the data
    dataTypeDf = pd.DataFrame(train.dtypes.value_counts()).reset_index().rename(columns={"index":"variableType",0:"count"})
    fig,ax = plt.subplots()
    fig.set_size_inches(20,5)
    sns.barplot(data=dataTypeDf,x="variableType",y="count",ax=ax,color="#34495e")
    ax.set(xlabel='Variable Type', ylabel='Count',title="Variables Count Across Datatype")
    sns.plt.show()

    # Check the missing data
    missingValueColumns = train.columns[train.isnull().any()].tolist()
    miss_ = pd.DataFrame(train[missingValueColumns].apply(lambda x: np.sum(x.isnull()), axis=0))
    miss_ = pd.DataFrame({"column" : miss_.index, "missing_counts": miss_[0].values})
    ax = sns.barplot(x = "column", y = "missing_counts", data = miss_, estimator=np.mean)
    sns.plt.show()

    # Plot the distribution of the target value
    ulimit = np.percentile(train.logerror.values, 99)
    llimit = np.percentile(train.logerror.values, 1)
    train['logerror'].loc[train['logerror'] > ulimit] = ulimit
    train['logerror'].loc[train['logerror'] < llimit] = llimit

    plt.figure(1)
    ax = sns.distplot(train['logerror'], kde=False, fit=st.laplace)
    ax.set(xlabel='logerror', title="Distribution Test, Laplace")
    sns.plt.show()

    plt.figure(2); plt.title('Normal')
    ax = sns.distplot(train['logerror'], kde=False, fit=st.norm)
    ax.set(xlabel='logerror', title="Distribution Test, Normal")
    sns.plt.show()

    plt.figure(3); plt.title('LogNormal')
    ax = sns.distplot(train['logerror'], kde=False, fit=st.lognorm)
    ax.set(xlabel='logerror', title="Distribution Test, LogNormal")
    sns.plt.show()

    f = pd.melt(train, value_vars=["taxamount"])
    g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False)
    g = g.map(sns.distplot, "value")
    sns.plt.show()

    # Bivariate Analysis
    train["year"] = train.transactiondate.map(lambda x: str(x).split("-")[0])
    train["year"] = pd.to_datetime(train['transactiondate'], format='%Y-%m-%d').dt.year
    train["month"] = train.transactiondate.map(lambda x: str(x).split("-")[1])
    train["day"] = train.transactiondate.map(lambda x: str(x).split("-")[2].split()[0])

    traingroupedMonth = train.groupby(["month"])["logerror"].mean().to_frame().reset_index()
    traingroupedDay = train.groupby(["day"])["logerror"].mean().to_frame().reset_index()
    fig,(ax1,ax2)= plt.subplots(nrows=2)
    fig.set_size_inches(20,15)
    sns.pointplot(x=traingroupedMonth["month"], y=traingroupedMonth["logerror"], data=traingroupedMonth, join=True,ax=ax1,color="#34495e")
    ax1.set(xlabel='Month Of The Year', ylabel='Log Error',title="Average Log Error Across Month Of 2016",label='big')
    sns.countplot(x=train["month"], data=train,ax=ax2,color="#34495e")
    ax2.set(xlabel='Month Of The Year', ylabel='No Of Occurences',title="No Of Occurunces Across Month In 2016",label='big')
    sns.plt.show()

    # Check the number of stories distribution by year of built
    fig,ax1= plt.subplots()
    fig.set_size_inches(20,10)
    train["yearbuilt"] = train["yearbuilt"].map(lambda x: str(x).split(".")[0])
    yearMerged = train.groupby(['yearbuilt', 'numberofstories'])["parcelid"].count().unstack('numberofstories').fillna(0)
    yearMerged.plot(kind='bar', stacked=True,ax=ax1)

    # Set capping and flooring for the numerical variables
    cols = ["bathroomcnt","bedroomcnt","roomcnt","numberofstories","logerror","calculatedfinishedsquarefeet"]
    mergedFiltered = train[cols].dropna()
    for col in cols:
        ulimit = np.percentile(train[col].dropna().values, 99.5)
        llimit = np.percentile(train[col].dropna().values, 0.5)
        train[col].loc[train[col]>ulimit] = ulimit
        train[col].loc[train[col]<llimit] = llimit

    # 3D plot on the (Bedroom Count + Bathroom Count) vs (LogError)
    fig = pylab.figure()
    fig.set_size_inches(20,10)
    ax = Axes3D(fig)
    ax.scatter(mergedFiltered.bathroomcnt, mergedFiltered.bedroomcnt, mergedFiltered.logerror,color="#34495e")
    ax.set_xlabel('Bathroom Count')
    ax.set_ylabel('Bedroom Count')
    ax.set_zlabel('Log Error')
    pyplot.show()

    ################################################################
    print("")
    ################################################################
