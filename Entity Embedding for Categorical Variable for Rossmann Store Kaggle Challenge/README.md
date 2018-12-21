This is the code used in the paper **"[Entity Embeddings of Categorical Variables](http://arxiv.org/abs/1604.06737)"**. If you want to get the original version of the code used for the Kaggle competition, please use [**the Kaggle branch**](https://github.com/entron/entity-embedding-rossmann/tree/kaggle).

To run the code one needs first download and unzip the `train.csv` and `store.csv` files on [Kaggle](https://www.kaggle.com/c/rossmann-store-sales/data) and put them in this folder.

The following packages are needed if you want to recover the result in the paper (we used python 3):

```
pip3 install -U scikit-learn
pip3 install -U xgboost
pip3 install keras==1.2.2
```
Please refer to [Keras](https://github.com/fchollet/keras) for more details regarding how to install keras. Note that the code used keras 1.x API so make sure to install the right version of keras as shown above.

Next, run the following scripts to extract the csv files and prepare the features:

```
python3 extract_csv_files.py
python3 prepare_features.py
```

To run the models:

```
python3 train_test_model.py
```

You can anaylize the embeddings with the ipython notebook included. This is the learned embeeding of German States printed in 2D (with the Kaggle branch):

[![](https://plot.ly/~entron/0/.png)](https://plot.ly/~entron/0.embed)

and this is the learned embeddings of 1115 Rossmann stores printed in 3D:

[![](https://plot.ly/~entron/2/.png)](https://plot.ly/~entron/2.embed)

The Data fields in `train.csv` data are
```
You are provided with historical sales data for 1,115 Rossmann stores. The task is to forecast the "Sales" column for the test set. Note that some stores in the dataset were temporarily closed for refurbishment.

Files
train.csv - historical data including Sales
test.csv - historical data excluding Sales
sample_submission.csv - a sample submission file in the correct format
store.csv - supplemental information about the stores
Data fields
Most of the fields are self-explanatory. The following are descriptions for those that aren't.

Id - an Id that represents a (Store, Date) duple within the test set
Store - a unique Id for each store
Sales - the turnover for any given day (this is what you are predicting)
Customers - the number of customers on a given day
Open - an indicator for whether the store was open: 0 = closed, 1 = open
StateHoliday - indicates a state holiday. Normally all stores, with few exceptions, are closed on state holidays. Note that all schools are closed on public holidays and weekends. a = public holiday, b = Easter holiday, c = Christmas, 0 = None
SchoolHoliday - indicates if the (Store, Date) was affected by the closure of public schools
StoreType - differentiates between 4 different store models: a, b, c, d
Assortment - describes an assortment level: a = basic, b = extra, c = extended
CompetitionDistance - distance in meters to the nearest competitor store
CompetitionOpenSince[Month/Year] - gives the approximate year and month of the time the nearest competitor was opened
Promo - indicates whether a store is running a promo on that day
Promo2 - Promo2 is a continuing and consecutive promotion for some stores: 0 = store is not participating, 1 = store is participating
Promo2Since[Year/Week] - describes the year and calendar week when the store started participating in Promo2
PromoInterval - describes the consecutive intervals Promo2 is started, naming the months the promotion is started anew. E.g. "Feb,May,Aug,Nov" means each round starts in February, May, August, November of any given year for that store
```
