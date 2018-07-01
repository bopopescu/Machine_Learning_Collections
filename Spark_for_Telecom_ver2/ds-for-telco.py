

# # Load the data
# 
# We need to load data from a file in to a Spark DataFrame.
# Each row is an observed customer, and each column contains
# attributes of that customer.
# 
# [Data from UCI data set repo, hosted by SGI](https://www.sgi.com/tech/mlc/db/churn.all)
# 
#     Fields:
# 
#     state: discrete.
#     account length: numeric.
#     area code: numeric.
#     phone number: discrete.
#     international plan: discrete.
#     voice mail plan: discrete.
#     number vmail messages: numeric.
#     total day minutes: numeric.
#     total day calls: numeric.
#     total day charge: numeric.
#     total eve minutes: numeric.
#     total eve calls: numeric.
#     total eve charge: numeric.
#     total night minutes: numeric.
#     total night calls: numeric.
#     total night charge: numeric.
#     total intl minutes: numeric.
#     total intl calls: numeric.
#     total intl charge: numeric.
#     number customer service calls: numeric.
#     churned: discrete.
# 
# 'Numeric' and 'discrete' do not adequately describe the fundamental differences in the attributes.
# 
# Area codes are considered numeric, but they a better thought of as a categorical variable. This is because attributes that are really numeric features have a reasonable concept of distance between points. Area codes do not fall into this cateogory. They can have a small distance |area_code_1 - area_code_2| but that distance doesn't correspond to a similarity in area codes.

from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.sql.types import *


conf = SparkConf().setAppName("ds-for-telco")
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)
schema = StructType([     StructField("state", StringType(), True),     StructField("account_length", DoubleType(), True),     StructField("area_code", StringType(), True),     StructField("phone_number", StringType(), True),     StructField("intl_plan", StringType(), True),     StructField("voice_mail_plan", StringType(), True),     StructField("number_vmail_messages", DoubleType(), True),     StructField("total_day_minutes", DoubleType(), True),     StructField("total_day_calls", DoubleType(), True),     StructField("total_day_charge", DoubleType(), True),     StructField("total_eve_minutes", DoubleType(), True),     StructField("total_eve_calls", DoubleType(), True),     StructField("total_eve_charge", DoubleType(), True),     StructField("total_night_minutes", DoubleType(), True),     StructField("total_night_calls", DoubleType(), True),     StructField("total_night_charge", DoubleType(), True),     StructField("total_intl_minutes", DoubleType(), True),     StructField("total_intl_calls", DoubleType(), True),     StructField("total_intl_charge", DoubleType(), True),     StructField("number_customer_service_calls", DoubleType(), True),     StructField("churned", StringType(), True)])

churn_data = sqlContext.read     .format('com.databricks.spark.csv')     .load('/tmp/churn.all', schema = schema)


# # Basic DataFrame operations
# 
# Dataframes essentially allow you to express sql-like statements. We can filter, count, and so on. [DataFrame Operations documentation.](http://spark.apache.org/docs/latest/sql-programming-guide.html#dataframe-operations)

count = churn_data.count()
voice_mail_plans = churn_data.filter(churn_data.voice_mail_plan == " yes").count()
churned_customers = churn_data.filter(churn_data.churned == " True.").count()

"total: %d, voice mail plans: %d, churned customers: %d " % (count, voice_mail_plans, churned_customers)


# How many customers have  more than one service call? 

service_calls = churn_data.filter(churn_data.number_customer_service_calls > 1).count()
service_calls2 = churn_data.filter(churn_data.number_customer_service_calls > 2).count()

"customers with more than 1 service call: %d, 2 service calls: %d" % (service_calls, service_calls2)


# # Feature Visualization
# 
# The data vizualization workflow for large data sets is usually:
# 
# * Sample data so it fits in memory on a single machine.
# * Examine single variable distributions.
# * Examine joint distributions and correlations.
# * Look for other types of relationships.
# 
# [DataFrame#sample() documentation](http://people.apache.org/~pwendell/spark-releases/spark-1.5.0-rc1-docs/api/python/pyspark.sql.html#pyspark.sql.DataFrame.sample)

sample_data = churn_data.sample(False, 0.5, 83).toPandas()
sample_data.transpose().head(21)


# # DataTypes
# 
import urllib
from IPython.display import Image
Image(filename="slides/datatypes.png")
# datatypes where continuous and integer are subtypes of numeric data and distinct from the categorical type(slides/datatypes.png)

# The type of visualization we do depends on the data type, so lets define what columns have different properties first:


numeric_cols = ["account_length", "number_vmail_messages", "total_day_minutes",
                "total_day_calls", "total_day_charge", "total_eve_minutes",
                "total_eve_calls", "total_eve_charge", "total_night_minutes",
                "total_night_calls", "total_night_charge", "total_intl_minutes", 
                "total_intl_calls", "total_intl_charge","number_customer_service_calls"]

categorical_cols = ["state", "international_plan", "voice_mail_plan", "area_code"]



get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sb

sb.distplot(sample_data['number_customer_service_calls'], kde=False)


# We can examine feature differences in the distribution of our features when we condition (split) our data in whether they churned or not.
# 
# [BoxPlot docs](http://stanford.edu/~mwaskom/software/seaborn/generated/seaborn.boxplot.html)


sb.boxplot(x="churned", y="number_customer_service_calls", data=sample_data)


# # Joint Distributions
# 
# Looking at joint distributions of data can also tell us a lot, particularly about redundant features. [Seaborn's PairPlot](http://stanford.edu/~mwaskom/software/seaborn/generated/seaborn.pairplot.html#seaborn.pairplot) let's us look at joint distributions for many variables at once.


example_numeric_data = sample_data[["total_intl_minutes", "total_intl_calls",
                                       "total_intl_charge", "churned"]]
sb.pairplot(example_numeric_data, hue="churned")


# Clearly, there are some strong linear relationships between some variables, let's get a general impression of the correlations between variables by using [Seaborn's heatmap functionality.](http://stanford.edu/~mwaskom/software/seaborn/generated/seaborn.heatmap.html)


corr = sample_data[["account_length", "number_vmail_messages", "total_day_minutes",
                    "total_day_calls", "total_day_charge", "total_eve_minutes",
                    "total_eve_calls", "total_eve_charge", "total_night_minutes",
                    "total_night_calls", "total_night_charge", "total_intl_minutes", 
                    "total_intl_calls", "total_intl_charge","number_customer_service_calls"]].corr()

sb.heatmap(corr)


# The heatmap shows which features we can eliminate due to high correlation. We then get the following:


reduced_numeric_cols = ["account_length", "number_vmail_messages", "total_day_calls",
                        "total_day_charge", "total_eve_calls", "total_eve_charge",
                        "total_night_calls", "total_night_charge", "total_intl_calls", 
                        "total_intl_charge","number_customer_service_calls"]


# # Build a classification model using MLLib
# 
# We want to build a predictive model.
# 
# The overall process:
# 
Image(filename="slides/model flow.png")
# Model building diagraming showing input data split into train and test data. The training data is aligned with a feature extraction and model training set, which outputs a fitter model. The testing data alignes with a feature extraction, scoring/model application, and model evaluation step

# # Feature Extraction and Model Training
# 
# We need to:
# * Code features that are not already numeric
# * Gather all features we need into a single column in the DataFrame.
# * Split labeled data into training and testing set
# * Fit the model to the training data.
# 
# ## Feature Extraction
# We need to define our input features.
# 
# [PySpark Pipeline Docs](https://spark.apache.org/docs/1.5.0/api/python/pyspark.ml.html)
# 
# ## Note
# StringIndexer --> Turns string value into numerical value. I.E. below we use it to map "churned" ("yes" or "no") into numerical values and put in it colum "label".
# 
# VectorAssembler --> takes input columns and returns a vector object into output column


from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler

label_indexer = StringIndexer(inputCol = 'churned', outputCol = 'label')
plan_indexer = StringIndexer(inputCol = 'intl_plan', outputCol = 'intl_plan_indexed')

assembler = VectorAssembler(
    inputCols = ['intl_plan_indexed'] + reduced_numeric_cols,
    outputCol = 'features')


# # Model Training
# 
# We can now define our classifier and pipeline. With this done, we can split our labeled data in train and test sets and fit a model.
# 
# To train the decision tree, give it the feature vector column and the label column. 
# 
# Pipeline is defined by stages. Index plan column, label column, create vectors, then define the decision tree. 


from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier

classifier = DecisionTreeClassifier(labelCol = 'label', featuresCol = 'features')

pipeline = Pipeline(stages=[plan_indexer, label_indexer, assembler, classifier])

(train, test) = churn_data.randomSplit([0.7, 0.3])
model = pipeline.fit(train)


# ## Model Evaluation
# 
# The most important question to ask:
#     
#     Is my predictor better than random guessing?
# 
# How do we quantify that?

# Measure the area under the ROC curve, abreviated to AUROC.
# 
# Plots True Positive Rate vs False Positive Rate for binary classification system
# 
# [More Info](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)
# 
# TL;DR for AUROC:
#     * .90-1 = excellent (A)
#     * .80-.90 = good (B)
#     * .70-.80 = fair (C)
#     * .60-.70 = poor (D)
#     * .50-.60 = fail (F)
# 
Image(filename="slides/roc_curve.png")

# example of an roc curve


from pyspark.ml.evaluation import BinaryClassificationEvaluator

predictions = model.transform(test)
evaluator = BinaryClassificationEvaluator()
auroc = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})
"The AUROC is %s " % (auroc)


# # Fit a RandomForestClassifier
# 
# Fit a random forest classifier to the data. Try experimenting with different values of the `maxDepth`, `numTrees`, and `entropy` parameters to see which gives the best classification performance. Do the settings that give the best classification performance on the training set also give the best classification performance on the test set?
# 
# Have a look at the [documentation](http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.classification.RandomForestClassifier]documentation).


from pyspark.ml.classification import RandomForestClassifier
 
classifier = RandomForestClassifier(labelCol = 'label', featuresCol = 'features', numTrees= 10)
pipeline = Pipeline(stages=[plan_indexer, label_indexer, assembler, classifier])
model = pipeline.fit(train)

predictions = model.transform(test)
evaluator = BinaryClassificationEvaluator()
auroc = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})
"The AUROC is %s " % (auroc)


# If you need to inspect the predictions...

#predictions.select('label','prediction','probability').filter('prediction = 1').show(50)
predictions.select('label','prediction','probability').filter('prediction = 1').toPandas().head(50)

