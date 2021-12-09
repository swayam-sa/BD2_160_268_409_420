import json
import numpy as np
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover
from pyspark.sql import SQLContext, SparkSession
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.feature_extraction.text import HashingVectorizer
from pyspark.sql.functions import concat_ws
from pyspark.sql.types import *
from sklearn.cluster import KMeans
import joblib

sc = SparkContext("local[2]", appName="spam")
ssc = StreamingContext(sc, 1)
sql_context = SQLContext(sc)
nb = MultinomialNB()
perceptron = Perceptron()
k_means = KMeans(n_clusters=2)

my_schema = StructType([
    StructField("feature0", StringType(), True),
    StructField("feature1", StringType(), True),
    StructField("feature2", StringType(), True),
])


def RDDtoDf(x):
    spark = SparkSession(x.context)
    if not x.isEmpty():
        y = x.collect()[0]
        z = json.loads(y)
        k = z.values()
        df = spark.createDataFrame(k, schema=my_schema)
        label_encoder(df)


def label_encoder(df):
    le = LabelEncoder()
    y = le.fit_transform(np.array(df.select('feature2').collect()))
    tokenizer(df, y)


def tokenizer(df, y):
    feature1 = RegexTokenizer(inputCol="feature1", outputCol='tokens1', pattern='\\W')
    tk1 = feature1.transform(df).select('tokens1')
    stop_words_remover(tk1, y)


def stop_words_remover(col1, y):
    feature1 = StopWordsRemover(inputCol='tokens1', outputCol='filtered_words1')
    swr1 = feature1.transform(col1).select('filtered_words1')
    hash_vectoriser(swr1, y)


def hash_vectoriser(swr1, y):
    data_with_new_line = np.array(swr1.withColumn("filtered_words1", concat_ws(" ", "filtered_words1")).collect()).tolist()
    data_hv = [i[0] for i in data_with_new_line]
    hv = HashingVectorizer(alternate_sign=False)
    output_hv = hv.fit_transform(data_hv).toarray()
    x_train, x_test, y_train, y_test = train_test_split(output_hv, y, test_size=0.2)

    nb.partial_fit(x_train, y_train, classes=np.unique(y_train))
    print("nb train accuracy: %.3f" % nb.score(x_test, y_test))
    filename_nb = 'nb_model.pkl'
    joblib.dump(nb, filename_nb)

    perceptron.partial_fit(x_train, y_train, classes=np.unique(y_train))
    print("perceptron train accuracy: %.3f" % perceptron.score(x_test, y_test))
    filename_perceptron = 'perceptron_model.pkl'
    joblib.dump(perceptron, filename_perceptron)

    y_test = k_means.fit_predict(x_test)
    print("K-means train accuracy: %.3f" % k_means.score(x_test, y_test))
    filename_k_means = 'k_means_model.pkl'
    joblib.dump(k_means, filename_k_means)


records = ssc.socketTextStream("localhost", 6100)
if records:
    records.foreachRDD(RDDtoDf)
ssc.start()
ssc.awaitTermination()
ssc.stop()
