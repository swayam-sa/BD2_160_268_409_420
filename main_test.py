import json
import numpy as np
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover
from pyspark.sql import SQLContext, SparkSession
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import HashingVectorizer
from pyspark.sql.functions import concat_ws
from sklearn.metrics import accuracy_score
from pyspark.sql.types import *
import joblib

sc = SparkContext("local[2]", appName="spam")
ssc = StreamingContext(sc, 1)
sql_context = SQLContext(sc)
nb = joblib.load('nb_model.pkl')
perceptron = joblib.load('perceptron_model.pkl')
k_means = joblib.load('k_means_model.pkl')

my_schema = StructType([
    StructField("feature0", StringType(), True),
    StructField("feature1", StringType(), True),
    StructField("feature2", StringType(), True),
])


#
def RDDtoDf(x):
    spark = SparkSession(x.context)
    if not x.isEmpty():
        y = x.collect()[0]

        z = json.loads(y)
        # print(z)
        k = z.values()
        # print(k)
        df = spark.createDataFrame(k, schema=my_schema)
        # df.show()
        label_encoder(df)


#
def label_encoder(df):
    le = LabelEncoder()
    y = le.fit_transform(np.array(df.select('feature2').collect()))
    tokenizer(df, y)


def tokenizer(df, y):
    feature1 = RegexTokenizer(inputCol="feature1", outputCol='tokens1', pattern='\\W')
    tk1 = feature1.transform(df).select('tokens1')
    stop_words_remover(tk1, y)


#
#
def stop_words_remover(col1, y):
    feature1 = StopWordsRemover(inputCol='tokens1', outputCol='filtered_words1')
    swr1 = feature1.transform(col1).select('filtered_words1')
    hash_vectoriser(swr1, y)


def hash_vectoriser(swr1, y):
    X1 = np.array(swr1.withColumn("filtered_words1", concat_ws(" ", "filtered_words1")).collect()).tolist()
    X11 = [i[0] for i in X1]

    hv = HashingVectorizer(alternate_sign=False)
    X111 = hv.fit_transform(X11).toarray()
    values_nb = nb.predict(X111)
    print("nb test accuracy: %.3f" % accuracy_score(y, values_nb))

    values_perceptron = perceptron.predict(X111)
    print("perceptron test accuracy: %.3f" % accuracy_score(y, values_perceptron))

    values_kmeans = k_means.predict(X111)
    print("K-Means test accuracy: %.3f" % accuracy_score(y, values_kmeans))


records = ssc.socketTextStream("localhost", 6100)
if records:
    records.foreachRDD(RDDtoDf)
ssc.start()
ssc.awaitTermination()
ssc.stop()
