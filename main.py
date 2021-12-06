# import pyspark
# import pandas
# import tqdm
# import numpy
import json
import numpy as np
import pyspark
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, Word2Vec
from pyspark.sql import SQLContext, SparkSession
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import HashingVectorizer
from pyspark.sql.functions import concat_ws

sc = SparkContext("local[2]", appName="spam")
ssc = StreamingContext(sc, 1)
sql_context = SQLContext(sc)
cls = MultinomialNB()


# schema = StructType([
#     StructField("feature0", StringType(), True),
#     StructField("feature1", StringType(), True),
#     StructField("feature2", StringType(), True),
# ])
#
def RDDtoDf(x):
    spark = SparkSession(x.context)
    if not x.isEmpty():
        y = x.collect()[0]
        z = json.loads(y)
        k = z.values()
        df = spark.createDataFrame(k, schema=['feature0', 'feature1', 'feature2'])
        # cols = df.select("feature0")
        label_encoder(df)
        # print(type(cols))
        # return df
        # feature0 = RegexTokenizer(inputCol="feature0", outputCol='tokens0', pattern='\\W')
        # tk = feature0.transform(df)
        # tk.show()


def label_encoder(df):
    le = LabelEncoder()
    y = le.fit_transform(np.array(df.select('feature2').collect()))
    # print(y)
    tokenizer(df, y)


def tokenizer(df, y):
    feature0 = RegexTokenizer(inputCol="feature0", outputCol='tokens0', pattern='\\W')
    feature1 = RegexTokenizer(inputCol="feature1", outputCol='tokens1', pattern='\\W')
    tk0 = feature0.transform(df).select('tokens0')
    tk1 = feature1.transform(df).select('tokens1')
    stop_words_remover(tk0, tk1, y)


def stop_words_remover(col1, col2, y):
    feature0 = StopWordsRemover(inputCol='tokens0', outputCol='filtered_words0')
    feature1 = StopWordsRemover(inputCol='tokens1', outputCol='filtered_words1')
    swr0 = feature0.transform(col1).select('filtered_words0')
    swr1 = feature1.transform(col2).select('filtered_words1')
    # word2vec(swr0, swr1, y)
    hash_vectoriser(swr0, swr1, y)

#
#
def hash_vectoriser(swr0, swr1, y):
    X1 = np.array(swr1.withColumn("filtered_words1", concat_ws(" ", "filtered_words1")).collect()).ravel()
    # print(X1)
    hv = HashingVectorizer(alternate_sign=False)
    X1 = hv.fit_transform(X1).toarray()
#     # print(X1, y)
#     print(np.shape(X1), np.shape(y))
#  print(type(model1))
    x_train, y_train, x_test, y_test = train_test_split(X1, y, test_size=0.2)
    res = cls.partial_fit(x_train, y_train, classes=np.unique(y_train))
    print("test accuracy: %.3f" % cls.score(x_test, y_test))
    print("test accuracy: %.3f" % cls.score(x_train, y_train))

#     # hv_0 = model0.fit(col1)
#     model0.show()
# w2v_1 = Word2Vec(inputCol='filtered_words1', outputCol='vector1', vectorSize=50)
# model0 = w2v_0.fit(col1)
# model1 = w2v_1.fit(col2)
# result0 = model0.transform(col1)
# result1 = model1.transform(col2)
# vectors = result1.select('vector1').collect()[0]

# x_train, x_test, y_train, y_test = train_test_split(np.array(vectors), y.traspose(), test_size=0.25, train_size=0.75)
# classifier = MultinomialNB()
# classifier.fit(x_train, y_train)
# preds = classifier.predict(x_test)
# print(np.shape(np.array(vectors)), np.shape(np.transpose(y)))
# print((np.array(vectors), y.transpose()))
# result1.show()
# result1.show()
# nb = NaiveBayes(smoothing=1.0, modelType="multinomial")
# model = nb.fit(result1)
# predictions = model.transform()


#         return cols

# def naiveBayes(w2v, df):

# def tokenizer(df):
#
#     # feature0 = RegexTokenizer(inputCol=df.select("feature0"), outputCol='tokens0', pattern='\\W')
#     # print(feature0)
#     feature0 = RegexTokenizer(inputCol="feature0", outputCol='tokens0', pattern='\\W')
#     tk = feature0.transform()
#     # tk.select("tokens0").show()
#     test(df)

# print(x)
# df.show()


records = ssc.socketTextStream("localhost", 6100)
# records.flatMap(lambda x: print(x))
if records:
    records.foreachRDD(RDDtoDf)
ssc.start()
ssc.awaitTermination()
ssc.stop()
