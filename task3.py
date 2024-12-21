from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, sum as _sum, col, date_format, year, month, dayofmonth, dayofweek
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DateType, FloatType
from pyspark.ml.feature import VectorAssembler
import pyspark.sql.functions as F
from pyspark.ml.regression import LinearRegression
from datetime import datetime
import calendar

# 创建SparkSession
spark = SparkSession.builder.appName("SimplifiedFundFlowPrediction").getOrCreate()

# 定义数据模式
schema = StructType([
    StructField("report_date", IntegerType(), False),
    StructField("total_purchase_amt", IntegerType(), False),
    StructField("total_redeem_amt", IntegerType(), False)
])

# 读取数据
data_path = "user_balance_table.csv"
df = spark.read.csv(data_path, header=True, schema=schema)

# 转换日期格式
df = df.withColumn("report_date", date_format(col("report_date"), "yyyy-MM-dd"))

# 聚合数据
aggregated_df = df.groupBy("report_date").agg(_sum("total_purchase_amt").alias("total_purchase_amt"), _sum("total_redeem_amt").alias("total_redeem_amt"))

# 特征工程
aggregated_df = aggregated_df.withColumn("year", F.year("report_date")) \
                             .withColumn("month", F.month("report_date")) \
                             .withColumn("day", F.dayofmonth("report_date")) \
                             .withColumn("day_of_week", F.dayofweek("report_date"))

# 选择特征列
feature_cols = ["year", "month", "day", "day_of_week"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# 转换特征
assembled_df = assembler.transform(aggregated_df)

# 拆分数据集为训练集和测试集
train_df = assembled_df.filter(col("report_date") < "2014-09-01")

# 训练模型
lr_purchase = LinearRegression(featuresCol="features", labelCol="total_purchase_amt")
lr_redeem = LinearRegression(featuresCol="features", labelCol="total_redeem_amt")
model_purchase = lr_purchase.fit(train_df)
model_redeem = lr_redeem.fit(train_df)

# 准备预测日期
year = 2014
month = 9
_, num_days = calendar.monthrange(year, month)
prediction_dates = [datetime(year, month, day) for day in range(1, num_days + 1)]

# 预测
predictions_purchase = []
predictions_redeem = []
for date in prediction_dates:
    features = [date.year, date.month, date.day, date.weekday()]
    features_df = spark.createDataFrame([features], ["year", "month", "day", "day_of_week"])
    assembled_features = assembler.transform(features_df)
    prediction_purchase = model_purchase.transform(assembled_features).collect()[0]["prediction"]
    prediction_redeem = model_redeem.transform(assembled_features).collect()[0]["prediction"]
    predictions_purchase.append((date.strftime("%Y%m%d"), prediction_purchase))
    predictions_redeem.append((date.strftime("%Y%m%d"), prediction_redeem))

# 创建预测DataFrame
predictions_schema = StructType([
    StructField("report_date", StringType(), True),
    StructField("predicted_total_purchase_amt", FloatType(), True),
    StructField("predicted_total_redeem_amt", FloatType(), True)
])
predictions_df = spark.createDataFrame(predictions_purchase, schema=predictions_schema).withColumnRenamed("predicted_total_purchase_amt", "purchase") \
    .join(spark.createDataFrame(predictions_redeem, schema=predictions_schema).withColumnRenamed("predicted_total_redeem_amt", "redeem"), "report_date")

# 显示预测结果
predictions_df.show()

# 停止SparkSession
spark.stop()