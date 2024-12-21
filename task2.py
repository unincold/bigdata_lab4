from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, col, to_date, sum as _sum, row_number
from pyspark.sql.window import Window
import pandas as pd

# 创建SparkSession
spark = SparkSession.builder.appName("Daily Capital Flow Calculation with RDD").getOrCreate()

# 读取CSV文件
balance_df = spark.read.csv("user_balance_table.csv", header=True, inferSchema=True)
profile_df = spark.read.csv("user_profile_table.csv", header=True, inferSchema=True)

# 连接两个表
df = balance_df.join(profile_df, on="user_id")

# 转换日期格式并过滤2014年3月1日的数据
df = df.withColumn("report_date", to_date(col("report_date"), "yyyyMMdd"))
march1_data = df.filter(col("report_date") == "2014-03-01")

# 按城市分组并计算平均余额
avg_balance_per_city = march1_data.groupBy("city").agg(avg("tBalance").alias("average_balance"))

# 按平均余额降序排列
avg_balance_per_city_sorted = avg_balance_per_city.orderBy(col("average_balance").desc())

# 将结果转换为 Pandas DataFrame 并保存为 CSV 文件
avg_balance_per_city_sorted.toPandas().to_csv("avg_balance_per_city_sorted.csv", index=False)

# 停止SparkSession
spark.stop()

# 创建新的SparkSession
spark = SparkSession.builder.appName("Top Users by City").getOrCreate()

# 读取CSV文件
balance_df = spark.read.csv("user_balance_table.csv", header=True, inferSchema=True)
profile_df = spark.read.csv("user_profile_table.csv", header=True, inferSchema=True)

# 连接两个表
df = balance_df.join(profile_df, on="user_id")

# 转换日期格式并过滤2014年8月的数据
df = df.withColumn("report_date", to_date(col("report_date"), "yyyyMMdd"))
august_data = df.filter((col("report_date") >= "2014-08-01") & (col("report_date") <= "2014-08-31"))

# 计算每个用户的总流量
total_flow_per_user_city = august_data.groupBy("city", "user_id").agg(
    _sum("total_purchase_amt").alias("total_purchase"),
    _sum("total_redeem_amt").alias("total_redeem")
)
total_flow_per_user_city = total_flow_per_user_city.withColumn("total_flow", col("total_purchase") + col("total_redeem"))

# 为每个城市的用户按总流量排序并选择前三
window_spec = Window.partitionBy("city").orderBy(col("total_flow").desc())
ranked_users = total_flow_per_user_city.withColumn("rank", row_number().over(window_spec))

# 选择每个城市总流量排名前三的用户
top3_users_per_city = ranked_users.filter(col("rank") <= 3).select("city", "user_id", "total_flow")

# 将结果转换为 Pandas DataFrame 并保存为 CSV 文件
top3_users_per_city.toPandas().to_csv("top3_users_per_city.csv", index=False)

# 停止SparkSession
spark.stop()