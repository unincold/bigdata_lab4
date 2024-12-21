from pyspark.sql import SparkSession
from datetime import datetime
# 创建SparkSession
spark = SparkSession.builder.appName("Daily Capital Flow Calculation with RDD").getOrCreate()
# 读取CSV文件
rdd = spark.read.csv("user_balance_table.csv", header=True, inferSchema=True).rdd
# 映射数据，将每一行转换为(date, (total_purchase_amt, total_redeem_amt))
mapped_rdd = rdd.map(lambda row: (row["report_date"], (row["total_purchase_amt"], row["total_redeem_amt"])))
# 聚合数据，计算每天的总资金流入和流出
aggregated_rdd = mapped_rdd.reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))
# 排序
sorted_rdd = aggregated_rdd.sortByKey()
# 格式化输出
result_rdd = sorted_rdd.map(lambda x: (x[0], x[1][0], x[1][1]))
# 保存结果
result_rdd.saveAsTextFile("task1_result")
# 停止SparkSession
spark.stop()


# 创建SparkSession
spark = SparkSession.builder.appName("Active Users Calculation").getOrCreate()
# 读取CSV文件
file_path = "user_balance_table.csv"  
rdd = spark.read.csv(file_path, header=True, inferSchema=True).rdd
# 过滤2014年8月的数据，并映射为(user_id, report_date)
filtered_rdd = rdd.filter(lambda row: datetime.strptime(str(row["report_date"]), "%Y%m%d").strftime("%Y%m") == "201408").map(lambda row: (row["user_id"], row["report_date"]))
# 去重，计算每个用户在8月的活跃天数
user_active_days = filtered_rdd.distinct().mapValues(lambda x: 1).reduceByKey(lambda a, b: a + b)
# 过滤出活跃用户（至少5天记录）
active_users = user_active_days.filter(lambda x: x[1] >= 5)
# 计算活跃用户总数
active_user_count = active_users.count()
# 打印活跃用户总数
print(active_user_count)
# 停止SparkSession
spark.stop()