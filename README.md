task1 
# 读取CSV文件
rdd = spark.read.csv
# 映射数据，将每一行转换
date, (total_purchase_amt, total_redeem_amt)
# 聚合数据，计算每天的总资金流入和
(lambda a, b: (a[0] + b[0], a[1] + b[1]))
# 排序格式化输出
# 保存结果
运行时间太长，多次爆内存，使得我只能分批筛选数据然后运行，最后进行合并
task2
几秒就能跑完
rdd
# 过滤2014年8月的数据，
映射为(user_id, report_date)
filtered_rdd = rdd.filter(lambda row: ）
# 去重，计算每个用户在8月的活跃天数
user_active_days 
# 过滤出活跃用户（至少5天记录）
active_users = user_active_days.filter(lambda x: x[1] >= 5)
# 计算活跃用户总数
active_user_count = active_users.count()
天池的比赛
数据分数可能不怎么样QWQ，本来是想做了一下递归，但是发现总是出问题，后来才进行普通的线性回归，然后手动加噪声查看影响
最后抱怨一下，本来是想把csv数据文件（输入数据）全部上传，但是文件太大，导致我git都捣鼓了半天最后放弃了。。。
