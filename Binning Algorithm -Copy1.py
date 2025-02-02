#!/usr/bin/env python
# coding: utf-8

# In[1]:


# spark imports 
import pyspark
import pyspark.sql.functions as f
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
import math
from pyspark.ml.feature import Bucketizer

import findspark

# Apache Spark imports
import pyspark
from pyspark.sql.functions import *
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import split, col, isnan, when, count
from pyspark.sql.types import StringType, DoubleType, IntegerType, BooleanType, TimestampType
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, Bucketizer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier, DecisionTreeClassifier, GBTClassifier, NaiveBayes
from pyspark.mllib.evaluation import MulticlassMetrics
import builtins


# Standard Python imports
import traceback
import sys
import numpy as np

findspark.init()
findspark.find()

spark = SparkSession.builder.appName("BinARuleMining").master("local[20]").config("spark.driver.cores", "4").config("spark.driver.memory", "6g").config("spark.executor.memory", "8g").config("spark.executor.cores", "4").getOrCreate()      

# Sanity check:  print spark configuration data.
for a in spark.sparkContext._conf.getAll():
    print(a)
    
# set path destination and show available attributes 
#path_to_data = "hdfs://hadoop-master:9000//bagui-class/RPlenkers/Datasets/2022-05-23/Full_Dataset/full_dataset.csv"
#path_to_data = "unbinnedprivandbenign.csv"
path_to_data = 'unbinned Benign+Discovery _Full Discovery Set.csv'
conn_df = spark.read.option("header", True).options(inferSchema = True).csv(path_to_data)

conn_df.printSchema()

#spark = SparkSession.builder.appName("BinARM").getOrCreate()


# In[ ]:





# In[2]:


#Assign attack status based on mitr_attack information

def assignAtkStaus(stringval):
    if stringval == f.nanvl or stringval == 'none':
        return 0
    else: 
        return 1

udfAtkStatus = f.udf(assignAtkStaus, "int")

conn_atk_df = conn_df.withColumn("labal_tactic", udfAtkStatus('label_tactic'))


# In[ ]:





# In[3]:


#Option B: Resulting transformation using pyspark dataframe instead of panda series: 
# Slower, but uses a more traditional slice which is value agnostic

def returnTrimmedDF(data_frame, col_name, percentTrim):
    
    new_df = data_frame.dropna()
    new_df = new_df.sort(col_name)

    count = new_df.count()
    outlier = math.floor(count * percentTrim)
    
    w = Window.orderBy(f.monotonically_increasing_id())
    new_df = new_df.withColumn("trim_row_num", f.row_number().over(w))

    new_df = new_df.filter(f.col("trim_row_num") >= outlier)
    new_df = new_df.filter(f.col("trim_row_num") <= (count - outlier))
    
    return new_df.select(col_name)


# In[4]:


# PySpark DF version of the method to generate a tuple of edges for bins

def genNumericEdges(clean_df, col_name):
    calc_df = clean_df.select(f.mean(col_name),f.stddev(col_name))
    mean_val = calc_df.collect()[0][0]
    stddev_val = calc_df.collect()[0][1]

    if stddev_val == f.nanvl:
        stddev_val = 0

    if mean_val == f.nanvl:
        mean_val = 0

    edge0 = float('-inf')
    edge1 = mean_val - stddev_val * 2
    edge2 = mean_val - stddev_val
    edge3 = mean_val
    edge4 = mean_val + stddev_val
    edge5 = mean_val + stddev_val * 2
    edge6 = float('inf')
    edges = [edge0, edge1, edge2, edge3, edge4, edge5, edge6]
    edges_distinct = []

    print("edges are" + str(edge0) + str(edge1) + str(edge2) + str(edge3) + str(edge4) + str(edge5) + str(edge6))

    for i in edges:
        if edges_distinct.__contains__(i):
            continue
        else: edges_distinct = edges_distinct + [i]
    return edges_distinct


# In[5]:


#Pyspark use generated edges to bin using the Bucketizer functionality


def binNumericFeature(df, col_name, new_col_name, edges, replace_bool):
    
    # Print the column names and dataframe information
    print(f'Column being bucketized: {col_name}')
    print(f'New column name for bucketized data: {new_col_name}')
    print(f'Dataframe before bucketizing:\n{df.select(col_name).show(truncate=False)}')
    
    # Bucketize the numeric column
    df_binned = Bucketizer(splits=edges, inputCol=col_name, outputCol=new_col_name).setHandleInvalid("keep").transform(df)
    
    # Print information after bucketizing
    print(f'Dataframe after bucketizing:\n{df_binned.select(new_col_name).show(truncate=False)}')
     # Calculate bin ranges
        
        
    # Calculate bin ranges
    bin_ranges = {}
    for i in range(len(edges) - 1):
        bin_ranges[i] = (edges[i], edges[i + 1])
    
    print(f"Bin Ranges for {col_name}: {bin_ranges}")
    bin_ranges_accumulator.append({"Column": col_name, "Bin_Ranges": bin_ranges})
    
    
    # Cast the bucketized column to integer and handle missing values
    df_binned = df_binned.withColumn(new_col_name, f.col(new_col_name).cast('int')).fillna(0, subset=[new_col_name])
    
    # Conditionally replace the original column with the bucketized one
    if replace_bool:
        df_binned = df_binned.drop(col_name).withColumnRenamed(new_col_name, col_name)
    
    return df_binned


# In[6]:


#Combined the above methods to output a new df with the binned column.


def genNumericBinnedDF(df, col_name, trim, replace_bool):
    
    new_col_name = col_name + "_bin"

    # Create a sorted and trimmed version of the col_df to 
    # use for edge creation
    
    trimmed_df = returnTrimmedDF(df.select(col_name), col_name, trim)
    # Use the trimmed_df to generate edges based on standard
    # deviation schema
    
    edge_bins = genNumericEdges(trimmed_df, col_name)
    
   # print('this is the edge bins' + edge_bins)
    
    # Use list of edges generated to bin column using Bucketizer
    
    binned_df = binNumericFeature(df,col_name, new_col_name, edge_bins, replace_bool)
    print(binned_df)

    return binned_df


# In[7]:


#Assigns the IP class designation (A, B, C) as 1, 2, 3

@f.udf
def ipClass(string_val):
    bin_num = 0
    
    if  string_val == f.nanvl or string_val.find(".") == -1:
        return bin_num
    
    first = int(string_val.split('.')[0])

    # Class A
    if first >= 0 and first <= 127:
        bin_num = 1

    # Class B
    elif first > 127 and first <= 191: 
        bin_num = 2
    
    # Class C
    elif first <= 223: 
        bin_num = 3
    
    else: bin_num = 4

    return bin_num


# In[8]:


#Using the udf method on the dataframe and column read in

def genIPBinnedDF(df, col_name, replace_bool):
    new_col_name = col_name + "_bin"
    new_df = df.withColumn(new_col_name, ipClass(col_name).cast("int") )
    
    if replace_bool == True:
        new_df = new_df.drop(col_name).withColumnRenamed(new_col_name, col_name)
    
    return new_df


# In[9]:


#This method is pretty self contained, generating a df with the counts of each occurring value,
#sorting by occurrence, then aggregating the sum until 80% of the total occurences are covered.
#These values will be assigned their own bins, while everything else is pooled into a single bin.
# Pyspark version of top 80%

# #Correctly Producing Bins 
def genNominalBinnedDF(df, col_name, percent_aggr, replace_bool):

    bin_col_name = col_name + '_bin'
    col_ref = col_name + '_ref'
    
    w = Window.orderBy(f.monotonically_increasing_id())
    
    df_bin = df.select(col_name).groupBy(col_name).count().sort("count", ascending=False).dropna()
    df_bin = df_bin.withColumn(bin_col_name, f.row_number().over(w)).withColumnRenamed(col_name, col_ref)
    
        
#     # Print unique values and their bins
    print("    ")
    print("    ")
    print("    ")
    print(bin_col_name)
    print('--------------------------------')
    seen_values = set()
    for row in df_bin.collect():
        value = row[col_ref]
        if value not in seen_values:
            print(f'{value} -> Bin {row[bin_col_name]}')
            seen_values.add(value)
            
    #print('Checking values equate to bins:')
    seen_values = set()
    for row in df_bin.collect():
        value = row[col_ref]
        #print(value)
        
        #if value not in seen_values:
            #print(f'{value} -> Bin {row[bin_col_name]}')
            #seen_values.add(value)

    unique_count = df_bin.count()
    sum_total = df_bin.select("count").groupBy().sum().collect()[0][0]
    sum_aggr = 0
    j = 0
    
    for i in range(unique_count):
        sum_aggr += df_bin.collect()[i][1]
        
        if sum_aggr <= 0:
            break
        elif sum_aggr >= (sum_total * percent_aggr):
            j = i
            break
        else:
            continue
    
    j = builtins.max(j+1, builtins.min(5, unique_count))
    
  
    df_bin = df_bin.filter( f.col(bin_col_name) <= j ).drop("count")
    
    df = df.join(df_bin, df.__getattr__(col_name) == df_bin.__getattr__(col_ref),"left").drop(col_ref)
    
    df_bin.unpersist()

    #Where the original column was null, assigned a 0 value
    
    df = df.withColumn(bin_col_name, f.when(df[col_name].isNull(), 0).otherwise(df[bin_col_name] )) 
    
    #Where the original column was not null, but not included in the top 80% select, assign +1 to the highest
    #current value

    df = df.withColumn(bin_col_name, f.when(df[bin_col_name].isNull(), j+1).otherwise(df[bin_col_name] )) 

    #If the replace boolean is true, replace the original column with the numeric bin column

    if replace_bool == True:
        df = df.drop(col_name).withColumnRenamed(bin_col_name, col_name)

    return df




# In[10]:


#This part will be used to setup some of the base values. The dataframe to be used as the base is
#established, as well as the columns to be binned. The atk attribute, as well as any other attribute
#to no be binned, can be included as well using the attrList_base object.
assignAtkStaus(conn_df)

df_to_bin = conn_atk_df
atk_name = "atk"

numeric_percent_trim = 0.10
nominal_percent_agg = 0.80
replace_bool = True

#Note, should not use 'src_port_zeek' and 'dest_port_zeek'in caclulations. Too many distinct values.

attrList_base = ['label_tactic'] #'uid', 'datetime', 'community_id'
attrList_ip_addr = ['dest_ip_zeek','src_ip_zeek']
attrList_nominal = ['proto', 'conn_state', 'local_orig', 'local_resp', 'history','service']
attrList_numeric = ['duration','orig_bytes','orig_pkts','orig_ip_bytes','resp_bytes','resp_pkts','resp_ip_bytes','missed_bytes']

#df_to_bin = attrList_base


# In[11]:


bin_ranges_accumulator = []
bin_nom = []


# In[ ]:





# In[ ]:





# In[12]:


#In this part I used the above methods for binning individual columns, then bring them all together
#again using sql joins on the row number index I generated in the prior panel. The result is the 
#indexed_binned_df data frame, with all attributes binned with integers, for use in our ML aglorithms.

def genFullBinnedDF (df, attrList_ip_addr, attrList_nominal, nominal_percent_agg, attrList_numeric, numeric_percent_trim, replace_bool):
    
#     for i in attrList_ip_addr:
#         #ip_df = df_to_bin.select(temp_index , i)
#         df = genIPBinnedDF(df, i, replace_bool)
#         #indexed_binned_df = indexed_binned_df.join(bin_df, indexed_binned_df.__getattr__(new_index) == bin_df.__getattr__(temp_index),"left").drop(temp_index)

    for i in attrList_nominal:
        #nom_df = df_to_bin.select(temp_index, i)
        df = genNominalBinnedDF(df, i, nominal_percent_agg, replace_bool)
        #indexed_binned_df = indexed_binned_df.join(bin_df, indexed_binned_df.__getattr__(new_index) == bin_df.__getattr__(temp_index),"left").drop(temp_index)

    for i in attrList_numeric:
        #num_df = df_to_bin.select(temp_index, i)
        df = genNumericBinnedDF(df, i, numeric_percent_trim, replace_bool)
        #indexed_binned_df = indexed_binned_df.join(bin_df, indexed_binned_df.__getattr__(new_index) == bin_df.__getattr__(temp_index),"left").drop(temp_index)

    return df


# In[13]:


binned_df = genFullBinnedDF(df_to_bin, attrList_ip_addr, attrList_nominal, nominal_percent_agg, attrList_numeric, numeric_percent_trim, replace_bool ).persist()


# In[ ]:





# In[14]:


binned_df.show(10)


# In[15]:


# Write bin ranges to CSV file
import csv
csv_file = 'binneddiscFinal.csv'
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Column', 'Bin_Number', 'Bin_Value'])
    
    # Write bin ranges from bin_ranges_accumulator
    for item in bin_ranges_accumulator:
        column = item['Column']
        for bin_number, value in sorted(item['Bin_Ranges'].items()):
            writer.writerow([column, bin_number, value])

print(f'CSV file {csv_file} has been created with the contents of bin_ranges_accumulator.')



# In[ ]:





# In[16]:


pandas_df = binned_df.toPandas()
pandas_df.to_csv('binned.csv', index=False)


# In[ ]:




