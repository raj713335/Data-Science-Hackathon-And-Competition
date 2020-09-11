import pandas as pd

# df=pd.read_csv('dataset/sample.csv')
#
# print(df.head())
#
#
#
#
# #df.sort_values(by=['Filename'])
#
# df=df.drop('Category',axis=1)
#
# df["Category"]="Toddler"
# print(df.head())
#
#
#
#
#
#
# df.to_csv("sub.csv")
#
#

import os

data=[]

for i in ["Adults","Teenagers","Toddler"]:
    path = "dataset/data/train/"+i
    for each in os.listdir(path):
        data.append([each,i])


print(data)



df = pd.DataFrame(data, columns=["Filename","Category"])


print(df)


df1=pd.read_csv('dataset/sample.csv')

df1=df1.drop('Category',axis=1)


result = pd.merge(df1, df, how='left', on=['Filename'])

print(result.head())


result.to_csv("sub.csv",index=False)