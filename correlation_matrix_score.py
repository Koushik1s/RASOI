import pandas as pd 
import pickle
import json

with open(r"D:\Previous Think\test\63_food_class.txt","rb") as f :
    food_data = pickle.load(f)


df = pd.read_csv("Final_data.csv")

plate_name = {}


for i in df.iterrows():
    if i[1]["filename"] not in plate_name.keys() :
        plate_name[i[1]["filename"]] = [json.loads(i[1]["region_attributes"])["name"]]
    else :
        plate_name[i[1]["filename"]].append(json.loads(i[1]["region_attributes"])["name"])


    
data_count = {}

for i in food_data : 
    if i not in data_count.keys():
        data_count[i] = {}
        for j in plate_name:
            if i in plate_name[j] :
                for k in plate_name[j]:
                    c = 0 
                    if k == i :
                        c+=1
                    if k != i :
                        if k not in data_count[i].keys():
                            data_count[i][k]=1
                        else:
                            data_count[i][k]+=1
                    if c>=2:
                        if k not in data_count[i].keys():
                            data_count[i][k]=1
                        else:
                            data_count[i][k]+=1
                            
    else :
        
         for j in plate_name:
            if i in plate_name[j] :
                for k in plate_name[j]:
                    c = 0 
                    if k == i :
                        c+=1
                    if k != i :
                        if k not in data_count[i].keys():
                            data_count[i][k]=1
                        else:
                            data_count[i][k]+=1
                    if c>=2:
                        if k not in data_count[i].keys():
                            data_count[i][k]=1
                        else:
                            data_count[i][k]+=1
                            


        

for i in food_data :
    for j in data_count:
        if i not in data_count[j].keys():
            data_count[j][i] = 0.5
     


    
food_data_equiq_thali_count = {}

for k in food_data:
    
    for i in plate_name :
        if k not in food_data_equiq_thali_count.keys():
            food_data_equiq_thali_count[k] = 0
        if len(plate_name[i]) >= 2 and k in plate_name[i]  :
            
            food_data_equiq_thali_count[k]+=1

   

import numpy as np


Final_matrix_data = {}

for i in food_data :
    Final_matrix_data[i] = []
    for k in food_data:
            
            Final_matrix_data[i].append(data_count[i][k])
        

for i in food_data :
    Final_matrix_data[i] = ((np.array(Final_matrix_data[i],dtype=np.float64)/food_data_equiq_thali_count[i]) )*5


#Final_matrix_data["Papad"] = np.array([0.589,0.779,0.989,0.089,0.089,0.089,0.489,0.389,0.089,1.25,0.089,0.379,0.189,0.389,0.089,0.089,0.679,1.964,0.089,0.289,0.089,0.089,0.089,0.089,0.989,0.279,0.089,0.489,0.536,0.089,0.089,0.536,0.089,0.289,0.089,0.089,0.889,0.089,3.214,4.2,0.089,0.357,0.089,0.089,0.089,0.089,0.089,0.389,0.089,2,0.989,0.089,0.509,0.179,0.089,0.089,0.089,0.089,0.089,0.489,0.089,0.179,0.189])
#Final_matrix_data["Rice"]  = np.array([0.375,1,0.062,0.062,0.062,0.062,0.125,0.062,0.062,0.625,0.062,1.25,0.125,0.125,0.062,0.062,0.25,2,0.062,0.062,0.062,0.062,0.062,0.062,0.5,0.062,0.25,0.062,0.5,0.062,0.125,1.5,0.062,0.062,0.062,0.375,0.062,0.062,2.375,0.062,0.062,0.062,0.125,0.125,0.125,0.062,0.062,0.062,0.062,3.375,0.062,0.062,0.125,0.25,0.125,0.062,0.125,0.125,0.062,0.125,0.062,0.062,0.375])


import  pickle

with open("Matrix_data_save.txt","wb") as f:
    pickle.dump(Final_matrix_data,f)    

with open ("matrix.csv","w+") as f :
    f.writelines(["name,",*[i+"," for i in food_data],"\n"])
    for i in food_data:
        s1 = f"{i},"
        for j in Final_matrix_data[i]:
            s1+=f"{j:.3f},"
        s1 = s1[:-1]+"\n"
        f.write(s1)
            


