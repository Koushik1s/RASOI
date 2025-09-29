from ultralytics.models import YOLO
from glob import glob
import pickle
import os 

Json_file_name = "class_data.json"  # Enter the Json File Name where you want to save your all updated preditected annotation

multiplier_factor = 0.64  # it's multiply to the minimum correlation score

model1 = YOLO(r"D:\Previous Think\Final Dataset\runs\detect\rasoi_yolov12s_aug_v1_extend_part_4\weights\best.pt")

dataset_path = r"D:\Previous Think\Final Dataset\train data final submit\test\images"

g1 = glob("*.jpg",root_dir=dataset_path)

g1.sort(key= lambda a : int(a[6:-4]))

with open(r"D:\Previous Think\Final Dataset\History\63_food_class.txt","rb") as f :
   _63_food_data = pickle.load(f) 


datasets = {}




for i in g1 :
    k = i
    datasets[k] = []
    i= os.path.join(dataset_path,i)
    result = model1.predict(i)[0]
    datasets[k].extend([ (_63_food_data[int(i1)],j1.tolist(),float(k1))  for i1,j1,k1 in zip( result.boxes.cls , result.boxes.xyxy,result.boxes.conf)])
    
with open(r"D:\Previous Think\Final Dataset\Matrix_data_save.txt","rb") as f :
    matrix = pickle.load(f)

_63_food_class_to_index = {}

for i,j in enumerate(_63_food_data) :
    _63_food_class_to_index[j] = i

import copy
main_delete_data = {}

for i in datasets:
    
    ex = [ (k[0],k[-1]) for k in datasets[i] ]
    highest = 0
    index = -1
    if len(ex) > 0 :
     highest = max(ex,key= lambda a : a[1])
     index = ex.index(highest)
    delete_food_index = []
    if len(ex) >= 2:
        food_col_data = matrix[highest[0]]
        
        k4 = 0
        while k4< len(ex):  #0.64
            if k4 != index :
                if food_col_data[ _63_food_class_to_index[ ex[k4][0]]]* ex[k4][1] < min(food_col_data)*multiplier_factor:
                    delete_food_index.append(k4)
            k4+=1
    main_delete_data[i] = copy.deepcopy(delete_food_index)


final_dataset = {}

for i in datasets :
    final_dataset[i] = []
    j = 0 
    while j < len(datasets[i]):
        if j not in main_delete_data[i]:
            final_dataset[i].append(datasets[i][j])
        j+=1


submit = {}
submit["root"] = dataset_path
submit["images"] = final_dataset

import json

s1 = json.dumps(submit)

with open (Json_file_name,"w+") as f :
    json.dump(fp=f,obj=submit)
    
    

    
# """
# final delete data : {'Plate 1.jpg': [], 'Plate 2.jpg': [], 'Plate 3.jpg': [], 'Plate 4.jpg': [5], 'Plate 5.jpg': [], 'Plate 6.jpg': [], 'Plate 7.jpg': [3, 4], 'Plate 8.jpg': [5], 'Plate 9.jpg': [], 'Plate 10.jpg': [], 'Plate 11.jpg': [], 'Plate 12.jpg': [], 'Plate 13.jpg': [], 'Plate 14.jpg': [], 'Plate 15.jpg': [], 'Plate 16.jpg': [], 'Plate 17.jpg': [], 'Plate 18.jpg': [], 'Plate 19.jpg': [], 'Plate 20.jpg': [], 'Plate 21.jpg': [], 'Plate 22.jpg': [], 'Plate 23.jpg': [], 'Plate 24.jpg': [], 'Plate 25.jpg': [], 'Plate 26.jpg': [], 'Plate 27.jpg': [], 'Plate 28.jpg': [], 'Plate 29.jpg': [], 'Plate 30.jpg': []}   
# """

# """{'Plate 1.jpg': [], 'Plate 2.jpg': [], 'Plate 3.jpg': [], 'Plate 4.jpg': [5], 'Plate 5.jpg': [], 'Plate 6.jpg': [], 'Plate 7.jpg': [3, 4], 'Plate 8.jpg': [5], 'Plate 9.jpg': [3], 'Plate 10.jpg': [], 'Plate 11.jpg': [], 'Plate 12.jpg': [], 'Plate 13.jpg': [], 'Plate 14.jpg': [], 'Plate 15.jpg': [], 'Plate 16.jpg': [], 'Plate 17.jpg': [], 'Plate 18.jpg': [], 'Plate 19.jpg': [], 'Plate 20.jpg': [], 'Plate 21.jpg': [], 'Plate 22.jpg': [], 'Plate 23.jpg': [], 'Plate 24.jpg': [], 'Plate 25.jpg': [], 'Plate 26.jpg': [], 'Plate 27.jpg': [], 'Plate 28.jpg': [], 'Plate 29.jpg': [], 'Plate 30.jpg': []}  
#     """

# """final delete data : {'Plate 1.jpg': [], 'Plate 2.jpg': [], 'Plate 3.jpg': [], 'Plate 4.jpg': [5], 'Plate 5.jpg': [], 'Plate 6.jpg': [], 'Plate 7.jpg': [3, 4], 'Plate 8.jpg': [5], 'Plate 9.jpg': [3], 'Plate 10.jpg': [], 'Plate 11.jpg': [], 'Plate 12.jpg': [], 'Plate 13.jpg': [], 'Plate 14.jpg': [], 'Plate 15.jpg': [], 'Plate 16.jpg': [], 'Plate 17.jpg': [], 'Plate 18.jpg': [], 'Plate 19.jpg': [], 'Plate 20.jpg': [], 'Plate 21.jpg': [], 'Plate 22.jpg': [], 'Plate 23.jpg': [], 'Plate 24.jpg': [], 'Plate 25.jpg': [], 'Plate 26.jpg': [], 'Plate 27.jpg': [], 'Plate 28.jpg': [], 'Plate 29.jpg': [], 'Plate 30.jpg': []}  "