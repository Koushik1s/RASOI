from ultralytics import YOLO






model = YOLO(r"D:\Previous Think\Final Dataset\runs\detect\rasoi_yolov12s_aug_v1_extend_part_4\weights\best.pt")

text_file_name =  "test_dataset_map"
csv_file_name = "test_dataset_map_per_class_wise_0.736.csv"
yaml_file = r"D:\Previous Think\Final Dataset\train data final submit\data.yaml"
split = "test"

result = model.val(data=yaml_file,imgsz=640,batch=16,split=split)

data = {"map_0.5_0.95" : result.box.map,"map_0.5":result.box.map50,"map_0.75":result.box.map75}
print(data)
per_class = {
    "class" : list(model.names.values()),
    "MAP50-95"   : result.box.maps
}


import pandas as pd

with open(text_file_name,"w+") as f:
    for i in data:
     f.write(f"{i} : {data[i]}\n")
df = pd.DataFrame(per_class)
print(df)
df.to_csv(csv_file_name,index=False)

# model.predict(source=r"D:\Previous Think\Final Dataset\train data final submit\test\images",save=True,name="Final test")


