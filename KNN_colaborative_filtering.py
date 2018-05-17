#VARIABLES ####################################################################
item_file_path='items.json'
rating_file_path='ratings.csv'
train_percent=0.7
user_feature_name='user'
item_feature_name='item'
rating_feature_name='rating'
item_id_feature_name='id'
feature_name_collection=['id']
knn_k = 100
#READ DATA#####################################################################
with open(item_file_path,'r') as item_file:
    import json
    item_data = [json.loads(line) for line in item_file]

with open(rating_file_path, 'r', newline='') as ratings_file:
    import csv
    reader = csv.DictReader(ratings_file)
    rating_data=[row for row in reader]

#FEATURE EXTRACTION############################################################
items_collection={}
for item in item_data:
    item_id=str(item[item_id_feature_name])
    """
    Any categorical features can be added to 'item', here.
    """
    items_collection[item_id]=item

Users=list(set([x[user_feature_name] for x in rating_data]))
###############################################################################
Prefrence_mat=[]
for feature_name in feature_name_collection:
    dataset=[[record[user_feature_name],int(record[rating_feature_name])
        ,items_collection[record[item_feature_name]][feature_name]] for record in rating_data]
    Feature_classes=set([x[2] for x in dataset])
    ##Remove repetitious records from data set by averaging from ratings 
    Dataset_collection={}
    for u,r,f in dataset:
        if u in Dataset_collection:
            if f in Dataset_collection[u]:
                Dataset_collection[u][f]=[r]+Dataset_collection[u][f]
            else: 
                Dataset_collection[u][f]=[r]
        else: 
            Dataset_collection[u]={f:[r],}
    dataset=[[u,sum((Dataset_collection[u])[f])/len((Dataset_collection[u])[f]),f] for u,r,f in dataset]    
    #TRAINING PROCESS##########################################################
    for index,user in enumerate(Users):
        from Collaborate_Filter import Collaborate_Filter
        cf = Collaborate_Filter(dataset, knn_k)
        k_nearest_neighbors = cf.k_nearest_neighbors(user, knn_k)
        prediction={}
        for item,val in items_collection.items():
            prediction[item]={cf.predict(user, val[feature_name], k_nearest_neighbors)}
        del cf
        Prefrence_mat.append(prediction)
        print(index)
###############################################################################







