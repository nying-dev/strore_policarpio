
#firebase fire store
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import json
import math

#firebase 
from flask import Flask,request,jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel
import pandas as pd
import numpy as np
import random 
import re
#
from ast import literal_eval
import collections


store =''
#data food data set
allergy_food= pd.read_csv("FoodData.csv", encoding= 'unicode_escape')
food = allergy_food['Food'].tolist()
allergy = allergy_food['Allergy'].tolist()
aller_indices = pd.Series(allergy_food.index,index = allergy_food['Allergy']).drop_duplicates()

#get data

#dataSetStore=pd.concat(map(pd.read_csv, ["Datasets/Policarpio_Store_Inventory.csv", "Datasets/Mycols_Store_inventory.csv"]), ignore_index=True)
dataSetStore = pd.read_csv("Policarpio_Store_Inventory.csv")

#clean column
policarpio_clean = dataSetStore.drop(columns=['STOCKQUANTITY','INVENTORYVALUE'])

#drop empty row
policarpio_clean.drop(policarpio_clean.filter(regex="Unnamed"),axis=1, inplace=True)

#clean words
tfv = TfidfVectorizer(min_df=3,max_features=None,
                  strip_accents='unicode',analyzer='word',token_pattern=r'\w{1,}',
                  ngram_range=(1,3),stop_words='english')

def c_merge(firstword,secondword):
   df = firstword + ' ' + secondword
   return df

for_price_clean = policarpio_clean['PRODUCTDESCRIPTION']
#clean nan word
print(type(for_price_clean))
policarpio_clean['INGREDIENTS'] = policarpio_clean['INGREDIENTS'].fillna('')

#conver to vector
tfv_price = tfv.fit_transform(for_price_clean)
tfv_policarpio = tfv.fit_transform(policarpio_clean['INGREDIENTS'])

#sigmoid computation
sig_for_price = sigmoid_kernel (tfv_price,tfv_price)
sig = np.genfromtxt('policarpio_sigmoid.csv',delimiter=',')

#indices
indices = pd.Series(policarpio_clean.index,index=policarpio_clean['PRODUCTNAME']).drop_duplicates()

app = Flask(__name__)

@app.route('/',methods=['GET'])
def test():
        quantity = request.args.get('quantity')
        c=policarpio_clean.to_json(orient="records")
        return jsonify((json.loads(c))[:int(quantity)])

@app.route('/recommend',methods=['POST'])
def recommend_fun():
	item = request.form.get('item')          
	list_rec=give_rec(item)
	df=list_rec.to_json(orient="records")
	data = json.loads(df)
	return jsonify(data[:10])

def give_rec(title,sig=sig):
#index of item
	idx=indices[title]
	print('index: '+str(idx))
	sig_scores=list(enumerate(sig[idx]))
	print('sig score list: '+str(sig_scores))
	#sorted score of sigmoid
	sig_scores=sorted(sig_scores,key=lambda x:x[1],reverse=True)
	print('sig scores sorted: '+str(sig_scores))
	#list zero to 10 list highest score
	sig_scores=sig_scores
	#get list in sig_scores
	policarpio_indices=[i[0] for i in sig_scores]
	print(policarpio_indices)
	#return the list of policarpio clean
	return policarpio_clean.iloc[policarpio_indices]


'''recommended item '''
@app.route('/cart',methods=['POST'])
def list_item():
	sig_enum=[]
	hold= request.form.getlist('list')
	lists = literal_eval(hold[0])
	ds={}
	#items
	indexes = list(indices[lists])
	price_items = policarpio_clean.iloc[indexes]
	#enumerate list and sorted
	for i in range(len(indexes)):
	   enum = list(enumerate(sig_for_price[indexes[i]]))
	   enum = sorted(enum,key=lambda x:x[1],reverse=True)
	   enum = enum[0:10]
	   item = [tupl[0] for tupl in enum]
	   item = policarpio_clean.iloc[item].to_json(orient="records")
	   item_json = json.loads(item)
	   check_price =[y for y in item_json if  float(y.get('PRICE'))
	   < float(policarpio_clean['PRICE'][indexes[i]])] 
	   sig_enum.append(sorted(check_price, key = lambda x:x['PRICE'],reverse=True )) 
	return jsonify(sig_enum)


'''filtering item with recommend'''
@app.route('/health',methods=['POST'])
def get_health():
	foods=[]
	user_aller = request.form.getlist('allergy')
	item= request.form.get('item')
	user_allergy = literal_eval(user_aller[0])
	if user_allergy and item:
	   #get item category
	   item_category = policarpio_clean['CATEGORY'][indices[item]] 
	   items = give_rec(item)
	   items = items.iloc[np.where((items['CATEGORY'] == item_category)==True)]
	   inx  = aller_indices[user_allergy].values
	   #list of ingredients not for have allergy
	   for i in inx:
	     foods.append(food[i].lower())
	   bad = items['INGREDIENTS'].apply(lambda x: any(item for item in foods if item in x.lower()))
	   cat_food = items.iloc[np.where(bad==False)[0]]
	   df=cat_food.to_json(orient="records")
	   data = json.loads(df)
	   return jsonify(data)
	else:
	   randomlist=[]
	   for i in range(0,20):
	      n = random.randint(1,len(policarpio_clean))
	      randomlist.append(n)
	   rando_item=policarpio_clean.iloc[randomlist]
	   df=rando_item.to_json(orient="records")
	   data = json.loads(df)
	   return jsonify(data)

'''check if item is good for you'''
@app.route('/allergy',methods=['POST'])
def allergy_for():
	foods=[]
	corpus=[]
	#request_data =request.get_json()
	item = request.form.get('item')
	allergy = request.form.getlist('allergy')
	if allergy and item:
	      '''array = literal_eval(allergy[0])'''
	      inx = aller_indices[allergy].values
	      idx = indices[item]
	      item_ingredients = policarpio_clean['INGREDIENTS'][idx]
	      for i in inx:
	         foods.append(food[i].lower())
	      is_not_good = any(item for item in foods if item in\
              item_ingredients.lower() and policarpio_clean['CATEGORY'][idx] not in \
              ['Personal Care','Household Care'])
	      return str(is_not_good)
	else:
	      return str(False)


@app.route('/history',methods=['POST'])
def history_recommend():
	_list = request.json['list']
	c = collections.Counter([x for sublist in _list for x in sublist])
	new_c = pd.Series(c.keys(),index = c.values())
	return jsonify(list(new_c.sort_index(ascending=False)))



    



#for doc in docs:
    #print('{} =>{}'.format(doc.id,doc.to_dict()))


#fire base 
if __name__ =='__main__':
    app.run(debug=True)
