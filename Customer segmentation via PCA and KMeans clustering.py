"""Created on Thu Apr 18 19:52:58 2019@author: Izadi"""

import os
from subprocess import check_output
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

"""After this section there is another one that I used boxplt to find outliers and the after filtering them
I created new csv filse and started to work with them in this section. In other words middle section is not part of the code
separated by ********
  """


os.chdir(r'D:\desktop\Python_DM_ML_BA\Bussinus Analytics')
aisles = pd.read_csv('aisles.csv')
aisles.shape
aisles.columns
aisles.isnull().sum()
aisles.head(10)
s = set(aisles.aisle)
l =len(s)
depts = pd.read_csv('departments.csv') 
depts.head()
depts.shape
depts.columns
s = depts.isnull().sum()

opt1 = pd.read_csv('opt1.csv',index_col=0)
opt1.shape
opt1.columns
s = opt1.isnull().sum()

opp1 = pd.read_csv('opp1.csv', index_col=0)
opp1.shape
opp1.columns
opp1.head()
s = opp1.isnull().sum()
s = orders1.isnull().sum()

orders1 = orders1.head()
sample = pd.read_csv('sample_submission.csv')
sample.shape
sample.columns
s = sample.isnull().sum()

sa = sample.head()
products = pd.read_csv('products.csv')
products.shape
products.columns
s = products.isnull().sum()
s
p = products.head()
pn = products.product_name
l = len(set(pn))                                    
l
#****************************************************************************************
opt = pd.read_csv('order_products__train.csv')
opt.shape
opt.head()
opt.columns
s = opt.isnull().sum()
s

opt.boxplot(column=['add_to_cart_order'])

opt.drop(opt[opt.add_to_cart_order > 22].index , inplace=True)

boxplot_2 = opt.boxplot(column=['add_to_cart_order', 'reordered'], patch_artist=True)

opt.to_csv('opt1.csv')

boxplot_all = products.boxplot(column= ['product_id'], patch_artist=True)
boxplot_all = products.boxplot(column= ['aisle_id', 'department_id'] , patch_artist=True)
boxplot_all = opt.boxplot(column= ['add_to_cart_order', 'reordered'], patch_artist=True)
boxplot_all = opt.boxplot(column=['order_id', 'product_id', 'add_to_cart_order', 
                                  'reordered'], patch_artist=True)boxplot_2 = opt.boxplot(column=['add_to_cart_order', 'reordered'])
boxplot_1 = opt.boxplot(column=['add_to_cart_order'])
boxplot_1 = opt.boxplot(column=['reordered'])

df.drop(df[df.score < 50].index, inplace=True)
df = df.drop(df[(df.score < 50) & (df.score > 20)].index, inplace=True)

opp = pd.read_csv('order_products__prior.csv')
opp.shape
opp.columns
s = opp.isnull().sum()
s
opp.head()

boxplot_all = opp.boxplot(column= ['order_id', 'product_id', 'add_to_cart_order', 'reordered'], patch_artist=True)

boxplot_all = opp.boxplot(column=['add_to_cart_order', 'reordered'], patch_artist=True)
boxplot_all = opp1.boxplot(column=['add_to_cart_order', 'reordered'], patch_artist=True)

boxplot_1opp = opp.boxplot(column=['add_to_cart_order'], patch_artist=True)
opp.drop(opp[opp.add_to_cart_order > 23].index, inplace=True)
opp.to_csv('opp1.csv')
opp1 = pd.read_csv('opp1.csv',index_col=0)
opp1.shape
opp1.head()
opp1.columns
s = opp1.isnull().sum()
boxplot_1opp1 = opp1.boxplot(column=['add_to_cart_order'], patch_artist=True)

orders  = pd.read_csv('orders.csv')
orders.shape
orders.columns
s = orders.isnull().sum()
s
orders.head()
orders.fillna({'days_since_prior_order':orders.days_since_prior_order.mode()[0]}, inplace=True)

boxplot_all = orders.boxplot(column=['order_number', 'order_dow',
       'order_hour_of_day', 'days_since_prior_order'], patch_artist=True)

boxplot_all = orders1.boxplot(column=['order_number', 'order_dow',
       'order_hour_of_day', 'days_since_prior_order'], patch_artist=True)

boxplot_1 = orders.boxplot(column=['order_number'], patch_artist=True)

r =orders.order_number > 34
orders.drop(orders[orders.order_number > 35].index, inplace=True)
orders.drop(orders[orders.order_hour_of_day < 3].index, inplace=True)
orders.shape

orders.to_csv('orders.csv')

orders.fillna({'days_since_prior_order':orders.days_since_prior_order.mode()[0]}, inplace=True)

boxplot_all = orders1.boxplot(column=['order_number', 'order_dow',
       'order_hour_of_day', 'days_since_prior_order'], patch_artist=True)

boxplot_1 = orders1.boxplot(column=['order_number'], patch_artist=True)


op = opp1[0:2000000]
op.shape

#order_prior = pd.merge(prior,orders,on=['order_id','order_id'])
#*******************************************************************************

prior = opp[0:2000000]
order_prior = pd.merge(prior,orders,on=['order_id','order_id'])
order_prior = order_prior.sort_values(by=['user_id','order_id'])
order_prior.head()
_mt = pd.merge(prior,products, on = ['product_id','product_id'])

_mt = pd.merge(_mt,orders,on=['order_id','order_id'])
_mt.head()
mt = pd.merge(_mt,aisles,on=['aisle_id','aisle_id'])
mt.head(10)
mt.tail(10)
m = mt.describe()
mt['product_name'].value_counts()[0:10]
len(mt['product_name'].unique())
len(mt['aisle'].unique())
mt['aisle'].value_counts()[0:10]
cust_prod = pd.crosstab(mt['user_id'], mt['aisle'])
cust_prod.head(10)
cust_prod.shape
from sklearn.decomposition import PCA
pca = PCA(n_components=8)
pca.fit(cust_prod)
pca_samples = pca.transform(cust_prod)
ps = pd.DataFrame(pca_samples)
ps.head()

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
tocluster = pd.DataFrame(ps[[7,1]])
print (tocluster.shape)
print (tocluster.head())

fig = plt.figure(figsize=(8,8))
plt.plot(tocluster[1], tocluster[7], 'o', markersize=10, color='red', alpha=0.5, label='class1')
plt.xlabel('x_values', size= 30)
plt.ylabel('y_values', size=30)
plt.title('PCA DATA', size=40)
plt.legend()
plt.show()

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
clusterer = KMeans(n_clusters=7,random_state=42).fit(tocluster)
centers = clusterer.cluster_centers_
c_preds = clusterer.predict(tocluster)
print(centers)
print (c_preds[0:100])

import matplotlib
fig = plt.figure(figsize=(8,8))
colors = ['orange','blue','purple','green', 'cyan', 'fuchsia', 'darkorchid' ]
colored = [colors[k] for k in c_preds]
print (colored[0:20])
plt.scatter(tocluster[1], tocluster[7],  color = colored)
for ci,c in enumerate(centers):
    plt.plot(c[0], c[1],  '*', markersize=20, color='red', alpha=0.9, label=''+str(ci))

plt.xlabel('x_values', size = 20)
plt.ylabel('y_values', size =20)
plt.title('SEVEN CLUSTERS', size =30)
plt.legend()
plt.show()

clust_prod = cust_prod.copy()
clust_prod['cluster'] = c_preds

clust_prod.head(10)

print (clust_prod.shape)
f,arr = plt.subplots(2,2,sharex=True,figsize=(15,15))

c1_count = len(clust_prod[clust_prod['cluster']==0])

c0 = clust_prod[clust_prod['cluster']==0].drop('cluster',axis=1).mean()
arr[0,0].bar(range(len(clust_prod.drop('cluster',axis=1).columns)),c0)
c1 = clust_prod[clust_prod['cluster']==1].drop('cluster',axis=1).mean()
arr[0,1].bar(range(len(clust_prod.drop('cluster',axis=1).columns)),c1)
c2 = clust_prod[clust_prod['cluster']==2].drop('cluster',axis=1).mean()
arr[1,0].bar(range(len(clust_prod.drop('cluster',axis=1).columns)),c2)
c3 = clust_prod[clust_prod['cluster']==3].drop('cluster',axis=1).mean()
arr[1,1].bar(range(len(clust_prod.drop('cluster',axis=1).columns)),c3)

c4 = clust_prod[clust_prod['cluster']==4].drop('cluster',axis=1).mean()
arr[0,0].bar(range(len(clust_prod.drop('cluster',axis=1).columns)),c4)
c5 = clust_prod[clust_prod['cluster']==5].drop('cluster',axis=1).mean()
arr[0,1].bar(range(len(clust_prod.drop('cluster',axis=1).columns)),c5)
c6 = clust_prod[clust_prod['cluster']==6].drop('cluster',axis=1).mean()
arr[1,0].bar(range(len(clust_prod.drop('cluster',axis=1).columns)),c6)
plt.show()

c0.sort_values(ascending=False)[0:10]
c1.sort_values(ascending=False)[0:10]
c2.sort_values(ascending=False)[0:10]
c3.sort_values(ascending=False)[0:10]
c4.sort_values(ascending=False)[0:10]
c5.sort_values(ascending=False)[0:10]
c6.sort_values(ascending=False)[0:10]


from IPython.display import display, HTML

cluster_means = [[c0['fresh fruits'],c0['fresh vegetables'],
                  c0['packaged vegetables fruits'],c0['yogurt'],
                  c0['packaged cheese'], c0['milk'],
                  c0['water seltzer sparkling water'],
                  c0['chips pretzels']],
                 [c1['fresh fruits'],c1['fresh vegetables'],c1['ice cream ice'],
                  c1['packaged vegetables fruits'],c1['yogurt'],c1['baby food formula'],
                  c1['packaged cheese'],c1['milk'],
                  c1['water seltzer sparkling water'],
                  c1['chips pretzels']],
                 [c2['fresh fruits'],c2['fresh vegetables'],c2['soy lactosefree'],
                  c2['packaged vegetables fruits'],c2['yogurt'],
                  c2['packaged cheese'],c2['milk'],
                  c2['water seltzer sparkling water'],
                  c2['chips pretzels']],
                 [c3['fresh fruits'],c3['fresh vegetables'],c3['canned jarred vegetables'],
                  c3['packaged vegetables fruits'],c3['yogurt'], 
                  c3['packaged cheese'],c3['milk'],
                  c3['water seltzer sparkling water'],
                  c3['chips pretzels']],
                 [c4['milk'],c4['refrigerated'],c4['energy granola bars']],
                 [c5['fresh herbs'],c5['frozen produce'],c5['water seltzer sparkling water']],
                 [c6['bread'],c6['soft drinks']]]
                
                 
cluster_means1 = pd.DataFrame(cluster_means, columns = ['fresh fruits','fresh vegetables',
                                                       'packaged vegetables fruits',
                                                       'yogurt','packaged cheese',
                                                       'milk','water seltzer sparkling water',
                                                       'chips pretzels'])
                 
                 
cluster_means = [[c0['bread'],c0['milk'],c0['yogurt'],c0['fresh herbs'],c0['fresh fruits'],c0['soft drinks'],
c0['fresh vegetables'],c0['packaged cheese'],c0['chips pretzels'],c0['baby food formula'],
c0['soy lactosefree'],c0['ice cream ice'],c0['refrigerated'],c0['frozen produce'],c0['energy granola bars'],
c0['packaged vegetables fruits'],c0['canned jarred vegetables'],c0['water seltzer sparkling water']], 
[c1['bread'],c1['milk'],c1['yogurt'],c1['fresh herbs'],c1['fresh fruits'],c1['soft drinks'],c1['fresh vegetables'], 
c1['packaged cheese'],c1['chips pretzels'],c1['baby food formula'],c1['soy lactosefree'],c1['ice cream ice'],
c1['refrigerated'],c1['frozen produce'],c1['energy granola bars'],c1['packaged vegetables fruits'], 
c1['canned jarred vegetables'],c1['water seltzer sparkling water']], 
[c3['bread'],c3['milk'],c3['yogurt'],c3['fresh herbs'],c3['fresh fruits'],c3['soft drinks'],c3['fresh vegetables'], 
c3['packaged cheese'],c3['chips pretzels'],c3['baby food formula'],c3['soy lactosefree'],c3['ice cream ice'],
c3['refrigerated'],c3['frozen produce'],c3['energy granola bars'],c3['packaged vegetables fruits'], 
c3['canned jarred vegetables'],c3['water seltzer sparkling water']], 
[c4['bread'],c4['milk'],c4['yogurt'],c4['fresh herbs'],c4['fresh fruits'],c4['soft drinks'],c4['fresh vegetables'], 
c4['packaged cheese'],c4['chips pretzels'],c4['baby food formula'],c4['soy lactosefree'],c4['ice cream ice'],
c4['refrigerated'],c4['frozen produce'],c4['energy granola bars'],c4['packaged vegetables fruits'], 
c4['canned jarred vegetables'],c4['water seltzer sparkling water']], 
[c5['bread'],c5['milk'],c5['yogurt'],c5['fresh herbs'],c5['fresh fruits'],c5['soft drinks'],c5['fresh vegetables'], 
c5['packaged cheese'],c5['chips pretzels'],c5['baby food formula'],c5['soy lactosefree'],c5['ice cream ice'],
c5['refrigerated'],c5['frozen produce'],c5['energy granola bars'],c5['packaged vegetables fruits'], 
c5['canned jarred vegetables'],c5['water seltzer sparkling water']], 
[c6['bread'],c6['milk'],c6['yogurt'],c6['fresh herbs'],c6['fresh fruits'],c6['soft drinks'],c6['fresh vegetables'], 
c6['packaged cheese'],c6['chips pretzels'],c6['baby food formula'],c6['soy lactosefree'],c6['ice cream ice'],
c6['refrigerated'],c6['frozen produce'],c6['energy granola bars'],c6['packaged vegetables fruits'], 
c6['canned jarred vegetables'],c6['water seltzer sparkling water']]] 

cluster_means = pd.DataFrame(cluster_means, columns = ['bread', 'milk','yogurt','fresh herbs',
    'fresh fruits','soft drinks', 'fresh vegetables', 
'packaged cheese','chips pretzels', 'baby food formula','soy lactosefree','ice cream ice',
'refrigerated','frozen produce', 'energy granola bars', 'packaged vegetables fruits', 
'canned jarred vegetables', 'water seltzer sparkling water'])

cluster_means 

cluster_means = pd.DataFrame(cluster_means, columns = ['bread','milk','yogurt','fresh herbs',
'fresh fruits','soft drinks', 'fresh vegetables', 
'packaged cheese','chips pretzels','baby food formula',
'soy lactosefree','ice cream ice',
'refrigerated','frozen produce', 'energy granola bars',
 'packaged vegetables fruits', 
'canned jarred vegetables', 'water seltzer sparkling water'])

cluster_means    

   from matplotlib import pyplot as plt
    from matplotlib import cm as cm
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 10)
    cax = ax1.imshow(cluster_means .corr('pearson'), interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    plt.title('Pearson Correlation', size =30)
    labels=cluster_means .columns
    ax1.set_xticklabels(labels,fontsize=20)
    ax1.set_yticklabels(labels,fontsize=20)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax, ticks=[.75,.8,.85,.90,.95,1])
    plt.show()
    
correlation_matrix(cluster_means )






















 










































