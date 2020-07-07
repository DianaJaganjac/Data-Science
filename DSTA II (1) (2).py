#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

fname = "GPPD.csv"

df = pd.read_csv(fname, usecols=['country_long', 'capacity_mw', 'latitude', 'longitude', 'primary_fuel', 'R_NR', 'commissioning_year','estimated_generation_gwh'])
df = df.dropna()

df.head()


# In[2]:


country = pd.factorize(df["country_long"])
df["country_long"] = list(country[0])

fuel = pd.factorize(df["primary_fuel"])
df["primary_fuel"] = list(fuel[0])

df.shape


# In[3]:


corrMatrix = df.corr()
sns.heatmap(corrMatrix, annot=True)


# In[4]:


df = pd.read_csv(fname, usecols=['capacity_mw', 'latitude', 'longitude', 'primary_fuel'])
df = df.dropna()
df.shape


# In[5]:


df.head()


# In[6]:


scatter_mat = sns.pairplot(df, hue = 'primary_fuel')


# In[7]:


df = pd.read_csv(fname, usecols=['capacity_mw', 'latitude', 'longitude', 'primary_fuel'])

ax = sns.scatterplot(x="longitude", y="capacity_mw", hue="primary_fuel", legend = "full", data=df)

ax.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)


# In[8]:


ax = sns.scatterplot(x="latitude", y="capacity_mw", hue="primary_fuel", legend = "full", data=df)

ax.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)


# In[9]:


ax_1 = sns.scatterplot(x="longitude", y="latitude", hue="primary_fuel", legend = "full", data=df)

ax_1.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)


# In[10]:


df = pd.read_csv(fname, usecols=['capacity_mw', 'latitude', 'longitude', 'R_NR'])

ax = sns.scatterplot(x="longitude", y="capacity_mw", hue="R_NR", legend = "full", data=df)

ax.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)


# In[11]:


ax = sns.scatterplot(x="latitude", y="capacity_mw", hue="R_NR", legend = "full", data=df)

ax.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)


# In[12]:


ax_1 = sns.scatterplot(x="longitude", y="latitude", hue="R_NR", legend = "full", data=df)

ax_1.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)


# In[13]:


from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from matplotlib.cm import gist_rainbow
from sklearn.preprocessing import StandardScaler

#I select the features
features = ['capacity_mw', 'latitude', 'longitude']
x = df.loc[:, features].values

#I scale features to standardise results
x = StandardScaler().fit_transform(x)

#I plot the 2D axes
fig = plt.figure(1, figsize=(8, 6))

# I set the 'R_NR' dimension to the colours variable 
colours = df["R_NR"]

# I perform PCA 
X_reduced = PCA(n_components=2).fit_transform(x)

#I produce a 2-D graph of the PCA with all relevant labels 
graph = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c = colours, cmap = plt.cm.summer, edgecolor='k', s=40)

plt.title("First two PCA directions", fontsize = 15)
plt.xlabel('First Eigenvector', fontsize = 10)
plt.ylabel('Second Eigenvector', fontsize = 10)


cbar = plt.colorbar(graph, ticks=[0, 1])
cbar.ax.set_yticklabels(["Non-Renewable", "Renewable"])
    
plt.show()


# In[14]:


y = df['R_NR']
x = StandardScaler().fit_transform(x)
model = PCA(n_components=3)
model.fit(x)
transformed = model.transform(x)
print(transformed)
print(transformed.shape)
print(model.components_)


# In[15]:


pca = PCA(n_components=3)
pca.fit(x)
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_)
print(model.explained_variance_)
plt.xticks(features)
plt.ylabel("Variance")
plt.xlabel("PCA Feature")
plt.show()


# In[ ]:




