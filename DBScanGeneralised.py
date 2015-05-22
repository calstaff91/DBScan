#Carry out a DBScan clustering process on coordinates


#Import required modules
import pandas as pd
import numpy as np
import os
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
#-----------------------------------------------------------------------------

#Import coordinates from csv to a dictionary
os.chdir('Directory Name Here')
database1 = pd.read_csv("File Here.csv")
data_as_dict = pd.DataFrame({'X Var': database1['X Variable Name Here'],
                     'Y Var': database1['Y Variable Name Here']})
#-----------------------------------------------------------------------------

#Iterates through dictionary and turns into list, with each entry having and X and Y variable
data = [dict(r.iteritems()) for _, r in data_as_dict.iterrows()]
le = preprocessing.LabelEncoder()
data = DictVectorizer(sparse=False).fit_transform(data)
#-----------------------------------------------------------------------------

#Runs DBScan algorithm. The variable 'eps', stands for epsilon, which specifies how close points 
#should be to each other to be considered a part of a cluster
db = DBSCAN(eps=???).fit(data)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
#-----------------------------------------------------------------------------

#Returns the number of different clusters in the data
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
#-----------------------------------------------------------------------------

#Returns a series of metrics relating to the dataset and clusters
print('Estimated number of clusters: %d' % n_clusters_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(data, labels))
print("Completeness: %0.3f" % metrics.completeness_score(data, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(data, labels))
print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(data, labels))
print("Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(data, labels))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(data, labels))
#-----------------------------------------------------------------------------

#Creates a dictionary containing location and cluster membership and creates a csv from that
final_dict = {'Longitude': database1['Longitude'],
            'Latitude': database1['Latitude'],
            'Cluster Membership': labels}
pd.DataFrame(final_dict).to_csv(r"Location and Name of New csv Here.csv",encoding="utf-8")