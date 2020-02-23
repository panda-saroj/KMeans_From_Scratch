
#All Required Imports 
import numpy as np
from sklearn import datasets
from math import sqrt
from statistics import mode, _counts

NO_CLASS = -9999    # If some cluster has no data elements, assign its class lable as NO_CLASS

from sklearn import preprocessing

import random



class KMeans:
    def __init__(self, n_clusters, dist_func = 'euclidean',  max_iter=1000):
        self.k = n_clusters
        self.dist_func = dist_func
        self.max_iter = max_iter
             
        self.clusters = {}                 # Clusters that hold the indexes of dataset
        self.cluster_actual_classes = {}    # Clusters with Actual class lables
        
        self.cluster_class_lable = {}    # Class Lable for the cluster based on max class elements in the cluster
        
        for i in range(n_clusters):
            self.clusters[i] = []
            self.cluster_actual_classes[i] = []
            self.cluster_class_lable[i] = []
            
            
    
    # Assigns class Lable to the cluster based on the majority data element class in the cluster
    def get_cluster_class_lable(self, cluster):
        
        class_count_table = _counts(cluster)
        len_table = len(class_count_table)

        if len_table == 1:                     # If only one class is majority
            cluster_class_lable = mode(cluster) # mode would return the majority class element value
        else:
            class_list = []
            for i in range(len_table):
                class_list.append(class_count_table[i][0])
            cluster_class_lable = min(class_list) # Return the Minimum class lable
        return cluster_class_lable
        
        
    def lables(self):
        print(self.cluster_class_lable)
        
        
    def fit(self, X):
        
        # Setup a blank centroid dictionary
        self.centroids = {}
        
        # Normalize the data
        X = preprocessing.normalize(X)
        
        # From the dataset select K number of data points as initial cluster
        K_samples = random.sample(list(X), self.k)
        
        for i in range (len(K_samples)):
            self.centroids[i] = K_samples[i]
                 
#         print(" Initial Centroids ")
#         print(self.centroids)
        
        
        for iteration in range(self.max_iter):

            self.class_features = {}               # Holds the features of the data points assigned to cluster
            for i in range(self.k):
                self.clusters[i] = []              # Clusters to hold the indexes of the data points
                
                # Dictionary to hold features of data points for new centroid calculation
                self.class_features[i] = []        
            
            x_index = 0   # To keep track of the indexes of data points in the original dataset
            for features in X:
                
                distances = []
                
                # find the distance between the point and cluster centroids 
                for centroid in self.centroids:
                    if(self.dist_func == "euclidean"):
                        distances.append(self.euclidean_distance(self.centroids[centroid], features))
                    elif(self.dist_func == "manhattan"):
                        distances.append(self.manhattan_distance(self.centroids[centroid], features)) 
                    else:
                        print("Got invalid distance measure")
                        pass
                
                # choose the nearest centroid
                cluster_class = distances.index(min(distances))
                self.clusters[cluster_class].append(x_index)
                
                self.class_features[cluster_class].append(features)
                x_index += 1
            
            
            previous_centroid = dict(self.centroids)
            
            # Calculate the New centroid of each cluster from the feature values in that cluster
            for cluster_class in self.clusters:
                # If there are data elements in the cluster then compute new centroid for the cluster
                # else borrow the previous centroid
                
                if (len(self.clusters[cluster_class]) > 0):
                    
                    self.centroids[cluster_class] = np.average(self.class_features[cluster_class], axis = 0)
                else:
                    self.centroids[cluster_class] = previous_centroid[cluster_class]
            
            # Calculate the change in cluster centrods
            # If the change is 0  KMeans has converged
            # and we exit the loop
            
            centroid_change_sum = 0
            
#             print("previous_centroid :", previous_centroid)
#             print("curr_centroid :", self.centroids)
            for centroid in self.centroids:
                original_centroid = previous_centroid[centroid]
                curr_centroid = self.centroids[centroid]
                
                centroid_change = np.sqrt(np.sum(np.square(curr_centroid - original_centroid)))
                centroid_change_sum += centroid_change
           
            
            if(centroid_change_sum ==0):
                print("No more Change in Centroid, KMeans Converged")
                print("Total number of iterations performed :", iteration)
                         
                break
                                          
        
# FUnction to calculate the accuracy of the clusters found from the ground truth Y of those data points        
    def accuracy(self, Y):

        total_data_points = len(Y)    # Total number of data points
        
        total_correct_cluster_points = 0 # total number of data points that were in correct clusters.
        
        
        for cluster in range(self.k):
            # Initialize each dictionary entry as empty list
            self.cluster_actual_classes[cluster] = []
            self.cluster_class_lable[cluster] = []
            
            # Go over the clusters which holds the indexes of the data points and
            # add their actual class lables to self.cluster_actual_classes
            for index in self.clusters[cluster]:
                self.cluster_actual_classes[cluster].append(Y[index])
            
            # If there are elements in the cluster then find the majority of the class in the cluster
            # That will be assigned as class lable of that cluster
            
            if(len(self.cluster_actual_classes[cluster]) > 0):
                class_lable = self.get_cluster_class_lable(self.cluster_actual_classes[cluster])
                self.cluster_class_lable[cluster] = class_lable
            else:
                self.cluster_class_lable[cluster] = NO_CLASS # if the cluster is empty
                
            
            # As Cluster class lable means majority of elements of that class belong to that class and can be considered
            # as correctly assigned to clusters and use that number to Calculate the correctly classified elements for
            # Accuracy calculation
            
            if(self.cluster_class_lable[cluster] != NO_CLASS):
                total_correct_cluster_points += self.cluster_actual_classes[cluster].count(class_lable)
            
           
#         print("\n Cluster Actual Classes")
#         print(self.cluster_actual_classes)
        
#         print("\n Cluster Class Lables")
#         print(self.cluster_class_lable)
        
        accuracy = total_correct_cluster_points/total_data_points
        
        return accuracy
                    
  
    
    # Calculates the Euclidian distance between two data points
    def euclidean_distance(self, data1, data2):
        distance = 0.0

        for i in range(len(data1)):
            distance += (data1[i] - data2[i]) * (data1[i] - data2[i])

        return sqrt(distance)
    
    # Calculates the Manhattan distance between two data points
    def manhattan_distance(self, data1, data2):
        distance = 0.0

        for i in range(len(data1)):
            distance += abs(data1[i] - data2[i]) 

        return distance
    

# Testing KMeans with IRIS and Breast Cancer Wisconson Datasets

#Load the IRIS dataset
iris_data = datasets.load_iris()
#Get the Iris Data and Lables
iris_data_part = iris_data.data
iris_lable_part = iris_data.target

print("#################################################################")
print("IRIS Dataset Testing")
model = KMeans(3)

from datetime import datetime

tstart = datetime.now()

model.fit(iris_data_part)
tend = datetime.now()

print("Clustering Start Time: ", tstart)
print("Clustering End Time:  ", tend)

print("Clustering Time ", tend - tstart)

accuracy = model.accuracy(iris_lable_part)

print("Accuracy = ", accuracy)
print("#################################################################")

print("\n\n")


# Load Breast Cancer Wisconson Dataset
bc_data = datasets.load_breast_cancer()

#Get the BC Data and Lables
bc_data_part = bc_data.data
bc_lable_part = bc_data.target

print("#################################################################")
print("Breaset Cancer Dataset Testing")
model = KMeans(2)
from datetime import datetime

tstart = datetime.now()


model.fit(bc_data_part)
tend = datetime.now()

print("Clustering Start Time: ", tstart)
print("Clustering End Time:  ", tend)

print("Clustering Time ", tend - tstart)
accuracy = model.accuracy(bc_lable_part)

print("Accuracy = ", accuracy)
print("#################################################################")






