import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv("turkiye-student-evaluation_generic.csv")
print(dataset.head())
plt.figure(figsize=(20, 6))
sns.countplot(x='class', data=dataset)
plt.show()

# Now we will calculate the mean for each question response
questionmeans = []
classlist = []
questions = []
totalplotdata = pd.DataFrame(list(zip(classlist, questions, questionmeans)), columns=['class', 'questions', 'mean'])
for class_num in range(1, 13):
    class_data = dataset[(dataset["class"] == class_num)]

    questionmeans = []
    classlist = []
    questions = []

    for num in range(1, 13):
        questions.append(num)
    # Class related questions are from Q1 to Q12
    for col in range(5, 17):
        questionmeans.append(class_data.iloc[:, col].mean())
    classlist += 12 * [class_num]
    print(classlist)
    plotdata = pd.DataFrame(list(zip(classlist, questions, questionmeans)), columns=['class', 'questions', 'mean'])
    totalplotdata = totalplotdata.append(plotdata, ignore_index=True)
# The follwoing plot will show us the maximum and minimum class ratings in the data
plt.figure(figsize=(20, 10))
sns.pointplot(x="questions", y="mean", data=totalplotdata, hue="class")
plt.show()
# The above graph shows that we have best ratings for class 2 and the worst for class 4

# Calculate mean for each question response for all the classes.
questionmeans = []
inslist = []
questions = []
totalplotdata = pd.DataFrame(list(zip(inslist, questions, questionmeans)), columns=['ins', 'questions', 'mean'])
for ins_num in range(1, 4):
    ins_data = dataset[(dataset["instr"] == ins_num)]
    questionmeans = []
    inslist = []
    questions = []

    for num in range(13, 29):
        questions.append(num)

    for col in range(17, 33):
        questionmeans.append(ins_data.iloc[:, col].mean())
    inslist += 16 * [ins_num]
    plotdata = pd.DataFrame(list(zip(inslist, questions, questionmeans)), columns=['ins', 'questions', 'mean'])
    totalplotdata = totalplotdata.append(plotdata, ignore_index=True)

plt.figure(figsize=(20, 5))
sns.pointplot(x="questions", y="mean", data=totalplotdata, hue="ins")
plt.show()
"""Based on the graph above we obserce that intructor 3 got the worst rating. We'll now see what course she/he teaches and what course got the least ratings."""

# Calculate mean for each question response for all the classes for Instructor 3
dataset_inst3 = dataset[(dataset["instr"] == 3)]
class_array_for_inst3 = dataset_inst3["class"].unique().tolist()
questionmeans = []
classlist = []
questions = []
totalplotdata = pd.DataFrame(list(zip(classlist, questions, questionmeans)), columns=['class', 'questions', 'mean'])
for class_num in class_array_for_inst3:
    class_data = dataset_inst3[(dataset_inst3["class"] == class_num)]

    questionmeans = []
    classlist = []
    questions = []

    for num in range(1, 13):
        questions.append(num)

    for col in range(5, 17):
        questionmeans.append(class_data.iloc[:, col].mean())
    classlist += 12 * [class_num]

    plotdata = pd.DataFrame(list(zip(classlist, questions, questionmeans)), columns=['class', 'questions', 'mean'])
    totalplotdata = totalplotdata.append(plotdata, ignore_index=True)

plt.figure(figsize=(20, 8))
sns.pointplot(x="questions", y="mean", data=totalplotdata, hue="class")
plt.show()

# This shows that course 4 and course 13 need to revised be instructor 3

# Now we will start clustering
dataset_questions = dataset.iloc[:, 5:33]

# PCA for feature dimensional reduction
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
dataset_questions_pca = pca.fit_transform(dataset_questions)

from sklearn.cluster import KMeans
wcss = []
for i in range(1, 7):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(dataset_questions_pca)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 7), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Based on  the graph, we must go for 3 clusters
kmeans = KMeans(n_clusters=3, init='k-means++')
y_kmeans = kmeans.fit_predict(dataset_questions_pca)

# Visualising the clusters
plt.scatter(dataset_questions_pca[y_kmeans == 0, 0], dataset_questions_pca[y_kmeans == 0, 1], s=100, c='green', label='Cluster 1')
plt.scatter(dataset_questions_pca[y_kmeans == 1, 0], dataset_questions_pca[y_kmeans == 1, 1], s=100, c='yellow', label='Cluster 2')
plt.scatter(dataset_questions_pca[y_kmeans == 2, 0], dataset_questions_pca[y_kmeans == 2, 1], s=100, c='red', label='Cluster 3')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='blue', label='Centroids')
plt.title('Clusters of students')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.show()
# Thus we obtain 3 clusters of students who have given negative, neutral and positive feedback
import collections
# print(collections.Counter(y_kmeans))

# Using the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(dataset_questions_pca, method='ward'))
plt.title('Dendrogram')
plt.xlabel('questions')
plt.ylabel('Euclidean distances')
# plt.show()
