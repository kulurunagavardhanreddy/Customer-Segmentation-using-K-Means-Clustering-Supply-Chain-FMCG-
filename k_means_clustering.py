# K - Means Clustering

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import base64
# Streamlit app
st.markdown("<h1 style='color: magenta;'>Customer Segmentation using K-Means Clustering</h1>", unsafe_allow_html=True)

def img(image_path):
    with open(image_path, "rb") as image_file:
        b64_image = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{b64_image}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    
# Set your image path here
img(r'C:\Users\nag15\.spyder-py3\Spyder Projects\Machine Learning\Clustering\center-2064919.jpg')
    
# Importing the dataset
data = pd.read_csv(r"C:\Users\nag15\.spyder-py3\Spyder Projects\Machine Learning\Clustering\Mall_Customers.csv")

st.write("### Dataset Preview:")
st.write(data.head())

X = data.iloc[:, [3,4]].values

# Using elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
#we are going to findout the optimal number of cluster & we have to use the elbow method
st.write("### Elbow Method to Determine Optimal Number of Clusters")

wcss = []

#to plot the elbow metod we have to compute WCSS for 10 different number of cluster since we gonna have 10 iteration
#we are going to write a for loop 

#to create a list of 10 different wcss for the10 number of clusters 
#thats why we have to initialise wcss[] & we start our loop 

#we choose 1-11 becuase the 11 bound is excluded & we want 10 wcss however the first bound is included so hear i = 1,2,3 to 10
#now in each iteration of loop we are going to do 2 things  1st we are going to fit the k-means algorithm into our data x and we are going to compute WCSS
#Now lets fit kmean to our data x
#now eare 

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
fig, ax = plt.subplots()
ax.plot(range(1, 11), wcss)
ax.set_title('The Elbow Method')
ax.set_xlabel('Number of clusters')
ax.set_ylabel('WCSS')
plt.show()
st.pyplot(fig) 

#wcss we have very good parameter called inertia_ credit goes to sklearn , that computes the sum of square , formula it will compute

# Asking the user to select the number of clusters based on the elbow method
n_clusters = st.slider('Select the number of clusters', 1, 10, 5)


# Training the K-Means model on the dataset
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=0)
y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters
st.write("### Visualizing the Clusters:")

fig, ax = plt.subplots()

ax.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1')
ax.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')
ax.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, c='green', label='Cluster 3')
ax.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=100, c='cyan', label='Cluster 4')
ax.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=100, c='magenta', label='Cluster 5')

# Adding centroids to the plot
ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')

# Setting titles and labels
ax.set_title('Clusters of customers')
ax.set_xlabel('Annual Income (k$)')
ax.set_ylabel('Spending Score (1-100)')
ax.legend()

# Display the plot in Streamlit
st.pyplot(fig)

# Displaying cluster information
st.write("### Cluster Information:")
data['Cluster'] = y_kmeans
st.write(data[['Annual Income (k$)', 'Spending Score (1-100)', 'Cluster']])

# Allowing user to download clustered data
csv = data.to_csv(index=False)
st.download_button("Download Clustered Data", data=csv, file_name="Clustered_Customers.csv")
