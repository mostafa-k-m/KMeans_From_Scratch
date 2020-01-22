import numpy as np
import matplotlib.pylab as plt

class KMeans:
    def __init__(self, data, K, iterations = 100):
        self.K = K
        self.iterations = iterations
        self.data = data
        self.label_index = data.shape[1]

    def distance_calculator(self, data, point):
        step_1 = (data - point)**2
        step_2 = np.sqrt(step_1.sum(axis=1))
        return step_2

    def scatterer(self, data, centroids, hue = "blue"):
        f, axes = plt.subplots(1, 1)
        plt.scatter(data[:,0],data[:,1], c= hue)
        for i in range(0, self.K):
            plt.scatter(centroids[i][0],centroids[i][1],c="red")
        return f

    def centroid_selector(self, data):
        index_c1 = np.random.randint(0,data.shape[0])
        c1 = data[index_c1]
        centroids = [c1]
        distances = self.distance_calculator(data,c1)
        for i in range(0,self.K-1):
            index_cx = np.argmax(distances)
            cx = data[index_cx]
            centroids.append(cx)
            distances = (np.vstack((distances,self.distance_calculator(data,cx)))).min(axis=0)
        return centroids


    def membership(self, data,centroids):
        distances_list = []
        for i in range(0, self.K):
            distances_list.append(self.distance_calculator(data, centroids[i]))
        distance_matrix = np.vstack(distances_list)
        rnk = []
        for i in range(0,data.shape[0]):
            rnk.append(np.argmin(distance_matrix[:,i]))
        return np.asarray(rnk)


    def update_centroids(self, data, centroids, rnk):
        labels = self.membership(data,centroids).reshape(data.shape[0],1)
        data_ = np.hstack((data,rnk.reshape(data.shape[0],1)))
        centroids = []
        for i in range(0,self.K):
            data_m = data_[data_[:,self.label_index]==float(i)]
            new_centroid = np.mean(data_m[:,0:self.label_index],axis=0)
            centroids.append(new_centroid)

        return centroids


    def objective_function(self, data, rnk, centroids):
        data_ = np.hstack((data,rnk.reshape(data.shape[0],1)))

        Avgdist = 0
        for i in range(0,self.K):
            data_m = data_[data_[:,self.label_index]==float(i)]
            Avgdist += sum(self.distance_calculator(data_m[:,:self.label_index], centroids[i])**2)
        return Avgdist


    def stopping_criteria(self, data, centroids_1, centroids_2, rnk_1, rnk_2):
        return self.objective_function(data,rnk_2, centroids_2)< self.objective_function(data,rnk_1, centroids_1)

    def train(self):
        data = self.data
        counter = 0
        centroids = self.centroid_selector(data)
        rnk = self.membership(data, centroids)
        centroids_2 = centroids
        rnk_2 = rnk
        while(counter <self.iterations):
            centroids_2 = self.update_centroids(data, centroids_2, rnk)
            rnk_2 = self.membership(data, centroids_2)
            if self.stopping_criteria(data, centroids, centroids_2, rnk, rnk_2):
                centroids = centroids_2
                rnk = rnk_2
            counter+=1
        f = self.scatterer(data, centroids, hue = rnk)
        f.show()
