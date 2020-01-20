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

    def m_selector(self, data):
        index_c1 = np.random.randint(0,data.shape[0])
        c1 = data[index_c1]
        distance_to_c1 = self.distance_calculator(data,c1)
        index_c2 = np.argmax(distance_to_c1)
        c2 = data[index_c2]
        data = np.delete(data,(index_c1,index_c2),axis = 0)
        centroids = [c1, c2]
        for i in range(0,self.K-2):
            distances_list = []
            for j in range(0, i+2):
                distances_list.append(self.distance_calculator(data,centroids[j]))
            distances = np.prod(distances_list)
            index_cx = np.argmax(distances)
            cx = data[index_cx]
            data = np.delete(data,(index_cx),axis = 0)
            centroids.append(cx)
        return centroids


    def Membership(self, data,centroids):
        distances_list = []
        for i in range(0, self.K):
            distances_list.append(self.distance_calculator(data, centroids[i]))
        distance_matrix = np.vstack(distances_list)
        rnk = []
        for i in range(0,data.shape[0]):
            rnk.append(np.argmin(distance_matrix[:,i]))
        return np.asarray(rnk)


    def Update_centroids(self, data, centroids, rnk):
        labels = self.Membership(data,centroids).reshape(data.shape[0],1)
        data_ = np.hstack((data,rnk.reshape(data.shape[0],1)))
        centroids = []
        for i in range(0,self.K):
            data_m = data_[data_[:,self.label_index]==float(i)]
            new_centroid = np.mean(data_m[:,0:self.label_index],axis=0)
            centroids.append(new_centroid)

        return centroids


    def Objective_Function(self, data, rnk, centroids):
        data_ = np.hstack((data,rnk.reshape(data.shape[0],1)))

        Avgdist = 0
        for i in range(0,self.K):
            data_m = data_[data_[:,self.label_index]==float(i)]
            Avgdist += sum(self.distance_calculator(data_m[:,:self.label_index], centroids[i])**2)
        return Avgdist


    def Stopping_Criteria(self, data, centroids_1, centroids_2):
        rnk_1 = self.Membership(data, centroids_1)
        rnk_2 = self.Membership(data, centroids_2)
        return self.Objective_Function(data,rnk_2, centroids_2)< self.Objective_Function(data,rnk_1, centroids_1)

    def train(self):
        data = self.data
        counter = 0
        while(counter <self.iterations):
            centroids = self.m_selector(data)
            rnk = self.Membership(data, centroids)
            centroids_2 = self.Update_centroids(data, centroids, rnk)
            if self.Stopping_Criteria(data, centroids, centroids_2):
                centroids = centroids_2
            counter+=1

        f = self.scatterer(data, centroids, hue = rnk)
        f.show()

class_ = KMeans(data, 3)
class_.train()
