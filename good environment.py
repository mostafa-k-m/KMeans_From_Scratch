import numpy as np
import matplotlib.pylab as plt

def distance_calculator(data, point):
    step_1 = (data - point)**2
    step_2 = np.sqrt(step_1.sum(axis=1))
    return step_2

def scatterer(data, m1, m2, m3):
    plt.scatter(data[:,0],data[:,1])
    plt.scatter(m1[0],m1[1],c="red")
    plt.scatter(m2[0],m2[1],c="red")
    plt.scatter(m3[0],m3[1],c="red")
    plt.show()

def Random_Select(data):
    """
    This Function should implement step 1 in slide 6, Lec 6
    Input: Dataset
    Return: 3-means
    """

    #1. Choose First Center randomly
    index_m1 = np.random.randint(0,data.shape[0])
    m1 = data[index_m1]

    #2. select the second one
    distance_to_m1 = distance_calculator(data,m1)
    index_m2 = np.argmax(distance_to_m1)
    m2 = data[index_m2]

    #3. select the third one
    data_left = np.delete(data,(index_m1,index_m2),axis = 0)
    index_m3 = np.argmax((distance_calculator(data_left,m1)*distance_calculator(data_left,m2)))
    m3 = data_left[index_m3]

    return m1,m2,m3


def Membership(data,m1,m2,m3):
    """
    This function should implement step 2 in slide 6, Lec 6
    Possible values for rnk will be 0 or one ore Two
    Input: Dataset, 3-means, and memership vector
    Return: New membership vector
    """
    distance_matrix = np.vstack((distance_calculator(data,m1), distance_calculator(data,m2), distance_calculator(data,m3)))
    rnk = []
    for i in range(0,data.shape[0]):
        rnk.append(np.argmin(distance_matrix[:,i]))
    return np.asarray(rnk)


def Update_Means(data,rnk):
    """
    This function should implement step 3 in slide 6, Lec 6

    Input: Dataset and memership vector
    Return: updated 3-means
    """
    labels = Membership(data,m1,m2,m3).reshape(data.shape[0],1)
    data_ = np.hstack((data,rnk.reshape(data.shape[0],1)))
    means = []
    for i in range(0,4):
        data_m = data_[data_[:,2]==float(i)]
        new_mean = np.mean(data_m[:,0:2],axis=0)
        means.append(new_mean)

    return means[0], means[1], means[2]


def Objective_Function(data,rnk,m1,m2,m3):
    """
    This function should implement  objective Function in slide 5,Lec 6

    Input: Dataset, memership vector, and 3-means
    Return: Total Average Distance
    """
    data_ = np.hstack((data,rnk.reshape(data.shape[0],1)))
    means = [m1, m2, m3]
    Avgdist = 0
    for i in range(0,3):
        data_m = data_[data_[:,2]==float(i)]
        Avgdist += sum(distance_calculator(data_m[:,:2], means[i])**2)
    return Avgdist


def Stopping_Criteria(m1_b,m2_b,m3_b,m1,m2,m3):
    """
    This function should Test Convergence

    Input:  3-means before updating and after them.
    Return: return True or  return False
    """
    rnk_b = Membership(data,m1_b,m2_b,m3_b)
    rnk = Membership(data,m1,m2,m3)
    return Objective_Function(data,rnk,m1,m2,m3)< Objective_Function(data,rnk_b,m1_b,m2_b,m3_b)



#________________________Main Code Statrs here___________________________________

data=np.loadtxt("Data.txt") # Load Data
m1, m2, m3 = Random_Select(data)
scatterer(data, m1, m2, m3)

rnk = Membership(data,m1,m2,m3).reshape(data.shape[0],1)
m1_b, m2_b, m3_b = Update_Means(data,rnk)
scatterer(data, m1_b, m2_b, m3_b)

Stopping_Criteria(m1_b,m2_b,m3_b,m1,m2,m3)

m1_b,m2_b,m3_b = Random_Select(data)

counter = 0
while(counter <100):
        # --------------------Intialization Step 1------------------------------#
        m1,m2,m3 = Random_Select(data)

        rnk = Membership(data,m1,m2,m3)

        m1, m2, m3 = Update_Means(data,rnk)

        if Stopping_Criteria(m1_b,m2_b,m3_b,m1,m2,m3):
            m1_b = m1
            m2_b = m2
            m3_b = m3
        counter+=1
