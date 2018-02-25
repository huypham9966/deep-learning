import numpy as np 

np.random.seed(5)
def generate_graphs(num_eg, vertex, edge_dens, label):
    data = {}
    G = np.zeros((num_eg, vertex, vertex))
    """for i in range(vertex):
        for j in range(i,vertex):
            G[i,j,:] = np.random.randint(2) * np.ones((1,1,num_eg))
        for j in range(i):
            G[i,j,:] = G[j,i,:]"""
    #for i in range(num_eg):
    #    G[:,:,i] = np.random.randint(2) * np.ones((vertex,vertex,1))
    for i in range(vertex):
        for j in range(i,vertex):
            X = np.random.rand(num_eg) < edge_dens
            G[:,i,j] = X.astype(float)
        for j in range(i):
            G[:,i,j] = G[:,j,i]
    data["graphs"] = G
    data["labels"] = np.ones((num_eg,1)) * label 
    return data

def one_random_graph(vertex, dens1, dens2):
    permutation = list(np.random.permutation(vertex))
    
    num1 = int(vertex/2)
    num2 = vertex - num1
    
    G = np.zeros((vertex,vertex))
    
    for i in range(num1):
        for j in range(i):
            G[i,j] = np.random.rand() < dens1 
            G[j,i] = G[i,j] 
    
    for i in range(num1,vertex):
        for j in range(i,vertex):
            G[i,j] = np.random.rand() < dens2 
            G[j,i] = G[i,j] 
    
    for i in range(num1):
        for j in range(num1,vertex):
            G[i,j] = np.random.rand() < 1/2
            G[j,i] = G[i,j]
    
    G = G[:, permutation]
    G = G[permutation, :] 
    return G 
    
def generate_ensemble(num_eg, vertex, edge_dens):
    data = {}
    data["graphs"] = np.zeros((np.sum(num_eg), vertex, vertex))
    data["labels"] = np.zeros((np.sum(num_eg), 1))
    curindex = 0 
    for i in range(len(num_eg)):
        ensemble = generate_graphs(num_eg[i], vertex, edge_dens[i], i)
        data["graphs"][curindex:(curindex + num_eg[i]), :, :] = ensemble["graphs"]
        data["labels"][curindex:(curindex + num_eg[i]), :] = ensemble["labels"]
        curindex = curindex + num_eg[i]
    return data 

def generate_ensemble_v2(num_eg, vertex):
    data = {}
    data["graphs"] = np.zeros((num_eg, vertex, vertex))
    data["labels"] = np.zeros((num_eg, 1))
    
    num = int(num_eg/2) 
    
    for i in range(num):
        data["graphs"][i, :, :] = one_random_graph(vertex, 1/3, 2/3)
        data["labels"][i, :] = 1
        
    ensemble = generate_graphs(num, vertex, 1/2, 0)
    data["graphs"][num:num_eg, :, :] = ensemble["graphs"]
    data["labels"][num:num_eg, :] = ensemble["labels"]
    return data 
