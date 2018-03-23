import numpy as np 
import networkx as nx 
import node2vec as n2v

#np.random.seed(5)
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

def generate_nx_graphs(num_eg, vertex, edge_dens, label):
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
    data["graphs"] = nx.from_numpy_matrix(G)
    data["labels"] = np.ones((num_eg,1)) * label 
    return data

def node2vec_graphs(num_eg, vertex, edge_dens, label):
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
    graph = nx.from_numpy_matrix(G)
    data["labels"] = np.ones((num_eg,1)) * label 
    node2vec = Node2Vec(graph, dimensions=64, walk_length=30, num_walks=200, workers=4)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    graphdat = np.zeros((64, vertex))
    for i in range(vertex):
        graphdat[:,i] = model[i]
    data["graphs"] = graphdat 
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

def one_random_graph_label(vertex, dens1, dens2):
    permutation = list(np.random.permutation(vertex))
    
    num1 = int(vertex/2)
    num2 = vertex - num1
    
    G = np.zeros((vertex,vertex))
    label = np.zeros((vertex,1))
    label[0:num1,:] = np.ones((num1,1))
    
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
    label = label[permutation,:] 
    return G, label 

def one_random_graph_hard(vertex):
    permutation = list(np.random.permutation(vertex))
    num1 = int(vertex/6)
    num2 = vertex - 5 * num1 
    G = np.zeros((vertex,vertex))
    for i in range(vertex):
        for j in range(i,vertex):
            G[i,j] = np.random.rand() < 1/3 
            G[j,i] = G[i,j] 
    for i in range(num1):
        for j in range(num1,(num1*2)):
            G[i,j] = np.random.rand() < 2/3 
            G[j,i] = G[i,j] 
        for j in range(num2,vertex):
            G[i,j] = np.random.rand() < 2/3 
            G[j,i] = G[i,j] 
    for i in range(num1, 2*num1):
        for j in range((2*num1),(num1*3)):
            G[i,j] = np.random.rand() < 2/3 
            G[j,i] = G[i,j] 
    for i in range(2*num1, 3*num1):
        for j in range((3*num1),(num1*4)):
            G[i,j] = np.random.rand() < 2/3 
            G[j,i] = G[i,j]    
    for i in range(3*num1, 4*num1):
        for j in range((4*num1),num2):
            G[i,j] = np.random.rand() < 2/3 
            G[j,i] = G[i,j]
    for i in range(4*num1, num2):
        for j in range((num2),(vertex)):
            G[i,j] = np.random.rand() < 2/3 
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

def generate_ensemble_v3(num_eg, vertex):
    data = {}
    data["graphs"] = np.zeros((num_eg, vertex, vertex))
    data["labels"] = np.zeros((num_eg, 1))
    
    num = int(num_eg/2) 
    
    for i in range(num):
        data["graphs"][i, :, :] = one_random_graph_hard(vertex)
        data["labels"][i, :] = 1
        
    ensemble = generate_graphs(num, vertex, 4/9, 0)
    data["graphs"][num:num_eg, :, :] = ensemble["graphs"]
    data["labels"][num:num_eg, :] = ensemble["labels"]
    return data 

def generate_emsemble_n2v(num_eg, vertex):
    data = {}
    data["graphs"] = np.zeros((num_eg, 64, vertex))
    data["labels"] = np.zeros((num_eg, 1))
    
    num = int(num_eg/2) 
    
    for i in range(num):
        data["graphs"][i, :, :] = node2vec_graphs(num_eg, vertex, 1/2, label)["graphs"]
        data["labels"][i, :] = node2vec_graphs(num_eg, vertex, 1/2, label)["labels"]
        
    return data 

def augment(data):
    num_eg = data["graphs"].shape[0]
    vertex = data["graphs"].shape[1]
    newdata = np.zeros((num_eg, vertex + 1, vertex + 1))
    newdata[:,0:vertex,0:vertex] = data["graphs"]
    newdata[:,vertex,:] = np.ones((num_eg,vertex+1))
    newdata[:,:,vertex] = np.ones((num_eg,vertex+1))
    data["graphs"] = newdata
    return data 

def generate_ensemble_v2_label(num_eg, vertex):
    data = {}
    data["graphs"] = np.zeros((num_eg, vertex, vertex))
    data["labels"] = np.zeros((num_eg, 1))
    data["vertex"] = np.zeros((num_eg, vertex, 1))
    num = int(num_eg/2) 
    
    for i in range(num):
        data["graphs"][i, :, :], data["vertex"][i, :] = one_random_graph_label(vertex, 1/3, 2/3)
        data["labels"][i, :] = 1
    ensemble = generate_graphs(num, vertex, 1/2, 0)
    data["graphs"][num:num_eg, :, :] = ensemble["graphs"]
    data["labels"][num:num_eg, :] = ensemble["labels"]
    return data 

def one_random_clique_label(vertex, dens):
    permutation = list(np.random.permutation(vertex))
    
    num1 = int(np.sqrt(vertex)/2)
    print(num1)
    G = np.zeros((vertex,vertex))
    label = np.zeros((vertex,1))
    label[0:num1,:] = np.ones((num1,1))
    
    for i in range(vertex):
        for j in range(i):
            G[i,j] = np.random.rand() < dens 
            G[j,i] = G[i,j] 
    
    for i in range(num1):
        for j in range(i):
            G[i,j] = 1
            G[j,i] = 1 
    
    G = G[:, permutation]
    G = G[permutation, :] 
    label = label[permutation,:] 
    return G, label 

def generate_ensemble_clique_label(num_eg, vertex):
    data = {}
    data["graphs"] = np.zeros((num_eg, vertex, vertex))
    data["labels"] = np.zeros((num_eg, 1))
    data["vertex"] = np.zeros((num_eg, vertex, 1))
    num = int(num_eg/2) 
    
    for i in range(num):
        data["graphs"][i, :, :], data["vertex"][i, :] = one_random_clique_label(vertex, 1/2)
        data["labels"][i, :] = 1
    ensemble = generate_graphs(num, vertex, 0.5, 0)
    data["graphs"][num:num_eg, :, :] = ensemble["graphs"]
    data["labels"][num:num_eg, :] = ensemble["labels"]
    return data

def generate_ensemble_same_label(num_eg, vertex):
    data = {}
    data["graphs"] = np.zeros((num_eg, vertex, vertex))
    data["labels"] = np.zeros((num_eg, 1))
    data["vertex"] = np.zeros((num_eg, vertex, 1))
    num = int(num_eg/2) 
    
    ensemble0 = generate_graphs(num, vertex, 0.5, 1)
    data["graphs"][range(num), :, :] = ensemble0["graphs"]
    data["labels"][range(num), :] = 1
    ensemble = generate_graphs(num, vertex, 0.5, 0)
    data["graphs"][num:num_eg, :, :] = ensemble["graphs"]
    data["labels"][num:num_eg, :] = ensemble["labels"]
    return data