import numpy as np
import math as math 

def random_mini_batches3(X, Y, Z, mini_batch_size = 64):
    
    m = X.shape[0]                  # number of training examples
    mini_batches = []
        
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :]
    shuffled_Y = Y[permutation, :].reshape((m,1))
    shuffled_Z = Z[permutation, :]
    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch_X = shuffled_X[range(k * mini_batch_size, (k+1)*mini_batch_size), :]
        mini_batch_Y = shuffled_Y[range(k * mini_batch_size, (k+1)*mini_batch_size), :].reshape((mini_batch_size,1))
        mini_batch_Z = shuffled_Z[range(k * mini_batch_size, (k+1)*mini_batch_size), :]
        
        ### END CODE HERE ###
        mini_batch = (mini_batch_X, mini_batch_Y, mini_batch_Z)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch_X = shuffled_X[range(num_complete_minibatches * mini_batch_size, m), :]
        mini_batch_Y = shuffled_Y[range(num_complete_minibatches * mini_batch_size, m), :]
        mini_batch_Z = shuffled_Z[range(num_complete_minibatches * mini_batch_size, m), :]
        
        ### END CODE HERE ###
        mini_batch = (mini_batch_X, mini_batch_Y, mini_batch_Z)
        mini_batches.append(mini_batch)
    
    return mini_batches

def random_mini_batches2(X, Y, mini_batch_size = 64):
    
    m = X.shape[0]                  # number of training examples
    mini_batches = []
        
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :]
    shuffled_Y = Y[permutation, :].reshape((m,1))
    
    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch_X = shuffled_X[range(k * mini_batch_size, (k+1)*mini_batch_size), :]
        mini_batch_Y = shuffled_Y[range(k * mini_batch_size, (k+1)*mini_batch_size), :].reshape((mini_batch_size,1))
        
        
        ### END CODE HERE ###
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch_X = shuffled_X[range(num_complete_minibatches * mini_batch_size, m), :]
        mini_batch_Y = shuffled_Y[range(num_complete_minibatches * mini_batch_size, m), :]
        
        ### END CODE HERE ###
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

def random_mini_batches_perm(X, Y, mini_batch_size = 64):
    
    m = X.shape[0]                  # number of training examples
    mini_batches = []
    vertexsq = X.shape[1]
    vertex = int(np.sqrt(vertexsq))
    
    X = X.reshape((-1, vertex, vertex))
    
    if True:
        for i in range(m):
            permutation = list(np.random.permutation(vertex))
            graph = X[i,:,:]
            graph = graph[permutation,:]
            graph = graph[:,permutation]
            X[i,:,:] = graph 
        
    X = X.reshape((-1,vertexsq))
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :]
    shuffled_Y = Y[permutation, :].reshape((m,1))
    
    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch_X = shuffled_X[range(k * mini_batch_size, (k+1)*mini_batch_size), :]
        mini_batch_Y = shuffled_Y[range(k * mini_batch_size, (k+1)*mini_batch_size), :].reshape((mini_batch_size,1))
        
        
        ### END CODE HERE ###
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch_X = shuffled_X[range(num_complete_minibatches * mini_batch_size, m), :]
        mini_batch_Y = shuffled_Y[range(num_complete_minibatches * mini_batch_size, m), :]
        
        ### END CODE HERE ###
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

def random_mini_batches_conv_perm(X, Y, mini_batch_size = 64):
    
    m = X.shape[0]                  # number of training examples
    mini_batches = []
    vertex = X.shape[1]
    
    if True:
        for i in range(m):
            permutation = list(np.random.permutation(vertex))
            graph = X[i,:,:]
            graph = graph[permutation,:]
            graph = graph[:,permutation]
            X[i,:,:] = graph 
        
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :]
    shuffled_Y = Y[permutation, :].reshape((m,1))
    
    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch_X = shuffled_X[range(k * mini_batch_size, (k+1)*mini_batch_size), :]
        mini_batch_Y = shuffled_Y[range(k * mini_batch_size, (k+1)*mini_batch_size), :].reshape((mini_batch_size,1))
        
        
        ### END CODE HERE ###
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch_X = shuffled_X[range(num_complete_minibatches * mini_batch_size, m), :]
        mini_batch_Y = shuffled_Y[range(num_complete_minibatches * mini_batch_size, m), :]
        
        ### END CODE HERE ###
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches