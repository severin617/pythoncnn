import tensorflow as tf
import matplotlib as plt
import numpy as np
import math
def getHessian(dim):
    # Each time getHessian is called, we create a new graph so that the default graph (which exists a priori) won't be filled with old ops.
    g = tf.Graph()
    with g.as_default():
        # First create placeholders for inputs: A, b, and c.
        A = tf.placeholder(tf.float32, shape=[dim, dim])
        b = tf.placeholder(tf.float32, shape=[dim, 1])
        c = tf.placeholder(tf.float32, shape=[1])
        # Define our variable
        x = tf.Variable(np.float32(np.repeat(1,dim).reshape(dim,1)))
        # Construct the computational graph for quadratic function: f(x) = 1/2 * x^t A x + b^t x + c
        fx = 0.5 * tf.matmul(tf.matmul(tf.transpose(x), A), x) + tf.matmul(tf.transpose(b), x) + c
        
        # Get gradients of fx with repect to x
        dfx = tf.gradients(fx, x)[0]
        # Compute hessian
        init_op = tf.initialize_all_variables()
        for i in range(dim):
            # Take the i th value of the gradient vector dfx 
            # tf.slice: https://www.tensorflow.org/versions/0.6.0/api_docs/python/array_ops.html#slice
            dfx_i = tf.slice(dfx, begin=[i,0] , size=[1,1])
            # Feed it to tf.gradients to compute the second derivative. 
            # Since x is a vector and dfx_i is a scalar, this will return a vector : [d(dfx_i) / dx_i , ... , d(dfx_n) / dx_n]
            ddfx_i = tf.gradients(dfx_i, x)[0] # whenever we use tf.gradients, make sure you get the actual tensors by putting [0] at the end
            print(ddfx_i) 
            if i == 0: 
                hess = ddfx_i
                print(hess)
            else: 
                hess = tf.concat( [hess, ddfx_i],1) 
            ## Instead of doing this, you can just append each element to a list, and then do tf.pack(list_object) to get the hessian matrix too.
            ## I'll use this alternative in the second example.  
        # Before we execute the graph, we need to initialize all the variables we defined
        init_op = tf.initialize_all_variables()
    
        with tf.Session() as sess:
            sess.run(init_op)
            # We need to feed actual values into the computational graph that we created above. 
            feed_dict = {A: np.float32(np.repeat(2,dim*dim).reshape(dim,dim)), b: np.float32(np.repeat(3,dim).reshape(dim,1)) , c: [1]}
            # sess.run() executes the graph. Here, "hess" will be calculated with the values in "feed_dict".
            print(sess.run(hess, feed_dict))

#getHessian(3)




def getHessianMLP(n_input, n_hidden, n_output):
    batch_size = 10
    # Each time getHessianMLP is called, we create a new graph so that the default graph (which exists a priori) won't be filled with old ops.
    g = tf.Graph()
    with g.as_default():
        # First create placeholders for inputs and targets: x_input, y_target
        x_input = tf.placeholder(tf.float32, shape=[batch_size, n_input])
        y_target = tf.placeholder(tf.float32, shape=[batch_size, n_output])
    
        # Start constructing a computational graph for multilayer perceptron
        ###  Since we want to store parameters as one long vector, we first define our parameters as below and then
        ### reshape it later according to each layer specification.
        parameters = tf.Variable(tf.concat( [tf.truncated_normal([n_input * n_hidden, 1]), tf.zeros([n_hidden, 1]),tf.truncated_normal([n_hidden * n_output,1]), tf.zeros([n_output, 1])],0))
        
        with tf.name_scope("hidden") as scope:
            idx_from = 0 
            weights = tf.reshape(tf.slice(parameters, begin=[idx_from, 0], size=[n_input*n_hidden, 1]), [n_input, n_hidden])
            idx_from = idx_from + n_input*n_hidden
            biases = tf.reshape(tf.slice(parameters, begin=[idx_from, 0], size=[n_hidden, 1]), [n_hidden]) # tf.Variable(tf.truncated_normal([n_hidden]))
            hidden = tf.matmul(x_input, weights) + biases
        with tf.name_scope("linear") as scope:
            idx_from = idx_from + n_hidden
            weights = tf.reshape(tf.slice(parameters, begin=[idx_from, 0], size=[n_hidden*n_output, 1]), [n_hidden, n_output])
            idx_from = idx_from + n_hidden*n_output
            biases = tf.reshape(tf.slice(parameters, begin=[idx_from, 0], size=[n_output, 1]), [n_output]) 
            output = tf.nn.softmax(tf.matmul(hidden, weights) + biases)
        # Define cross entropy loss
        loss = -tf.reduce_sum(y_target * tf.log(output))
        
        ### Note: We can call tf.trainable_variables to get GraphKeys.TRAINABLE_VARIABLES 
        ### because we are using g as our default graph inside the "with" scope. 
        # Get trainable variables
        tvars = tf.trainable_variables()
        # Get gradients of loss with repect to parameters
        dloss_dw = tf.gradients(loss, tvars)[0]
        dim, _ = dloss_dw.get_shape()
        hess = []
        for i in range(dim):
            # tf.slice: https://www.tensorflow.org/versions/0.6.0/api_docs/python/array_ops.html#slice
            dfx_i = tf.slice(dloss_dw, begin=[i,0] , size=[1,1])
            ddfx_i = tf.gradients(dfx_i, parameters)[0] # whenever we use tf.gradients, make sure you get the actual tensors by putting [0] at the end
            hess.append(ddfx_i)
        hess = tf.squeeze(hess) 
        init_op = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(init_op)
            feed_dict = {x_input: np.random.random([batch_size, n_input]), y_target: np.random.random([batch_size, n_output])}
            #print(sess.run(loss, feed_dict))
            print(hess.get_shape())
            print(sess.run(hess, feed_dict))
            print(sess.run(loss, feed_dict))
            print("loss is ", loss)
getHessianMLP(n_input=3,n_hidden=4,n_output=3)
