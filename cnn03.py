#!/usr/bin/python

import sys
import tensorflow as tf
from tensorflow import keras
import numpy as np
import itertools
import time
 
training=True
save_hessian=True
global_batch_size = 7500
modelname="cnn03"


###### some functions

def jacobian(batchstart=0,batchend=256):
	f=open(modelname+"_"+format(batchstart,'06d')+"_"+format(batchend,'06d')+".bin","w")
        for j in range (batchstart,batchend):
		jacobian_m=[]
                for jj in range(10):
                        gradients_o = keras.backend.gradients(outputTensor[0][0][jj], listOfVariableTensors)
                        J= np.array(sess.run(gradients_o,feed_dict={model.input:data[j:j+1]})).flatten()
                        JJ=[J[i].flatten() for i in range(3)]
                        J=np.hstack(JJ)
                        #print np.shape(J)
                        jacobian_m.append(J)
		f.write(np.array(jacobian_m))
		print("At",j, "from", batchend-batchstart)
	f.close()

def check_workaround():
        print("Test consistency of workaround")
        dfx_i_test = tf.slice(gradients[0], begin=[0,0] , size=[1,1])
        gradgrad = keras.backend.gradients(dfx_i_test,listOfVariableTensors)
        workaround=sess.run(gradgrad, feed_dict={model.inputs[0]:data})
        print(HH[0][0][0][0][0])
        print(workaround[0][0][0])
        print("HH=%e, workaround=%e" % (HH[0][0][0][0][0], workaround[0][0][0]))

def finite_diff():
        print("Check finite differences. One sample")
        n=10
        h=[1./2**i for i in  range(n)]
        f_true=[0 for i in range(n)]
        f_finite_diff=[0 for i in range(n)]
        error=[float("inf") for i in range(n+1)]
        weights=model.get_weights()
        loss_eval=sess.run(loss,feed_dict={model.inputs[0]:data})
        for i in range(n):
                weights[0][0][0]=weights[0][0][0]+h[i]
                model.set_weights(weights)
                f_true[i]=sess.run(loss,feed_dict={model.inputis[0]:data})
                f_finite_diff[i]=loss_eval+h[i]*evaluated_gradients[0][0][0] + h[i]**2 * HH[0][0][0][0][0]
                weights[0][0][0]=weights[0][0][0]-h[i]
                error[i]=f_true[i]-f_finite_diff[i]
                print("hh=", h[i], "f(x)=",loss_eval,"f(x+dw)=",f_true[i],"f(x)+hh*g=",f_finite_diff[i], "delta", error[i])

def index_layer(ind):
        if ind >= aggregated_layer_weigths[no_layers]:
                return (no_layers-1, ind - aggregated_layer_weigths[no_layers-1])
        lev = 0
        for i in range (no_layers):
                if ind < (aggregated_layer_weigths[i+1]):
                        lev = i
                        break
        local_id = ind - aggregated_layer_weigths[lev]
        return (lev, local_id)

def hessian_single_eval(begin=0, ending=1):
        #HH=np.zeros((25408,num_rows))
        hessi = []
        no_of_levels = np.shape(listOfVariableTensors)[0]
        begin_vec = index_layer(begin)
        end_vec = index_layer(ending)
        print end_vec
        levels = np.arange(begin_vec[0],end_vec[0]+1)
        print levels
        for level in levels:
                I = np.asarray(np.shape(gradients[level]),dtype=np.int)
                tensor_dim = int(np.shape(I)[0])
                print tensor_dim
                print("I=",I) # must be a nested loop of I
                if tensor_dim == 1:
                        if (level == begin_vec[0]):
                                beg1 = begin_vec[1]%I[0]
                        else:
                                beg1 = 0
                        for i in range(beg1,I[0]):
                                if (end_vec[1]==i) and (end_vec[0] == level ):
                                        return
                                dfx_i = tf.slice(gradients[level], begin=[i] , size=[1])
                                print("dfx_i=", dfx_i, "shape",np.shape(dfx_i))
                                gradgrad = keras.backend.gradients(dfx_i,listOfVariableTensors)
                                A=sess.run(gradgrad, feed_dict={model.inputs[0]:data[:global_batch_size]})
                                print (np.shape(A), np.shape(A[-2]))
                                for iii in range(no_of_levels):
                                        f.write(A[iii])
                if tensor_dim == 2:
                        if (level == begin_vec[0]):
                                beg1 = begin_vec[1]/(I[1])
                                tmp = begin_vec[1]%(I[1])
                                beg2 = tmp
                        else:
                                beg1,beg2=0,0
                        for i in range(beg1,I[0]):
                                for j in range(beg2,I[1]):
                                        if (end_vec[1]==i*I[1]+j) and (end_vec[0] == level ):
                                                return
                                        dfx_i = tf.slice(gradients[level], begin=[i,j] , size=[1,1])
                                        print("dfx_i=", dfx_i, "shape",np.shape(dfx_i))
                                        gradgrad = keras.backend.gradients(dfx_i,listOfVariableTensors)
                                        A=sess.run(gradgrad, feed_dict={model.inputs[0]:data[:global_batch_size]})
                                        print (np.shape(A), np.shape(A[-2]))
                                        for iii in range(no_of_levels):
                                                f.write(A[iii])
                                        #f.write(A[0])
                                        #f.write(A[1])
                                        #f.write(A[2])
                                        #f.write(A[3])
                                        #hessi.append(gradgrad[0])
                                        #hessi.append(gradgrad[1])
                                beg2=0
                if tensor_dim == 4:
                        if (level == begin_vec[0]):
                                beg1 = begin_vec[1]/(I[1]*I[2]*I[3])
                                tmp = begin_vec[1]%(I[1]*I[2]*I[3])
                                beg2 = tmp/(I[2]*I[3])
                                tmp = tmp%(I[2]*I[3])
                                beg3 = tmp/(I[2]*I[3])
                                tmp = tmp%(I[3])
                                beg4 = tmp#/(I[3])
                                tmp = tmp%(I[1])
                                print(beg1,beg2,beg3,beg4)
                        else:
                                beg1,beg2,beg3,beg4=0,0,0,0
                        for i in range(beg1,I[0]):
                                for j in range(beg2,I[1]):
                                        for k in range(beg3,I[2]):
                                                for l in range(beg4,I[3]):
                                                        if (end_vec[1]==i*I[1]*I[2]*I[3]+j*I[2]*I[3]+k*I[3]+l) and (end_vec[0] == level ):
                                                                return
                                                        dfx_i = tf.slice(gradients[level], begin=[i,j,k,l] , size=[1,1,1,1])
                                                        print("dfx_i=", dfx_i, "shape",np.shape(dfx_i))
                                                        gradgrad = keras.backend.gradients(dfx_i,listOfVariableTensors)
                                                        A=sess.run(gradgrad, feed_dict={model.inputs[0]:data[:global_batch_size]})
                                                        print (np.shape(A), np.shape(A[-2]))
                                                        for iii in range(no_of_levels):
                                                                f.write(A[iii])
                                                        #f.write(A[0])
                                                        #f.write(A[1])
                                                        #f.write(A[2])
                                                        #f.write(A[3])
                                                        #hessi.append(gradgrad[0])
                                                        #hessi.append(gradgrad[1])
                                                        #hessi.append(gradgrad[2])
                                                        #hessi.append(gradgrad[3])
                                                beg4=0
                                        beg3=0
                                beg2=0

        #print("Shape new hessi",np.shape(hessi), np.shape(hessi[0]))
        #HH = sess.run(hessi, feed_dict={model.input:data})
        #print np.shape(HH)
        #return hessi





model = keras.Sequential()

#model.add(tf.keras.layers.Conv2D(16,input_shape=(28,28,1), kernel_size=(3, 3),strides=(1, 1),  activation='relu'))
#model.add(tf.keras.layers.MaxPooling2D(pool_size=(4, 4)))
#model.add(tf.keras.layers.Flatten())
#model.add(tf.keras.layers.Dense(128, activation='relu'))
#model.add(tf.keras.layers.Dense(10, activation='softmax'))



model.add(tf.keras.layers.Conv2D(4,input_shape=(28,28,1), kernel_size=(4, 4),strides=(1, 1),  activation='relu',use_bias=False))
#model.add(tf.keras.layers.MaxPooling2D(pool_size=(4, 4)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(16, activation='relu',use_bias=False))
#model.add(tf.keras.layers.Dense(10, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax',use_bias=False))




# To mitigate more than 1-dim input
#model.add(tf.keras.layers.Flatten())
# Adds a densely-connected layer with 64 units to the model:
#model.add(keras.layers.Dense(32, activation='relu',kernel_initializer='orthogonal', use_bias=True,  bias_initializer=keras.initializers.constant(1.0)))
# Dropout consists in randomly setting a fraction rate of input units to 0 at each update during training time, which helps prevent overfitting
#model.add(tf.keras.layers.Dropout(0.2))
# Add a softmax layer with 10 output units:
#model.add(keras.layers.Dense(10, activation='softmax'))#, use_bias=False))


# Create a sigmoid layer:
#layers.Dense(64, activation='sigmoid')
# Or:
#layers.Dense(64, activation=tf.sigmoid)

# A linear layer with L1 regularization of factor 0.01 applied to the kernel matrix:
#layers.Dense(64, kernel_regularizer=keras.regularizers.l1(0.01))
# A linear layer with L2 regularization of factor 0.01 applied to the bias vector:
#layers.Dense(64, bias_regularizer=keras.regularizers.l2(0.01))

# A linear layer with a kernel initialized to a random orthogonal matrix:
#layers.Dense(64, kernel_initializer='orthogonal')
# A linear layer with a bias vector initialized to 2.0s:
#layers.Dense(64, bias_initializer=keras.initializers.constant(2.0))


#model.compile(optimizer=tf.train.AdamOptimizer(0.001),
#              loss='categorical_crossentropy',
#              metrics=['accuracy'])

# Configure a model for mean-squared error regression.
model.compile(optimizer=tf.train.AdamOptimizer(0.01),
              loss='sparse_categorical_crossentropy',      
              metrics=['accuracy'])  


# Configure a model for categorical classification.
#model.compile(optimizer=tf.train.RMSPropOptimizer(0.01),
#              loss=keras.losses.categorical_crossentropy,
#              metrics=[keras.metrics.categorical_accuracy])


mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
#mnist = tf.contrib.learn.datasets.load_dataset("mnist")
#data = mnist.train.images  # Returns np.array
#labels = np.asarray(mnist.train.labels, dtype=np.int32)
#data = np.random.random((10, 32))
#labels = np.random.random((10, 1)) tf.reshape(x, [-1, 28, 28, 1])
#data = x_train.reshape(x_train.shape[0], 28, 28,1)
data = x_train[:global_batch_size].reshape(global_batch_size, 28, 28,1)

#input_shape = (1, img_rows, img_cols)
#data = np.asarray(tf.reshape(x_train, [-1, 28, 28, 1]))
#labels = y_train.reshape(y_train.shape[0],1)
labels = y_train[:global_batch_size].reshape(global_batch_size,1)

#val_data = mnist.test.images  # Returns np.array
#val_labels = np.asarray(mnist.test.labels, dtype=np.int32)
val_data = x_test.reshape(x_test.shape[0], 28, 28,1) # np.random.random((100, 32))
val_labels = y_test.reshape(y_test.shape[0],1) # np.random.random((100,1))

del x_train, y_train, x_test, y_test 

if training:
	sess = tf.InteractiveSession()
	sess.run(tf.global_variables_initializer()) # deprecated: initialize_all_variables())
	history = model.fit(data, labels, epochs=10, batch_size=global_batch_size)
	model.save_weights('./saved_models/cnn03')
	print("done training")
else:
	sess = tf.InteractiveSession()
	model.load_weights('./saved_models/cnn03')
	print("done loading")
	l , acc = model.evaluate( data[:global_batch_size], labels[:global_batch_size] )
	#print("model acc:{:5.2f}%".format(100*acc))
	#model.load_weights('./saved_models/test_cnn')
	#l , acc = model.evaluate( val_data, val_labels )
	#print("model acc:{:5.2f}%".format(100*acc))

if save_hessian: 
	pass
else:
	exit(1)


print("finished setting up")
outputTensor = model.outputs
#loss=  model.total_loss #model.loss_functions#tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=outputTensor)
#a=tf.reshape(tf.one_hot(labels[:],10),(1000,10))
#print a
print outputTensor
#loss = tf.losses.sparse_softmax_cross_entropy(labels,outputTensor)
loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(labels[:global_batch_size],10),logits=outputTensor)
#loss = tf.losses.sparse_categorical_crossentropy(labels,outputTensor) #softmax_cross_entropy (labels,outputTensor) # -tf.reduce_sum(labels * tf.log(outputTensor))
print loss
#print(sess.run(loss,feed_dict={model.input:data   }))
#loss_history = history.history["loss"]
#labels_pred=model.predict(data)
#loss=keras.losses.mean_squared_error(labels,labels_pred)
listOfVariableTensors = model.trainable_weights
print("Trainable weights", listOfVariableTensors)

gradients = keras.backend.gradients(loss, listOfVariableTensors)

# H =  tf.hessians(outputTensor, listOfVariableTensors)
# Get gradients of loss with repect to parameters
#parameters = tf.concat([tf.reshape(listOfVariableTensors[0],[32*64,1]),listOfVariableTensors[1]],0)
#print parameters
#print np.shape(gradients[0])
#print np.shape(listOfVariableTensors[0])
# hessian_single_eval()

#HH =  tf.hessians(loss_history, listOfVariableTensors)
#dloss_dw = tf.gradients(outputTensor, parameters)[0]
#evaluated_concat_gradient = sess.run(gradgrad,feed_dict={model.input:data})
#print(evaluated_concat_gradient)
#dim, _ = dloss_dw.get_shape()
#print("Shapes of parameters", parameters.shape)
#print dim
#hess = []
#for i in range(dim):
	# tf.slice: https://www.tensorflow.org/versions/0.6.0/api_docs/python/array_ops.html#slice
#	dfx_i = tf.slice(dloss_dw, begin=[i,0] , size=[1,1])
#	ddfx_i = tf.gradients(dfx_i, parameters)[0] # whenever we use tf.gradients, make sure you get the actual tensors by putting [0] at the end
#	print(np.shape(dfx_i),np.shape(ddfx_i))
#	hess.append(ddfx_i)

#print("Shape new hessian",np.shape(hess), np.shape(hess[0]))
#HH = sess.run(hess, feed_dict={model.input:data})
#print np.shape(HH)

#print gradients
#print H

# evaluated_gradients = sess.run(gradients,feed_dict={model.input:data})

#HH = sess.run(H,feed_dict={model.input:data})
#evaluated_concat_gradient = sess.run(dloss_dw,feed_dict={model.input:data})

#for i in range(2):
#	print gradients[i]
#	print("Norm Gradients" , np.linalg.norm(evaluated_gradients[i]))
#	print("Size Hessian" , np.shape(HH[i]))
#	print("Norm Hessian" , np.linalg.norm(HH[i]))


# Instantiates a toy dataset instance:
#dataset = tf.data.Dataset.from_tensor_slices((data, labels))
#dataset = dataset.batch(32)
#dataset = dataset.repeat()



# Don't forget to specify `steps_per_epoch` when calling `fit` on a dataset.
#model.fit(dataset, epochs=10, steps_per_epoch=30)

#for i in range(1):
#        print np.linalg.norm(evaluated_gradients[i])


#dataset = tf.data.Dataset.from_tensor_slices((data, labels))
#dataset = dataset.batch(32).repeat()

#val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
#val_dataset = val_dataset.batch(32).repeat()

#model.fit(dataset, epochs=10, steps_per_epoch=30,
#          validation_data=val_dataset,
#          validation_steps=3)
#evaluated_gradients = sess.run(gradients,feed_dict={model.input:data})
#for i in range(1):
#        print np.linalg.norm(evaluated_gradients[i])


#model.evaluate(dataset, steps=30)

#print("Check finite differences. One sample")
#print("shape data",tf.shape(data),"shape labels",labels[0,:].shape)
#test=np.random.random((1,32))
#predictions = model.predict(test)
#print predictions
#evaluated_gradients = sess.run(gradients,feed_dict={model.input:test})
#print np.shape(evaluated_gradients)
#print np.linalg.norm(evaluated_gradients)
#print evaluated_gradients

#print("h\t\tf(w+dw)\t\tf(x)+h*f'x+h**2*f''(x)\t\t error \t\t error reduction")
#for i in range(n):
#	print("%e \t %e \t %e \t %e \t %e" % (h[i],f_true[i],f_finite_diff[i],error[i],error[i]/error[i+1]))
#print([error[i]/error[i+1] for i in range(n-1)])
#print("[DONE] Check finite differences. One sample")


#model.predict(dataset, steps=30)
#opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)
#print opt.compute_gradients(predictions, tf.trainable_variables())


no_layers = np.shape(listOfVariableTensors)[0]
layer_weights = np.zeros((no_layers),dtype=np.int)
for  i in range(no_layers):
	layer_weights[i] = int(np.prod(listOfVariableTensors[i].shape))
	
aggregated_layer_weigths = [sum(layer_weights[:i]) for i in range(no_layers+1)]


### command line handling for hessian saving
tmp_nodestart = aggregated_layer_weigths[-1] * int(sys.argv[3])/int(sys.argv[4])
size_local = aggregated_layer_weigths[-1] / int(sys.argv[2]) / int(sys.argv[4])
local_start = (int(sys.argv[1])-1)*aggregated_layer_weigths[-1]/ int(sys.argv[2]) / int(sys.argv[4])
start = tmp_nodestart + local_start
end = tmp_nodestart + local_start + size_local
print(start,end)
### command line handling for jacobians: sys.argv[3] is batchsize
tmp_nodestart = int(sys.argv[5]) * int(sys.argv[3])/int(sys.argv[4])
size_local = int(sys.argv[5]) / int(sys.argv[2]) / int(sys.argv[4])
local_start = (int(sys.argv[1])-1)*int(sys.argv[5])/ int(sys.argv[2]) / int(sys.argv[4])
start_jacob = tmp_nodestart + local_start
end_jacob = tmp_nodestart + local_start + size_local
print("Jacobian:", start_jacob,end_jacob)

start_time=time.time()

#full_jacobian=[]
### Jacobian ###
jacobian(start_jacob, end_jacob) 
#f=open("jacobian_cnn02.bin","w")
#f.write(np.array(full_jacobian))
#f.close()


#f=open("./data/cnn01s"+str(start)+"e"+str(end),"wa")  
#hessian_single_eval(start,end)
#f.close()
#print(appendedList)
#HH=sess.run(appendedList[:10], feed_dict={model.input:data})
print ("Done")
#H=np.zeros((25408,128))

#for i in range(128):
#	H[:25088,i]=np.reshape(HH[2*i],(25088))
#	H[25088:,i]=np.reshape(HH[2*i+1],(320))

end_time=time.time()

print("time for row-wise is ", end_time-start_time)


#print(H)
