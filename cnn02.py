#!/usr/bin/python

import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np
import itertools
import time
 
training=True
save_hessian=True
global_batch_size = 10000

batch_size = 256
num_classes = 10
epochs = 100
data_augmentation = True
cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
#(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
#mnist = tf.contrib.learn.datasets.load_dataset("mnist")
#data = mnist.train.images  # Returns np.array

#data = x_train[:global_batch_size].reshape(global_batch_size, 32, 32,3)


model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

    # Compile the model
model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0001, decay=1e-6),
                  metrics=['accuracy'])





# Configure a model for categorical classification.
#model.compile(optimizer=tf.train.RMSPropOptimizer(0.01),
#              loss=keras.losses.categorical_crossentropy,
#              metrics=[keras.metrics.categorical_accuracy])


#mnist = tf.contrib.learn.datasets.load_dataset("mnist")
#data = mnist.train.images  # Returns np.array
#labels = np.asarray(mnist.train.labels, dtype=np.int32)
#data = np.random.random((10, 32))
#labels = np.random.random((10, 1)) tf.reshape(x, [-1, 28, 28, 1])
#data = x_train.reshape(x_train.shape[0], 28, 28,1)
#input_shape = (1, img_rows, img_cols)
#data = np.asarray(tf.reshape(x_train, [-1, 28, 28, 1]))
#labels = y_train.reshape(y_train.shape[0],1)

#val_data = mnist.test.images  # Returns np.array
#val_labels = np.asarray(mnist.test.labels, dtype=np.int32)
#val_data = x_test.reshape(x_test.shape[0], 28, 28,1) # np.random.random((100, 32))
#val_labels = y_test.reshape(y_test.shape[0],1) # np.random.random((100,1))


if training:
	sess = tf.InteractiveSession()
	sess.run(tf.global_variables_initializer()) # deprecated: initialize_all_variables())
	history = model.fit(x_train, y_train, epochs=epochs, batch_size= batch_size,
		validation_data=(x_test,y_test),shuffle=True,
		callbacks=[EarlyStopping(min_delta=0.001, patience=3)])
	model.save_weights('./saved_models/test_cnn_cifar10')
	print("done training")
        fig1 = plt.figure(0)
    	plt.plot(model.history.history['acc'],'r')
    	plt.plot(model.history.history['val_acc'],'g')
    	plt.xticks(np.arange(0, 101, 10))
    	plt.rcParams['figure.figsize'] = (8, 6)
    	plt.xlabel("Num of Epochs")
    	plt.ylabel("Accuracy")
    	plt.title("Training Accuracy vs Validation Accuracy")
    	plt.legend(['train','validation'])
    	fig2 = plt.figure(1)
    	plt.plot(model.history.history['loss'],'r')
    	plt.plot(model.history.history['val_loss'],'g')
    	plt.xticks(np.arange(0, 101, 10))
    	plt.rcParams['figure.figsize'] = (8, 6)
    	plt.xlabel("Num of Epochs")
    	plt.ylabel("Loss")
    	plt.title("Training Loss vs Validation Loss")
    	plt.legend(['train','validation'])
    	fig1.savefig("cifar10acc.pdf")
    	fig2.savefig("cifar10loss.pdf")

    	# Evaluate the model
    	scores = model.evaluate(X_test / 255.0, to_categorical(Y_test))

    	print('Loss: %.3f' % scores[0])
    	print('Accuracy: %.3f' % scores[1])


else:
	sess = tf.InteractiveSession()
	model.load_weights('./saved_models/test_cnn_cifar10')
	l , acc = model.evaluate( data[:global_batch_size], labels[:global_batch_size] )
	#print("model acc:{:5.2f}%".format(100*acc))
	#model.load_weights('./saved_models/test_cnn')
	#l , acc = model.evaluate( val_data, val_labels )
	#print("model acc:{:5.2f}%".format(100*acc))


if save_hessian:
        pass
else:
        exit(1)



outputTensor = model.output
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
def finite_diff():
	print("Check finite differences. One sample")
	n=10
	h=[1./2**i for i in  range(n)]
	f_true=[0 for i in range(n)]
	f_finite_diff=[0 for i in range(n)]
	error=[float("inf") for i in range(n+1)]
	weights=model.get_weights()
	loss_eval=sess.run(loss,feed_dict={model.input:data})
	for i in range(n):
		weights[0][0][0]=weights[0][0][0]+h[i]
		model.set_weights(weights)
		f_true[i]=sess.run(loss,feed_dict={model.input:data})
		f_finite_diff[i]=loss_eval+h[i]*evaluated_gradients[0][0][0] + h[i]**2 * HH[0][0][0][0][0]
		weights[0][0][0]=weights[0][0][0]-h[i]
		error[i]=f_true[i]-f_finite_diff[i]
		print("hh=", h[i], "f(x)=",loss_eval,"f(x+dw)=",f_true[i],"f(x)+hh*g=",f_finite_diff[i], "delta", error[i])

#print("h\t\tf(w+dw)\t\tf(x)+h*f'x+h**2*f''(x)\t\t error \t\t error reduction")
#for i in range(n):
#	print("%e \t %e \t %e \t %e \t %e" % (h[i],f_true[i],f_finite_diff[i],error[i],error[i]/error[i+1]))
#print([error[i]/error[i+1] for i in range(n-1)])
#print("[DONE] Check finite differences. One sample")

def check_workaround():
	print("Test consistency of workaround")
	dfx_i_test = tf.slice(gradients[0], begin=[0,0] , size=[1,1])
	gradgrad = keras.backend.gradients(dfx_i_test,listOfVariableTensors)
	workaround=sess.run(gradgrad, feed_dict={model.input:data})
	print(HH[0][0][0][0][0])
	print(workaround[0][0][0])
	print("HH=%e, workaround=%e" % (HH[0][0][0][0][0], workaround[0][0][0]))


#model.predict(dataset, steps=30)
#opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)
#print opt.compute_gradients(predictions, tf.trainable_variables())


no_layers = np.shape(listOfVariableTensors)[0]
layer_weights = np.zeros((no_layers),dtype=np.int)
for  i in range(no_layers):
	layer_weights[i] = int(np.prod(listOfVariableTensors[i].shape))
	
aggregated_layer_weigths = [sum(layer_weights[:i]) for i in range(no_layers+1)]

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
start = (int(sys.argv[1])-1)*aggregated_layer_weigths[-1] / int(sys.argv[2])
end = (int(sys.argv[1]))*aggregated_layer_weigths[-1] / int(sys.argv[2])
print(start,end)

#param_grad = tf.gradients(model.total_loss, model.model_params)
def hessian_single_eval(begin=0, end=1):
	#HH=np.zeros((25408,num_rows))
	hessi = []
	no_of_levels = np.shape(listOfVariableTensors)[0]
	begin_vec = index_layer(begin)
	end_vec = index_layer(end)
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
                                A=sess.run(gradgrad, feed_dict={model.input:data[:global_batch_size]})
                                f.write(A[0])
                                f.write(A[1])
                                f.write(A[2])
                                f.write(A[3])
		if tensor_dim == 2:
			if (level == begin_vec[0]):
				beg1 = begin_vec[1]/(I[0]*I[1])
				tmp = begin_vec[1]%(I[0]*I[1])
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
                                        A=sess.run(gradgrad, feed_dict={model.input:data[:global_batch_size]})
                                        f.write(A[0])
                                        f.write(A[1])
                                        f.write(A[2])
                                        f.write(A[3])
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
							A=sess.run(gradgrad, feed_dict={model.input:data[:global_batch_size]})
							f.write(A[0])
							f.write(A[1])
							f.write(A[2])
							f.write(A[3])
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

start_time=time.time()
f=open("data/cnn01s"+str(start)+"e"+str(end),"wa")
hessian_single_eval(start,end)
f.close()
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
