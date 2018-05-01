
# coding: utf-8

# In[36]:


import tensorflow as tf
#from tensorflow.examples.tourials.minist import input_data
import input_data

from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Convolution2D,MaxPooling2D
from keras import backend as k

logPath=".\\tb_logs"

mnist = input_data.read_data_sets("H:\\zhou\\stanford_TF\\data\\MNIST", one_hot=True)

sess=tf.InteractiveSession()    
#sess=tf.InteractiveSession()
# with tf.name_scope("MNIST_Input"):
#     x = tf.placeholder(tf.float32, [None,784],name="x")
#     y_ = tf.placeholder(tf.float32, [None,10],name="y_")
 
image_rows = 28
image_cols = 28

train_images = mnist.train.images.reshape(mnist.train.images.shape[0], image_rows,image_cols,1)
test_images = mnist.test.images.reshape(mnist.test.images.shape[0], image_rows,image_cols,1)

num_filters = 32
max_pool_size = (2,2)
conv_kernel_size = (3,3)
imag_shape = (28,28,1)
num_classes = 10
drop_prob = 0.5

model = Sequential()

#1st
model.add(Convolution2D(num_filters,conv_kernel_size[0], conv_kernel_size[1],
                        border_mode='valid', input_shape = imag_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=max_pool_size))

#2nd
model.add(Convolution2D(num_filters,conv_kernel_size[0], conv_kernel_size[1],
                        border_mode='valid', input_shape = imag_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=max_pool_size))

#fully connected layer
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))

#dropout layer
model.add(Dropout(drop_prob))

#readout layer
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

batch_size = 128
num_epoch = 2

model.fit(train_images,mnist.train.labels,batch_size=batch_size, nb_epoch = num_epoch,
           verbose=1, validation_data=(test_images,mnist.test.labels))

# def variable_summaries(var):
#     with tf.name_scope("sumaries"):
#         mean=tf.reduce_mean(var)
#         tf.summary.scalar("mean",mean)
#         with tf.name_scope("stddev"):
#             stddev=tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
#         tf.summary.scalar("stddev",stddev)
#         tf.summary.scalar("max",tf.reduce_max(var))
#         tf.summary.scalar("min",tf.reduce_min(var))
#         tf.summary.histogram("histogram",var)
        
# #定义一个函数，用于初始化所有的权值 W
# def weight_variable(shape,name=None):
#     initial = tf.truncated_normal(shape, stddev=0.1,name=name)
#     return tf.Variable(initial)

# #定义一个函数，用于初始化所有的偏置项 b
# def bias_variable(shape,name=None):
#     initial = tf.constant(0.1, shape=shape,name=name)
#     return tf.Variable(initial)
  
# #定义一个函数，用于构建卷积层
# def conv2d(x, W,name=None):
#     return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME',name=name)

# #定义一个函数，用于构建池化层
# def max_pool(x,name=None):
#     return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME',name=name)

# #构建网络
# # with tf.name_scope("Input_Reshape"):
# #     x_image = tf.reshape(x, [-1,28,28,1],name="x_Image")         #转换输入数据shape,以便于用于网络中

# with tf.name_scope("Conv1"):
#     with tf.name_scope("weights"):
#         W_conv1 = weight_variable([5, 5, 1, 32],name="weight") 
#         variable_summaries(W_conv1)
#     with tf.name_scope("biases"):
#         b_conv1 = bias_variable([32],name="bias") 
#         variable_summaries(b_conv1)
#     conv1_wx_b = conv2d(x_image, W_conv1, name="conv2d") + b_conv1
#     tf.summary.histogram("conv1_wx_b",conv1_wx_b)
    
#     h_conv1 = tf.nn.relu(conv1_wx_b,name="relu")     #第一个卷积层
#     tf.summary.histogram("h_conv1",h_conv1)
    
#     h_pool1 = max_pool(h_conv1, name="pool")                                  #第一个池化层

# with tf.name_scope("Conv2"):
#     with tf.name_scope("weights"):
#         W_conv2 = weight_variable([5, 5, 32, 64],name="weight")
#         variable_summaries(W_conv2)
#     with tf.name_scope("biases"):
#         b_conv2 = bias_variable([64],name="bias")
#         variable_summaries(b_conv2)
#     conv2_wx_b = conv2d(h_pool1, W_conv2,name="conv2d") + b_conv2
#     tf.summary.histogram("conv2_wx_b",conv2_wx_b)
    
#     h_conv2 = tf.nn.relu(conv2_wx_b,name="relu")      #第二个卷积层  
#     tf.summary.histogram("h_conv2",h_conv2)
        
#     h_pool2 = max_pool(h_conv2,name="pool")                                   #第二个池化层

# with tf.name_scope("FC"):
#     W_fc1 = weight_variable([7 * 7 * 64, 1024],name="weight")
#     b_fc1 = bias_variable([1024],name="bias")
#     h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])    #reshape成向量 x2 why
#     h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1,name="relu")    #第一个全连接层

# keep_prob = tf.placeholder(tf.float32,name="keep_prob") 
# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)                  #dropout层
# variable_summaries(h_fc1_drop)


# with tf.name_scope("Readout"):
#     W_fc2 = weight_variable([1024, 10],name="weight")
#     b_fc2 = bias_variable([10],name="bias")
#     variable_summaries(W_fc2)
#     variable_summaries(b_fc2)
    
#     #define model
#     y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
#     #y_predict=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)   #softmax层
#     variable_summaries(y_conv)

# learning_rate = tf.placeholder(tf.float32, shape=[])
# tf.summary.scalar("learning_rate", learning_rate)

# #cross_entropy = -tf.reduce_sum(y_actual*tf.log(y_predict))     #交叉熵
# with tf.name_scope("cross_entropy"):
#     #cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv,labels=y_))     #交叉熵
#     cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
#     variable_summaries(cross_entropy)
    
# with tf.name_scope("loss_optimizer"):
#     #train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)    #梯度下降法
#     train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    
# with tf.name_scope("accuracy"):
#     correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))    
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))                 #精确度计算

# #tf.summary.scalar("cross_entropy_sc1",cross_entropy)
# #tf.summary.scalar("training_accuracy",accuracy)

# merged = tf.summary.merge_all()
# tbWriter = tf.summary.FileWriter(logPath, sess.graph)

# #sess.run(tf.initialize_all_variables())
# #sess.run(tf.global_variables_initializer())
# tf.global_variables_initializer().run()

# import time

# #num_steps = 3000
# #display_every = 100
# num_steps = 30
# display_every = 10

# start_time=time.time()
# end_time = time.time()

# #train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    
# for i in range(num_steps):
#     batch = mnist.train.next_batch(50)
#     #sess.run(train_step,feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
#     #_,summary = sess.run([train_step,summarize_all], feed_dict={x: batch[0], y_: batch[1], learning_rate:0.02, keep_prob: 0.5})
#     #summary, _=sess.run([merged,train_step], feed_dict={x: batch[0], y_: batch[1], learning_rate:0.02, keep_prob: 0.5})
#     summary, _= sess.run([merged, train_step], feed_dict={x:batch[0], y_: batch[1], learning_rate:0.02, keep_prob: 0.5})
    
#     if i%display_every == 0:                  #训练100次，验证一次
#         train_acc = accuracy.eval(feed_dict={x:batch[0], y_:batch[1], keep_prob: 1.0})
#         print("step {0}, training accuracy {1:.2f}".format(i,train_acc))
#         tbWriter.add_summary(summary, i)
        
# end_time = time.time()
# print("Total training time for {0} bathes :{1:.2f} seconds".format(i+1,end_time-start_time))

# # cannot run the eval funciton it will cause the mem read too much.
# #test_acc=accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
# #print("test accuracy {0:.2f}".format(test_acc))

# sess.close()

