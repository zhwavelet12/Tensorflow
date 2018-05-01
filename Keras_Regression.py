import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation #import animation support
from keras.models import Sequential
from keras.layers.core import Dense, Activation

num_house = 160
np.random.seed(42)
house_size = np.random.randint(low=1000,high=3500,size=num_house)

np.random.seed(42)
house_price = house_size*100.0+np.random.randint(low=20000,high=70000,size=num_house)

# Plot generated house and size  
#plt.plot(house_size, house_price, "bx")  # bx = blue x 
#plt.ylabel("Price") 
#plt.xlabel("Size") 
#plt.show()

# you need to normalize values to prevent under/overflows. 
def normalize(array):
    return (array - array.mean()) / array.std()

# define number of training samples, 0.7 = 70%.  

# We can take the first 70% since the values are randomized 

num_train_samples = math.floor(num_house * 0.7)  



# define training data 

train_house_size = np.asarray(house_size[:num_train_samples]) 

train_price = np.asanyarray(house_price[:num_train_samples:])  

train_house_size_norm = normalize(train_house_size) 

train_price_norm = normalize(train_price)  

train_house_size_std = train_house_size.std()
train_house_size_mean = train_house_size.mean()
train_price_std = train_price.std()
train_price_mean = train_price.mean()

# define test data 

test_house_size = np.array(house_size[num_train_samples:]) 

test_house_price = np.array(house_price[num_train_samples:])  

test_house_size_norm = normalize(test_house_size) 

test_house_price_norm = normalize(test_house_price)

model = Sequential()
model.add(Dense(1, input_shape=(1,), init='uniform', activation='linear'))
model.compile(loss='mean_squared_error',optimizer='sgd')#Loss and optimizer

model.fit(train_house_size_norm,train_price_norm,nb_epoch=300)

score = model.evaluate(test_house_size_norm,test_house_price_norm)
print("\nloss on test: {0}".format(score))


#====================== all deleted ------------
#  Set up the TensorFlow placeholders that get updated as we descend down the gradient
#   Constant: constant value
#   Variable: values adjusted in graph
#   PlaceHolder: used to pass data into graph.
#
#Place holder example:
#x = tf.placeholder(tf.float32, shape=(1024, 1024))  
#y = tf.matmul(x, x)  
#  
#with tf.Session() as sess:  
#  print(sess.run(y))  # ERROR: x is not put value with feed_dict
#  
#  rand_array = np.random.rand(1024, 1024)  
#  print(sess.run(y, feed_dict={x: rand_array}))  # Will succeed. 
#
#
#tf_house_size = tf.placeholder("float", name="house_size")
#tf_price = tf.placeholder("float", name="price")

#print(tf_house_size)
#print(tf_price)

# Define the variables holding the size_factor and price we set during training.  
# We initialize them to some random values based on the normal distribution.
#tf_size_factor = tf.Variable(initial_value=np.random.randn(), name="size_factor")
#tf_price_offset = tf.Variable(initial_value=np.random.randn(), name="price_offset")
#print(tf_size_factor)
#print(tf_price_offset)

# 2. Define the operations for the predicting values - predicted price = (size_factor * house_size ) + price_offset
#  Notice, the use of the tensorflow add and multiply functions.  These add the operations to the computation graph,
#  AND the tensorflow methods understand how to deal with Tensors.  Therefore do not try to use numpy or other library 
#  methods.
#tf_price_pred = tf.add(tf.multiply(tf_size_factor, tf_house_size), tf_price_offset)

# 3. Define the Loss Function (how much error) - Mean squared error
#tf_cost = tf.reduce_sum(tf.pow(tf_price_pred-tf_price, 2))/(2*num_train_samples)

# Optimizer learning rate.  The size of the steps down the gradient
#learning_rate = 0.1

# 4. define a Gradient descent optimizer that will minimize the loss defined in the operation "cost".
#optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_cost)

# Initializing the variables
#init = tf.global_variables_initializer()

# keep iterating the training data
# display_every=1
# num_training_iter = 1
# sess = tf.Session()
# sess.run(init)
# fit_plot_idx=1
# fit_size_factor = np.empty(math.floor(num_training_iter + 1 ))
# fit_price_offsets = np.empty(math.floor(num_training_iter + 1))

# for iteration in range(num_training_iter):
#     # Fit all training data
#     for (x, y) in zip(train_house_size_norm, train_price_norm):
#         sess.run(optimizer, feed_dict={tf_house_size:x, tf_price:y})
#         # Display current status
#         if (iteration + 1) % display_every == 0:
#             c = sess.run(tf_cost, feed_dict={tf_house_size: train_house_size_norm, tf_price:train_price_norm})
#             print("iteration #:", '%04d' % (iteration + 1), "cost=", "{:.9f}".format(c), \
#                 "size_factor=", sess.run(tf_size_factor), "price_offset=", sess.run(tf_price_offset))
#             # Save the fit size_factor and price_offset to allow animation of learning process
#     fit_size_factor[fit_plot_idx] = sess.run(tf_size_factor)
#     fit_price_offsets[fit_plot_idx] = sess.run(tf_price_offset)
#     fit_plot_idx = fit_plot_idx + 1

# print("Optimization Finished!")
# training_cost = sess.run(tf_cost, feed_dict={tf_house_size: train_house_size_norm, tf_price: train_price_norm})
# print("Trained cost=", training_cost, "size_factor=", sess.run(tf_size_factor), "price_offset=", sess.run(tf_price_offset), '\n')

# # 
# # Plot another graph that animation of how Gradient Descent sequentually adjusted size_factor and price_offset to 
# # find the values that returned the "best" fit line.
# fig, ax = plt.subplots()
# line, = ax.plot(house_size, house_price)

# plt.rcParams["figure.figsize"] = (10,8)
# plt.title("Gradient Descent Fitting Regression Line")
# plt.ylabel("Price")
# plt.xlabel("Size (sq.ft)")
# plt.plot(train_house_size, train_price, 'go', label='Training data')
# plt.plot(test_house_size, test_house_price, 'mo', label='Testing data')

# def animate(i):
#     line.set_xdata(train_house_size_norm * train_house_size_std + train_house_size_mean)  # update the data
#     line.set_ydata((fit_size_factor[i] * train_house_size_norm + fit_price_offsets[i]) * train_price_std + train_price_mean)  # update the data
#     return line,
 
# # Init only required for blitting to give a clean slate.
# def initAnim():
#     line.set_ydata(np.zeros(shape=house_price.shape[0])) # set y's to 0
#     return line,

# ani = animation.FuncAnimation(fig, animate, frames=np.arange(0, fit_plot_idx), init_func=initAnim,
#                                  interval=1000, blit=True)
# plt.show() 
