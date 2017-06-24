from __future__ import print_function
import tensorflow as tf
import read_array as rd
import random
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix
import sklearn as sk


learning_rate = 0.001
training_iters = 150000
batch_size = 128
display_step = 10
checkpoint_dir = "checkpoints/"

n_input = 50176 
n_classes = 2 
dropout = 1 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


def conv2d(x, W, b, strides=1,p='SAME'):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=p)
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2,strides=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, strides, strides, 1],
                          padding='SAME')
def avepool2d(x, k=2,strides=2):
    # MaxPool2D wrapper
    return tf.nn.avg_pool(x, ksize=[1, k, k, 1], strides=[1, strides, strides, 1],
                          padding='VALID')
def conv_net(x, weights, biases, dropout):
    x = tf.reshape(x, shape=[-1, 224, 224, 1])
    
    conv1 = conv2d(x, weights['wc1'], biases['bc1'],4,p='SAME')
    conv1 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv1 = conv2d(conv1, weights['wc3'], biases['bc3'])
    conv1 = maxpool2d(conv1, k=3,strides=2)
    fc1 = tf.reshape(conv1, shape = [-1, 75264])
    fc1 = tf.nn.bias_add(tf.matmul(fc1, weights['wd1']), biases['bd1'])

    conv2 = conv2d(conv1, weights['wc4'], biases['bc4'],p='SAME')
    conv2 = conv2d(conv2, weights['wc5'], biases['bc5'])
    conv2 = conv2d(conv2, weights['wc6'], biases['bc6'])
    conv2 = maxpool2d(conv2, k=3,strides=2)
    fc2 = tf.reshape(conv2, shape = [-1, 50176])
    fc2 = tf.nn.bias_add(tf.matmul(fc2, weights['wd2']), biases['bd2'])

    conv3 = conv2d(conv2, weights['wc7'], biases['bc7'],p='SAME')
    conv3 = conv2d(conv3, weights['wc8'], biases['bc8'])
    conv3 = conv2d(conv3, weights['wc9'], biases['bc9'])
    conv3 = maxpool2d(conv3, k=3,strides=2)
    conv3 = tf.nn.dropout(conv3, dropout)
    print ('conv3', conv3.get_shape())
    fc3 = tf.reshape(conv3, shape = [-1, 18816])
    fc3 = tf.nn.bias_add(tf.matmul(fc3, weights['wd3']), biases['bd3'])
    
    conv4 = conv2d(conv3, weights['wc10'], biases['bc10'])
    conv4 = conv2d(conv4, weights['wc11'], biases['bc11'])
    conv4 = conv2d(conv4, weights['wc12'], biases['bc12'])
    conv4 = avepool2d(conv4,k=7,strides=1)
    conv4 = tf.reshape(conv4, shape=[-1, 1000])
    fc4 = tf.nn.bias_add(tf.matmul(conv4, weights['wd4']), biases['bd4'])
    return [fc1, fc2, fc3, fc4]
weights = {

    'wc1' : tf.get_variable("wc1", shape=[11, 11, 1, 96], initializer=tf.contrib.layers.xavier_initializer()),
    'wc2' : tf.get_variable("wc2", shape=[1,1, 96, 96], initializer=tf.contrib.layers.xavier_initializer()),
    'wc3' : tf.get_variable("wc3", shape=[1,1, 96, 96], initializer=tf.contrib.layers.xavier_initializer()),
    'wc4' : tf.get_variable("wc4", shape=[5, 5, 96, 256], initializer=tf.contrib.layers.xavier_initializer()),
    'wc5' : tf.get_variable("wc5", shape=[1, 1, 256, 256], initializer=tf.contrib.layers.xavier_initializer()),
    'wc6' : tf.get_variable("wc6", shape=[1, 1, 256, 256], initializer=tf.contrib.layers.xavier_initializer()),
    'wc7' : tf.get_variable("wc7", shape=[3, 3, 256, 384], initializer=tf.contrib.layers.xavier_initializer()),
    'wc8' : tf.get_variable("wc8", shape=[1, 1, 384, 384], initializer=tf.contrib.layers.xavier_initializer()),
    'wc9' : tf.get_variable("wc9", shape=[1, 1, 384, 384], initializer=tf.contrib.layers.xavier_initializer()),
    'wc10' : tf.get_variable("wc10", shape=[3, 3, 384, 1024], initializer=tf.contrib.layers.xavier_initializer()),
    'wc11' : tf.get_variable("wc11", shape=[1, 1, 1024, 1024], initializer=tf.contrib.layers.xavier_initializer()),
    'wc12' : tf.get_variable("wc12", shape=[1, 1, 1024, 1000], initializer=tf.contrib.layers.xavier_initializer()),
    'wd1' : tf.get_variable("wd1", shape=[75264, 2], initializer=tf.contrib.layers.xavier_initializer()),
    'wd2' : tf.get_variable("wd2", shape=[50176, 2], initializer=tf.contrib.layers.xavier_initializer()),
    'wd3' : tf.get_variable("wd3", shape=[18816, 2], initializer=tf.contrib.layers.xavier_initializer()),
    'wd4' : tf.get_variable("wd4", shape=[1000, 2], initializer=tf.contrib.layers.xavier_initializer()),

   # 'wc1': tf.Variable(tf.random_normal([11, 11, 1, 96])),
   # 'wc2': tf.Variable(tf.random_normal([1,1, 96, 96])),
   # 'wc3': tf.Variable(tf.random_normal([1,1, 96, 96])),
   # 'wc4': tf.Variable(tf.random_normal([5, 5, 96, 256])),
   # 'wc5': tf.Variable(tf.random_normal([1, 1, 256, 256])),
   # 'wc6': tf.Variable(tf.random_normal([1, 1, 256, 256])),
    #'wc7': tf.Variable(tf.random_normal([3, 3, 256, 384])),
    #'wc8': tf.Variable(tf.random_normal([1, 1, 384, 384])),
    #'wc9': tf.Variable(tf.random_normal([1, 1, 384, 384])),
    #'wc10': tf.Variable(tf.random_normal([3, 3, 384, 1024])),
    #'wc11': tf.Variable(tf.random_normal([1, 1, 1024, 1024])),
    #'wc12': tf.Variable(tf.random_normal([1, 1, 1024, 1000])),
    #'wd1': tf.Variable(tf.random_normal([1000, 2])),
}    
biases = {
    'bc1': tf.Variable(tf.random_normal([96])),
    'bc2': tf.Variable(tf.random_normal([96])),
    'bc3': tf.Variable(tf.random_normal([96])),
    'bc4': tf.Variable(tf.random_normal([256])),
    'bc5': tf.Variable(tf.random_normal([256])),
    'bc6': tf.Variable(tf.random_normal([256])),
    'bc7': tf.Variable(tf.random_normal([384])),
    'bc8': tf.Variable(tf.random_normal([384])),
    'bc9': tf.Variable(tf.random_normal([384])),
    'bc10': tf.Variable(tf.random_normal([1024])),
    'bc11': tf.Variable(tf.random_normal([1024])),
    'bc12': tf.Variable(tf.random_normal([1000])),
    'bd1': tf.Variable(tf.random_normal([2])),
    'bd2': tf.Variable(tf.random_normal([2])),
    'bd3': tf.Variable(tf.random_normal([2])),
    'bd4': tf.Variable(tf.random_normal([2])),
} 
# Construct model
pred_list  =conv_net(x, weights, biases, keep_prob)
cost_list = []
optimizer_list = []
correct_pred_list = []
accuracy_list = []

for pred in pred_list:
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    cost_list.append(cost)
    optimizer_list.append(optimizer)
    correct_pred_list.append(correct_pred)
    accuracy_list.append(accuracy)

# Initializing the variables
init = tf.global_variables_initializer()
def tran(bx,batchsize):
    for i in range(0,batchsize):
        tmp=np.reshape(bx[i,0:50176],(224,224))
        box=(random.uniform(0, 15)  ,random.uniform(0, 15)  ,224-random.uniform(0, 15),224-random.uniform(0, 15))
        roi=Image.fromarray(tmp)
        roi=roi.crop(box)
        #roi=img.crop(box)
        roi = roi.resize((224, 224))
        roi = np.asarray(roi)
        bx[i,0:50176]=np.reshape(roi,(50176))
        return bx
# Launch the graph
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    for num_layer in list(range(4)):
        print ('begin layer = ', num_layer)
        pred = pred_list[num_layer]
        cost = cost_list[num_layer]
        optimizer = optimizer_list[num_layer]
        correct_pred = correct_pred_list[num_layer]
        accuracy = accuracy_list[num_layer]
        step = 0
        loss = 0
        num_batch = int(5615 / batch_size)
        while step * batch_size < training_iters:
            if (step%num_batch)==0:
                batch_train=rd.shuffle_train()
                batch_eval=rd.shuffle_eval()
                print("shuffled!")
            stepmod=step%num_batch
            batch_x=batch_train[(stepmod)*batch_size:(stepmod+1)*batch_size,0:50176]
            batch_x=tran(batch_x,batch_size)*255
            batch_y=batch_train[(stepmod)*batch_size:(stepmod+1)*batch_size,50176:50178]            
            # Run optimization op (backprop)
            #now_pred = pred.eval(feed_dict={x: batch_x, y: batch_y,keep_prob: dropout})
            #print ('shape')
            #print (now_pred.shape)
            _, loss_c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y,
                                           keep_prob: dropout})
            loss += loss_c * 1.0 / display_step
            if step % display_step == display_step - 1:
                # Calculate batch loss and accuracy
                y_p = tf.argmax(pred, 1)
                acc, y_pred = sess.run([accuracy, y_p], feed_dict={x: batch_eval[:, 0:50176] * 255,
                                                                  y: batch_eval[:,50176:50178],
                                                                  keep_prob: 1.})
                # loss_t = sess.run(cost, feed_dict = {x : batch_x, y : batch_y, keep_prob : 1.})
                y_true = np.argmax(batch_eval[:,50176:50178],1)
                print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Testing Accuracy= " + \
                      "{:.5f}".format(acc) + ", Precision = " + \
                      "{:.5f}".format(sk.metrics.precision_score(y_true, y_pred, pos_label = 0)) + ", Recall = " + \
                      "{:.5f}".format(sk.metrics.recall_score(y_true, y_pred, pos_label = 0)) + ", f1_score = " + \
                      "{:.5f}".format(sk.metrics.f1_score(y_true, y_pred, pos_label = 0)))

                loss = 0
            step += 1
            saver.save(sess, checkpoint_dir + 'model.ckpt', global_step = num_layer)
    print("Optimization Finished!")
    batch_eval=rd.shuffle_eval()

    # Calculate accuracy for 256 mnist test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: batch_eval[:,0:50176]*255,
                                      y: batch_eval[:,50176:50178],
                                      keep_prob: 1.}))