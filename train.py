#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 00:31:50 2018

@author: aidanrocke
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 10:32:53 2018

@author: aidanrocke
"""

import random
import tensorflow as tf
import numpy as np

from free_energy_agent import free_agent

## set random seed:
random.seed(42)
tf.set_random_seed(42)

def train(epochs,batch_size,basic_needs,success_probability):
    # initialize count:
    count = 0
    
    ## define number of evaluations:
    N = int(epochs/100)
    
    # initialize food vectors and total consumption:
    food_policies = np.zeros((N,24))
    total_consumption = np.zeros(N)
        
    with tf.Session() as sess:
                
        F = free_agent(basic_needs,sess,success_probability) 
        log_loss = F.surprise()
        
        ### it might be a good idea to regularise the squared loss:
        surprisal = -1.0*tf.reduce_mean(log_loss)
        
        ### define the optimiser:
        optimizer = tf.train.AdagradOptimizer(0.01)
        train_agent = optimizer.minimize(surprisal)
        
        ### initialise the variables:
        sess.run(tf.global_variables_initializer())
        
        log_loss = np.zeros(epochs)
                        
        for i in range(epochs):
            
            mini_batch = basic_needs*np.ones((batch_size,1),dtype=np.float32)
                
            train_feed = {F.survival : mini_batch}
            sess.run(train_agent,feed_dict = train_feed)
            
            log_loss[i] = sess.run(surprisal,feed_dict = train_feed)
            
            ## check variances:
            if i % 100 == 0:
                evaluation_feed = {F.survival : mini_batch[0].reshape(1,1)}
                #print(np.shape(F.sess.run([F.strategy],feed_dict=evaluation_feed)[0]))
                #break
                food_policies[count] = F.sess.run([F.strategy],feed_dict=evaluation_feed)[0]
                total_consumption[count] = np.sum(food_policies[count])
                count += 1
            
    return log_loss, food_policies, total_consumption
    
