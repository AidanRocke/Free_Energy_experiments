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

import matplotlib.pyplot as plt
from free_energy_agent import free_agent

## set random seed:
random.seed(42)
tf.set_random_seed(42)

## define number of epochs:
epochs = 1000 ### must be a perfect square
batch_size = 50

## define main parameters:
basic_needs = 5.0
success_probability = 1.0

def train(epochs,batch_size,basic_needs,success_probability):
    # the concatenation of a state and action
    count = 0
    
    ## define arrays to contain means and variances:
    N = int(epochs/100)
    
    hunting_priors = basic_needs*np.ones(N)
    total_consumption = np.zeros(N)
        
    with tf.Session() as sess:
                
        F = free_agent(basic_needs,sess,success_probability) 
        
        ### it might be a good idea to regularise the squared loss:
        surprisal = tf.reduce_mean(tf.multiply(tf.constant(-1.0),F.surprise())) 
        
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
                total_consumption[count] = np.sum(F.sess.run([F.strategy],feed_dict=train_feed))
                count += 1
            
    return hunting_priors, total_consumption
    
