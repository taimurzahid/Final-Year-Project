import tensorflow as tf
import numpy as np
import pandas as pd

# NOTES Regarding the shape of the tensors after each operation
#Shape of conv1_2 after maxpool (?, 14, 14, 14, 64)
#Shape of conv2_2 after maxpool (?, 14, 14, 14, 64)
#Shape of conv1_2 after transpose (?, 64, 14, 14, 14)
#Shape of conv1_2 after reshape (?, 64, 175616)
#Shape of conv2_2 after transpose (?, 64, 14, 14, 14)
#Shape of conv2_2 after reshape (?, 64, 175616)
#Shape of conv2_2_T after transpose (?, 175616, 64)
#Shape of convFinal after matmul (?, 64, 64)
#Shape of convFinal after reshape (?, 64, 64)

IMG_PIXELS_SIZE = 64
TOTAL_SLICES = 64

NUM_CLASSES = 2
BATCH_SIZE = 10

x = tf.placeholder('float')
y = tf.placeholder('float')

keep_rate = 0.8

def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1,1,1,1,1], padding='VALID')

def maxpool3d(x):
    return tf.nn.max_pool3d(x, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='VALID')

def convolutional_neural_network(x):
    weights = {'Weights_conv1_1':tf.Variable(tf.random_normal([3,3,3,1,32])),
               'Weights_conv2_1':tf.Variable(tf.random_normal([3,3,3,1,32])),
              
               'Weights_conv1_2':tf.Variable(tf.random_normal([3,3,3,32,64])),
               'Weights_conv2_2':tf.Variable(tf.random_normal([3,3,3,32,64])),
              
               'Weights_fully_connected':tf.Variable(tf.random_normal([11239424,1024])), 
               'Output':tf.Variable(tf.random_normal([1024, NUM_CLASSES]))}

    biases = {'Bias_conv1_1':tf.Variable(tf.random_normal([32])),
              'Bias_conv2_1':tf.Variable(tf.random_normal([32])),
              
              'Bias_conv1_2':tf.Variable(tf.random_normal([64])),
              'Bias_conv2_2':tf.Variable(tf.random_normal([64])),
              
              'Bias_fully_connected':tf.Variable(tf.random_normal([1024])),
              'Output':tf.Variable(tf.random_normal([NUM_CLASSES]))}

    x = tf.reshape(x, shape=[-1, IMG_PIXELS_SIZE, IMG_PIXELS_SIZE, TOTAL_SLICES, 1])

    conv1_1 = tf.nn.relu(conv3d(x, weights['Weights_conv1_1']) + biases['Bias_conv1_1'])
    print (conv1_1.get_shape()); #(?, 48, 48, 18, 32)
    conv1_1 = maxpool3d(conv1_1)
    print (conv1_1.get_shape()); #(?, 24, 24, 9, 32)
    
    conv2_1 = tf.nn.relu(conv3d(x, weights['Weights_conv2_1']) + biases['Bias_conv2_1'])
    print (conv2_1.get_shape()); #(?, 48, 48, 18, 32)
    conv2_1 = maxpool3d(conv2_1)
    print (conv2_1.get_shape()); #(?, 24, 24, 9, 32)
    
    conv1_2 = tf.nn.relu(conv3d(conv1_1, weights['Weights_conv1_2']) + biases['Bias_conv1_2'])
    print (conv1_2.get_shape()); #(?, 22, 22, 7, 64)
    conv1_2 = maxpool3d(conv1_2)
    print('Shape of conv1_2 after maxpool', conv1_2.get_shape())
    #print (conv1_2.get_shape()); #(?, 11, 11, 3, 64)

    conv2_2 = tf.nn.relu(conv3d(conv2_1, weights['Weights_conv2_2']) + biases['Bias_conv2_2'])
    print (conv2_2.get_shape()); #(?, 22, 22, 7, 64)
    conv2_2 = maxpool3d(conv2_2)
    print('Shape of conv2_2 after maxpool', conv2_2.get_shape())
    #print (conv2_2.get_shape()); (?, 11, 11, 3, 64)
    
    #print (conv1_2.get_shape()); (?, 14, 14, 14, 64)
    #print (conv2_2.get_shape()); (?, 14, 14, 14, 64)
    
    #SUMMING TO CHECK whether they can be combined.
    #convFinal = conv1_2 + conv2_2; 
    
    
    #outer product
    #conv1_2 = tf.transpose(conv1_2)       
                                          
    #conv1_2 = tf.reshape(conv1_2,[-1,23232])  
    #conv2_2 = tf.transpose(conv2_2)        
    #conv2_2 = tf.reshape(conv2_2,[-1,23232])  

    conv1_2 = tf.transpose(conv1_2, perm=[0,4,1,2,3])       
    print('Shape of conv1_2 after transpose', conv1_2.get_shape())   
    #print (conv1_2.get_shape()); (?, 64, 11, 11, 3)
    # 64*64*64 = 262144
    # 14*14*14 = 2744
    conv1_2 = tf.reshape(conv1_2, [-1, 64, 175616])            
    print('Shape of conv1_2 after reshape', conv1_2.get_shape())                                                              
    #conv1_2_T = tf.transpose(conv1_2, perm=[0,2,1])            
    
    
                                                            
    #convFinal = tf.matmul(conv1_2, conv1_2_T)                 

    conv2_2 = tf.transpose(conv2_2, perm=[0,4,1,2,3])       
    print('Shape of conv2_2 after transpose', conv2_2.get_shape()) 
                                                              
    #print (conv1_2.get_shape()); (?, 64, 11, 11, 3)
    conv2_2 = tf.reshape(conv2_2, [-1, 64, 175616])            
    print('Shape of conv2_2 after reshape', conv2_2.get_shape())   
                                                           
    conv2_2_T = tf.transpose(conv2_2, perm=[0,2,1])            
    print('Shape of conv2_2_T after transpose', conv2_2_T.get_shape())
    
                                                            
    convFinal = tf.matmul(conv1_2, conv2_2_T)                 
    print('Shape of convFinal after matmul', convFinal.get_shape())

    #convFinal = tf.reshape(convFinal,[-1,175616])    #175616       
    #print('Shape of convFinal after reshape', convFinal.get_shape())

    #convFinal = tf.divide(convFinal,784.0)  
    #print('Shape convFinal after division', convFinal.get_shape())  

    #convFinal = tf.matmul(conv1_2[1], conv2_2[1])  

    #print (convFinal.get_shape()); (?, 11, 11, 3, 64)
    #transpose_a = False, transpose_b = False, adjoint_a = False, 
    #adjoint_b = False, a_is_sparse = False, b_is_sparse = False,name = None
    
    fc = tf.reshape(convFinal,[-1, 11239424]) #262144 #Reshape from [BATCH_SIZE,512,512] to [BATCH_SIZE, 512*512]
    print('Shape of convFinal after reshape', convFinal.get_shape())    
    fc = tf.nn.relu(tf.matmul(fc, weights['Weights_fully_connected']) + biases['Bias_fully_connected'])
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['Output'])+biases['Output']

    return output


much_data = np.load('C:/Users/taimurzahid/Pictures/muchdata-64-64-64.npy')
train_data = much_data[:-100]
validation_data = much_data[-100:]

Result=[]

def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits = prediction ,labels = y) )
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)
    
    hm_epochs = 10
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        
        successful_runs = 0
        total_runs = 0
        
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for data in train_data:
                total_runs += 1
                try:
                    X = data[0]
                    Y = data[1]
                    _, c = sess.run([optimizer, cost], feed_dict={x: X, y: Y})
                    epoch_loss += c
                    successful_runs += 1
                except Exception as e:
                    # this may be caused due to the number of slices of a particular patient may not be equal to the specified by the variable TOTAL_SLICES
                    pass

            print('Epoch number: ', epoch+1, 'completed out of ',hm_epochs,' loss: ',epoch_loss)

            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

            print('Accuracy: ',accuracy.eval({x:[i[0] for i in validation_data], y:[i[1] for i in validation_data]}))
            
        print('Done. Finishing accuracy:')
        print('Accuracy:',accuracy.eval({x:[i[0] for i in validation_data], y:[i[1] for i in validation_data]}))
        
        print('fitment percent:',successful_runs/total_runs)
        
        array=sess.run(correct,feed_dict={x:[i[0] for i in validation_data], y:[i[1] for i in validation_data]})

        for i in range(len(validation_data)):
            if(validation_data[i][1][0]==1):
                Result.append([validation_data[i][2],0,array[i]])
            else:
                Result.append([validation_data[i][2],1,array[i]])

# Run this locally:
train_neural_network(x)

## Writes Result to "Outputput.csv"
a= pd.DataFrame(Result,columns=['id','cancer','result'])
print(a)
a.to_csv('D:/final/Result_images.csv')
