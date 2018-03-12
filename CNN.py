import tensorflow as tf
import numpy as np
import pandas as pd

#  NOTES Regarding the shape of the tensors after each operation
#Shape of x before reshape <unknown>
#Shape of x afte reshape (?, 32, 32, 32, 1)
#Shape of conv1_1 after maxpool (?, 15, 15, 15, 32)
#Shape of conv1_2 after maxpool (?, 6, 6, 6, 64)
#Shape of conv1_3 after maxpool (?, 6, 6, 6, 64)
#Shape of conv1_4 after reshape (?, 409600)


IMG_PIXELS_SIZE = 32
TOTAL_SLICES = 32

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
               'Weights_conv1_2':tf.Variable(tf.random_normal([3,3,3,32,64])),
               'Weights_conv1_3':tf.Variable(tf.random_normal([3,3,3,64,128])),
              #'Weights_conv1_4':tf.Variable(tf.random_normal([3,3,3,128,256])),                                       

               'Weights_fully_connected':tf.Variable(tf.random_normal([409600,1024])), # 262144 = 64*64*64, 16777216, 175616
               'Output':tf.Variable(tf.random_normal([1024, NUM_CLASSES]))}

    biases = {'Bias_conv1_1':tf.Variable(tf.random_normal([32])),
              'Bias_conv1_2':tf.Variable(tf.random_normal([64])),   
              'Bias_conv1_3':tf.Variable(tf.random_normal([128])),
             #'Bias_conv1_4':tf.Variable(tf.random_normal([256])),   
                                                       
              'Bias_fully_connected':tf.Variable(tf.random_normal([1024])),
              'Output':tf.Variable(tf.random_normal([NUM_CLASSES]))}

    print('Shape of x before reshape', x.get_shape())
    x = tf.reshape(x, shape=[-1, IMG_PIXELS_SIZE, IMG_PIXELS_SIZE, TOTAL_SLICES, 1])
    print('Shape of x afte reshape', x.get_shape())
    
    conv1_1 = tf.nn.relu(conv3d(x, weights['Weights_conv1_1']) + biases['Bias_conv1_1'])
    conv1_1 = maxpool3d(conv1_1)
    print('Shape of conv1_1 after maxpool', conv1_1.get_shape())
    
    
    conv1_2 = tf.nn.relu(conv3d(conv1_1, weights['Weights_conv1_2']) + biases['Bias_conv1_2'])
    conv1_2 = maxpool3d(conv1_2)
    print('Shape of conv1_2 after maxpool', conv1_2.get_shape())

    
    conv1_3 = tf.nn.relu(conv3d(conv1_2, weights['Weights_conv1_3']) + biases['Bias_conv1_3'])
    conv1_3 = maxpool3d(conv1_3)
    print('Shape of conv1_3 after maxpool', conv1_2.get_shape())

  
    #conv1_4 = tf.nn.relu(conv3d(conv1_3, weights['Weights_conv1_4']) + biases['Bias_conv1_4'])
    #conv1_4 = maxpool3d(conv1_4)
    #print('Shape of conv1_4 after maxpool', conv1_4.get_shape())
   
    
    fc = tf.reshape(conv1_3, [-1, 409600]) #262144 #Reshape from [BATCH_SIZE,a,b] to [BATCH_SIZE, a*b]
    print('Shape of conv1_4 after reshape', fc.get_shape())    
    fc = tf.nn.relu(tf.matmul(fc, weights['Weights_fully_connected']) + biases['Bias_fully_connected'])
    fc = tf.nn.dropOutput(fc, keep_rate)

    Outputput = tf.matmul(fc, weights['Output'])+biases['Output']

    return Outputput


much_data = np.load('C:/Users/taimurzahid/Pictures/muchdata-64-64-64.npy')
train_data = much_data[:-100]
validation_data = much_data[-100:]

Result=[]

def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits = prediction ,labels = y) )
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)
    
    hm_epochs = 100
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        
        successful_runs = 0
        total_runs = 0
        
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for num,data in enumerate(train_data):
                if (num%100==0):
                    print(num)
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

train_neural_network(x)

# Write Result to csv
a= pd.DataFrame(Result,columns=['id','cancer','result'])
print(a)
a.to_csv('D:/final/Result_images.csv')
