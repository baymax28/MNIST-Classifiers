import tensorflow as tf 

NUM_CLASSES = 10

IMAGE_DEPTH = 1
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

def convolutional_model(images, hidden1_filter_span, hidden2_filter_span, hidden1_filter_number, hidden2_filter_number, fully_conected_number):
	
	images_reshaped = tf.reshape(images, [-1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_DEPTH])

	with tf.name_scope('Hidden1'):
		weights = tf.Variable(tf.truncated_normal([hidden1_filter_span, hidden1_filter_span, IMAGE_DEPTH, hidden1_filter_number]), name='weights')
		biases = tf.Variable(tf.truncated_normal([hidden1_filter_number]), name='biases')
		hiddenConv1 = tf.nn.relu(tf.add(tf.nn.conv2d(images_reshaped, weights, strides = [1, 1, 1, 1], padding='SAME'), biases))
		hiddenPool1 = tf.nn.max_pool(hiddenConv1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding='SAME')

	with tf.name_scope('Hidden2'):
		weights = tf.Variable(tf.truncated_normal([hidden2_filter_span, hidden2_filter_span, hidden1_filter_number, hidden2_filter_number]), name='weights')
		biases = tf.Variable(tf.truncated_normal([hidden2_filter_number]), name='biases')
		hiddenConv2 = tf.nn.relu(tf.add(tf.nn.conv2d(hiddenPool1, weights, strides = [1, 1, 1, 1], padding='SAME'), biases))
		hiddenPool2 = tf.nn.max_pool(hiddenConv2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding='SAME')

	with tf.name_scope('FullyConnected'):
		hiddenPool2_flat = tf.reshape(hiddenPool2, [-1, tf.shape(hiddenPool2)[1]*tf.shape(hiddenPool2)[2]*tf.shape(hiddenPool2)[3]])
		weights = tf.Variable(tf.truncated_normal([3136, fully_conected_number]), name='weights')
		biases = tf.Variable(tf.truncated_normal([fully_conected_number]), name='biases')
		hiddenFullyConnected = tf.nn.relu(tf.add(tf.matmul(hiddenPool2_flat, weights), biases))

	with tf.name_scope('Logits'):
		weights = tf.Variable(tf.random_normal([fully_conected_number, NUM_CLASSES]), name='weights')
		biases = tf.Variable(tf.random_normal([NUM_CLASSES]), name='biases')
		logits = tf.add(tf.matmul(hiddenFullyConnected, weights), biases)

	return logits

def loss(logits, labels):
	loss_tensor = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
	loss = tf.reduce_mean(loss_tensor)
	return loss

def training(loss, learning_rate):
	tf.summary.scalar('Loss', loss)
	# optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	optimizer = tf.train.AdamOptimizer(learning_rate)
	global_step = tf.Variable(0, name='global_step', trainable='false')
	train_op = optimizer.minimize(loss, global_step=global_step)
	return train_op

def evaluation(logits, labels):
	correct = tf.nn.in_top_k(logits, labels, k=1)
	return tf.reduce_sum(tf.cast(correct, tf.int32))