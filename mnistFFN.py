import tensorflow as tf 

NUM_CLASSES = 10

IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

def feed_forward_model(images, hidden1_nodes, hidden2_nodes):
	
	with tf.name_scope('Hidden1'):
		weights = tf.Variable(tf.random_normal([IMAGE_PIXELS, hidden1_nodes]), name='weights')
		biases = tf.Variable(tf.random_normal([hidden1_nodes]), name='biases')
		hidden1 = tf.nn.relu(tf.add(tf.matmul(images, weights), biases))

	with tf.name_scope('Hidden2'):
		weights = tf.Variable(tf.random_normal([hidden1_nodes, hidden2_nodes]), name='weights')
		biases = tf.Variable(tf.random_normal([hidden2_nodes]), name='biases')
		hidden2 = tf.nn.relu(tf.add(tf.matmul(hidden1, weights), biases))

	with tf.name_scope('Logits'):
		weights = tf.Variable(tf.random_normal([hidden2_nodes, NUM_CLASSES]), name='weights')
		biases = tf.Variable(tf.random_normal([NUM_CLASSES]), name='biases')
		logits = tf.add(tf.matmul(hidden2, weights), biases)

	return logits

def loss(logits, labels):
	loss_tensor = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
	loss = tf.reduce_mean(loss_tensor)
	return loss

def training(loss, learning_rate):
	tf.summary.scalar('Loss', loss)
	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	global_step = tf.Variable(0, name='global_step', trainable='false')
	train_op = optimizer.minimize(loss, global_step=global_step)
	return train_op

def evaluation(logits, labels):
	correct = tf.nn.in_top_k(logits, labels, k=1)
	return tf.reduce_sum(tf.cast(correct, tf.int32))