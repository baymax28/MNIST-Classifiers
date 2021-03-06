import tensorflow as tf 

NUM_CLASSES = 10

IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

def feed_forward_model(images, hidden1_nodes, hidden2_nodes):
	
	with tf.name_scope('Hidden1'):
		weights = tf.Variable(tf.random_normal([IMAGE_PIXELS, hidden1_nodes])/tf.sqrt(float(IMAGE_PIXELS)/2), name='weights')
		# weights = tf.Variable(tf.random_normal([IMAGE_PIXELS, hidden1_nodes]), name='weights')
		templates = tf.reshape(tf.transpose(weights), [hidden1_nodes, IMAGE_SIZE, IMAGE_SIZE, 1])
		tf.summary.image('Trained templates', templates, max_outputs=hidden1_nodes)
		biases = tf.Variable(tf.random_normal([hidden1_nodes]), name='biases')
		hidden1 = tf.nn.relu(tf.add(tf.matmul(images, weights), biases))
		# hidden1 = tf.nn.sigmoid(tf.add(tf.matmul(images, weights), biases))
		# hidden1 = tf.nn.tanh(tf.add(tf.matmul(images, weights), biases))
		tf.summary.histogram('FirstLayerActivation', hidden1)

	with tf.name_scope('Hidden2'):
		weights = tf.Variable(tf.random_normal([hidden1_nodes, hidden2_nodes])/tf.sqrt(float(hidden1_nodes)/2), name='weights')
		# weights = tf.Variable(tf.random_normal([hidden1_nodes, hidden2_nodes]), name='weights')
		biases = tf.Variable(tf.random_normal([hidden2_nodes]), name='biases')
		hidden2 = tf.nn.relu(tf.add(tf.matmul(hidden1, weights), biases))
		# hidden2 = tf.nn.sigmoid(tf.add(tf.matmul(hidden1, weights), biases))
		# hidden2 = tf.nn.tanh(tf.add(tf.matmul(hidden1, weights), biases))
		tf.summary.histogram('SecondLayerActivation', hidden2)

	with tf.name_scope('Logits'):
		weights = tf.Variable(tf.random_normal([hidden2_nodes, NUM_CLASSES]), name='weights')
		biases = tf.Variable(tf.random_normal([NUM_CLASSES]), name='biases')
		logits = tf.add(tf.matmul(hidden2, weights), biases)

	return logits

def feed_forward_model_single_layer(images, hidden1_nodes, hidden2_nodes):

	with tf.name_scope('Logits'):
		# weights = tf.Variable(tf.random_normal([IMAGE_PIXELS, NUM_CLASSES]), name='weights')
		# weights = tf.Variable(tf.random_normal([IMAGE_PIXELS, NUM_CLASSES])/tf.sqrt(float(IMAGE_PIXELS)/2), name='weights')
		weights = tf.Variable(tf.zeros([IMAGE_PIXELS, NUM_CLASSES]))
  		templates = tf.reshape(tf.transpose(weights), [NUM_CLASSES, IMAGE_SIZE, IMAGE_SIZE, 1])
		tf.summary.image('Trained templates', templates, max_outputs=NUM_CLASSES)
		# biases = tf.Variable(tf.random_normal([NUM_CLASSES]), name='biases')
		biases = tf.Variable(tf.zeros([NUM_CLASSES]))
		# logits = tf.nn.relu(tf.add(tf.matmul(images, weights), biases))
		logits = tf.add(tf.matmul(images, weights), biases)

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