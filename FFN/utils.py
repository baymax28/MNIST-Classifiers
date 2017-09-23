import tensorflow as tf

def get_placeholders(batch_size, input_dimensions, output_dimensions):
	input_placeholder = tf.placeholder(tf.float32, shape=(batch_size, input_dimensions))
	labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
	return input_placeholder, labels_placeholder

def get_feed_dict(input_data, input_placeholder, labels_placeholder, batch_size, fake_data=False):
	# batch_size = int(labels_placeholder.shape[0])
	input_feed, labels_feed = input_data.next_batch(batch_size, fake_data)
	feed_dict = {
		input_placeholder : input_feed,
		labels_placeholder : labels_feed,
	}
	return feed_dict