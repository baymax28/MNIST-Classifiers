import tensorflow as tf

import argparse
import os
import sys
import time

from tensorflow.examples.tutorials.mnist import input_data

import mnistFFN

def get_placeholders(batch_size, input_dimensions, output_dimensions):
	input_placeholder = tf.placeholder(tf.float32, shape=(batch_size, input_dimensions))
	labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
	return input_placeholder, labels_placeholder

def get_feed_dict(input_data, input_placeholder, labels_placeholder, fake_data=False):
	batch_size = int(labels_placeholder.shape[0])
	input_feed, labels_feed = input_data.next_batch(batch_size, fake_data)
	feed_dict = {
		input_placeholder : input_feed,
		labels_placeholder : labels_feed,
	}
	return feed_dict

def do_eval(sess, input_data, eval_correct, input_placeholder, labels_placeholder):
	steps_per_epoch = input_data.num_examples / FLAGS.batch_size
	correct = 0
	
	num_examples = steps_per_epoch * FLAGS.batch_size
	
	for _ in xrange(steps_per_epoch):
		feed_dict = get_feed_dict(input_data, input_placeholder, labels_placeholder, FLAGS.fake_data)
		correct += sess.run(eval_correct, feed_dict = feed_dict)
	precision = float(correct) / num_examples
	print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' % (num_examples, correct, precision))


def do_training():
	data_sets = input_data.read_data_sets(FLAGS.input_data_dir, FLAGS.fake_data)

	with tf.Graph().as_default():
		input_placeholder, labels_placeholder = get_placeholders(FLAGS.batch_size, mnistFFN.IMAGE_PIXELS, mnistFFN.NUM_CLASSES)

		logits = mnistFFN.feed_forward_model(input_placeholder, FLAGS.hidden1_nodes, FLAGS.hidden2_nodes)

		loss = mnistFFN.loss(logits, labels_placeholder)

		train_op = mnistFFN.training(loss, FLAGS.learning_rate)

		eval_correct = mnistFFN.evaluation(logits, labels_placeholder)

		init = tf.global_variables_initializer()

		summary_all = tf.summary.merge_all()

		sess = tf.Session()

		summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

		saver = tf.train.Saver();

		saver.restore(sess, tf.train.latest_checkpoint('../models'))

		print("Model restores from " + tf.train.latest_checkpoint('../models'))
		for step in xrange(FLAGS.max_steps):

			if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
				do_eval(sess, data_sets.train, eval_correct, input_placeholder, labels_placeholder)
				do_eval(sess, data_sets.validation, eval_correct, input_placeholder, labels_placeholder)
				do_eval(sess, data_sets.test, eval_correct, input_placeholder, labels_placeholder)


def main(_):
	if tf.gfile.Exists(FLAGS.log_dir):
		tf.gfile.DeleteRecursively(FLAGS.log_dir)
	tf.gfile.MakeDirs(FLAGS.log_dir)
	do_training()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
			'--learning_rate',
			type=float,
			default=0.01,
			help='Initial learning rate.'
	)
	parser.add_argument(
			'--max_steps',
			type=int,
			default=2000,
			help='Number of steps to run trainer.'
	)
	parser.add_argument(
			'--hidden1_nodes',
			type=int,
			default=128,
			help='Number of units in hidden layer 1.'
	)
	parser.add_argument(
			'--hidden2_nodes',
			type=int,
			default=32,
			help='Number of units in hidden layer 2.'
	)
	parser.add_argument(
			'--batch_size',
			type=int,
			default=100,
			help='Batch size.  Must divide evenly into the dataset sizes.'
	)
	parser.add_argument(
			'--input_data_dir',
			type=str,
			default='../data',
			help='Directory to put the input data.'
	)
	parser.add_argument(
			'--log_dir',
			type=str,
			default='./log_data',
			help='Directory to put the log data.'
	)
	parser.add_argument(
			'--fake_data',
			default=False,
			help='If true, uses fake data for unit testing.',
			action='store_true'
	)

	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)