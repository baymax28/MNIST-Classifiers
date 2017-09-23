import tensorflow as tf

import argparse
import os
import sys
import time

from tensorflow.examples.tutorials.mnist import input_data

import utils
import mnistFFN

def do_eval(sess, input_data, eval_correct, input_placeholder, labels_placeholder):
	steps_per_epoch = input_data.num_examples / FLAGS.batch_size
	num_examples = steps_per_epoch * FLAGS.batch_size

	for _ in xrange(steps_per_epoch):
		feed_dict = utils.get_feed_dict(input_data, input_placeholder, labels_placeholder, FLAGS.fake_data)
		correct += sess.run([eval_correct], feed_dict = feed_dict)
	precision = correct / num_examples
	print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' % (num_examples, true_count, precision))


def do_training():
	data_sets = input_data.read_data_sets(FLAGS.input_data_dir, FLAGS.fake_data)

	with tf.Graph().as_default():
		input_placeholder, labels_placeholder = utils.get_placeholders(FLAGS.batch_size, mnistFFN.IMAGE_PIXELS, mnistFFN.NUM_CLASSES)

		logits = mnistFFN.feed_forward_model(input_placeholder, FLAGS.hidden1_nodes, FLAGS.hidden2_nodes)

		loss = mnistFFN.loss(logits, labels_placeholder)

		train_op = mnistFFN.training(loss, FLAGS.learning_rate)

		eval_correct = mnistFFN.evaluation(logits, labels_placeholder)

		init = tf.global_variables_initializer()

		sess = tf.Session()

		sess.run(init)

		for steps in max_steps:
			start_time = time.time()
			feed_dict = utils.get_feed_dict(data_sets.train, input_placeholder, labels_placeholder, FLAGS.fake_data)
			_, loss = sess.run([train_op, loss], feed_dict=feed_dict)
			duration = time.time() - start_time

			if steps % 100 == 0:
				print('Step %d: loss = %0.2f (%0.2f)' % (steps, loss, duration))

			if steps % 1000 == 0 or (steps + 1) == max_steps:
				do_eval(sess, data_sets.trian, eval_correct, input_placeholder, labels_placeholder)
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
      '--nEpochs',
      type=int,
      default=2000,
      help='Number of epochs to run trainer.'
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
      default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                           'tensorflow/mnist/input_data'),
      help='Directory to put the input data.'
  )
  parser.add_argument(
      '--log_dir',
      type=str,
      default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                           'tensorflow/mnist/logs/fully_connected_feed'),
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