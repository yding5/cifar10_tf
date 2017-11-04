# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Evaluation for CIFAR-10.

Accuracy:
cifar10_train.py achieves 83.0% accuracy after 100K steps (256 epochs
of data) as judged by cifar10_eval.py.

Speed:
On a single Tesla K40, cifar10_train.py processes a single batch of 128 images
in 0.25-0.35 sec (i.e. 350 - 600 images /sec). The model reaches ~86%
accuracy after 100K steps in 8 hours of training time.

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
#import matplotlib.pyplot as plt
import tensorflow as tf

import cifar10

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', './eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', './ckpt_prune_1',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 5 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 10000,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', True,
                         """Whether to run eval only once.""")

name_dir = {"conv1/weights:0": "conv1",
            "conv2/weights:0": "conv2",
            "local3/weights:0": "local3",
            "local4/weights:0": "local4",
            "softmax_linear/weights:0": "softmax"}

#def plotData(titleName, flatW, numBin=None):
#    sortW = np.sort(flatW)
#    uniqW = np.unique(sortW)
#    print("The # of cluster: " + str(uniqW.shape[0]))
#    for item in uniqW:
#        print("%f : %d" %(item, sortW.tolist().count(item)))

#DYK
#     fig = plt.figure()
#     fig.suptitle(titleName)
#     curr_plot = fig.add_subplot(111)
# #    binBoundaries = np.linspace(minW, maxW, 2**bit + 1)
#     if numBin == None:
#         curr_plot.hist(flatW[flatW!=0], bins=256, edgecolor='None');
#     else:
#         curr_plot.hist(flatW[flatW!=0], bins=2**numBin, edgecolor='None');
#     curr_plot.set_xlabel('Weight Value')
#     curr_plot.set_ylabel('Count')
# #    curr_plot.set_xlim(minW, maxW)
#     curr_plot.grid(True)
#     fig.savefig(titleName + '.pdf')
#     plt.close('all')


def quanization(sess, name_and_condition):
    index_w = {}
    for var in tf.trainable_variables():
        if var.name in name_and_condition:
            #print(var.name)
            org_w = sess.run(var)
            flat_w = org_w.flatten()
            copy_w = flat_w.copy()

            max_w, min_w  = np.amax(flat_w), np.amin(flat_w)
            interval = np.linspace(min_w, max_w, 2**name_and_condition[var.name] + 1)
            for i in xrange(2**name_and_condition[var.name]):
                indexW = (flat_w >= interval[i]) & (flat_w < interval[i+1])
                if i == (2**name_and_condition[var.name] - 1):
                    indexW = (flat_w >= interval[i]) & (flat_w <= interval[i+1])
                copy_w[indexW] = i
                if np.any(indexW):
                    flat_w[indexW] = np.mean(flat_w[indexW])
            #index_w[var.name] = copy_w.astype(np.int32).reshape(var.get_shape())
            index_w[var.name] = tf.Variable(copy_w.astype(np.int32).reshape(var.get_shape()),
                                            trainable=False, collections=[tf.GraphKeys.QUANTABLE])
            sess.run(var.assign(tf.convert_to_tensor(flat_w.reshape(var.get_shape()))))
    return index_w

    
def pruning(sess, name_and_condition):
    index_w = {}
    for var in tf.trainable_variables():
        if var.name in name_and_condition:
            org_w = sess.run(var)
            ## show information
            print(var.name, "-num of non-zero weight before pruning: ", np.count_nonzero(org_w))
            #plotData("BefPrune_"+name_dir[var.name], org_w.flatten())
             
            threshold = np.std(org_w) * name_and_condition[var.name]
            under_threshold = np.absolute(org_w) < threshold
            org_w[under_threshold] = 0

            index_w[var.name] = tf.Variable(tf.constant(-under_threshold, dtype=tf.float32), 
                                            trainable=False, collections=[tf.GraphKeys.PRUNING])
            sess.run(var.assign(tf.convert_to_tensor(org_w)))
            
            ## show information
            print(var.name, "-num of non-zero weight after pruning: ", np.count_nonzero(org_w))
            #plotData("AftPrune_"+name_dir[var.name], org_w.flatten()) 
    return index_w

    
def eval_once(saver, summary_writer, top_k_op, summary_op, error_max_op, error_sum_op, sm_error_max_op, sm_error_sum_op):
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    print (FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return


#    tf.GraphKeys.QUANTABLE = "QUANTABLE"
#    name_and_condition = {"conv1/weights:0": 8,
#                          "conv2/weights:0": 8}
#                         "local3/weights:0": 8,
#                         "local4/weights:0": 8,
#                         "softmax_linear/weights:0": 8}
#    # Quanization
#    index_w = quanization(sess, name_and_condition)

    tf.GraphKeys.PRUNING = "PRUNING"
    name_and_condition = {"conv1/weights:0": 0.39,
                          "conv2/weights:0": 0.39,
                          "local3/weights:0": 0.39,
                          "local4/weights:0": 0.39,
                          "softmax_linear/weights:0": 0.39}
    # Pruning
    #index_w = pruning(sess, name_and_condition)
    #sess.run(tf.initialize_variables(tf.get_collection(tf.GraphKeys.PRUNING)))
    
        
    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))
      start_time = time.time()
      num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
      true_count = 0  # Counts the number of correct predictions.

      sm_error_sum_count = 0

      total_sample_count = num_iter * FLAGS.batch_size
      step = 0
      while step < num_iter and not coord.should_stop():
        predictions = sess.run([top_k_op])
        true_count += np.sum(predictions)

        sm_error_sum = sess.run([sm_error_sum_op])
        sm_error_sum_count += np.sum(sm_error_sum)

        step += 1
      duration = time.time() - start_time
      
      # Compute precision @ 1.
      precision = true_count / total_sample_count
      print('%s: precision @ 1 = %.3f, druation = %.6f' % (datetime.now(), precision, duration))
      sm_error_mean = sm_error_sum_count / total_sample_count
      print('%s: mean error after softmax  = %.3f' % (datetime.now(), sm_error_mean))

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision @ 1', simple_value=precision)
      summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)
        
        
def evaluate():
  """Eval CIFAR-10 for a number of steps."""
  with tf.Graph().as_default() as g:
    # Get images and labels for CIFAR-10.
    eval_data = FLAGS.eval_data == 'test'
    images, labels = cifar10.inputs(eval_data=eval_data)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = cifar10.inference(images)

    # Calculate predictions.
    top_k_op = tf.nn.in_top_k(logits, labels, 1)
    # Calcuate the error Y.D.
    print(logits.get_shape().as_list())

    label_matrix = tf.one_hot(labels,10)

    error_op = tf.subtract(logits,tf.to_float(label_matrix))
    error_max_op = tf.reduce_max(error_op,axis=1)
    error_sum_op = tf.reduce_sum(error_op)
    print(error_max_op.get_shape().as_list())
    print(error_sum_op.get_shape().as_list())

    softmax_logits = tf.nn.softmax (logits)
    sm_error_op = tf.subtract(softmax_logits,tf.to_float(label_matrix))
    sm_error_max_op = tf.reduce_max(sm_error_op,axis=1)
    sm_error_sum_op = tf.reduce_sum(sm_error_op)
    print(sm_error_max_op.get_shape().as_list())
    print(sm_error_sum_op.get_shape().as_list())

    # Restore the moving average version of the learned variables for eval.
    #variable_averages = tf.train.ExponentialMovingAverage(
    #    cifar10.MOVING_AVERAGE_DECAY)
    #variables_to_restore = variable_averages.variables_to_restore()
    #saver = tf.train.Saver(variables_to_restore)
    saver = tf.train.Saver()
    
    # Build the summary operation based on the TF collection of Summaries.
    summary_op =  tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

    while True:
      eval_once(saver, summary_writer, top_k_op, summary_op, error_max_op, error_sum_op, sm_error_max_op, sm_error_sum_op)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)
    
    
def main(argv=None):  # pylint: disable=unused-argument
  cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  evaluate()


if __name__ == '__main__':
  tf.app.run()
