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

"""A binary to train CIFAR-10 using a single GPU.

Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import math
import time

import numpy as np
import matplotlib.pyplot as plt
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import cifar10

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', './ckpt',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('prune_dir', './ckpt_prune_1',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 10000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

name_dir = {"conv1/weights:0": "conv1",
            "conv2/weights:0": "conv2",
            "local3/weights:0": "local3",
            "local4/weights:0": "local4",
            "softmax_linear/weights:0": "softmax"}
              
def plotData(titleName, flatW, numBin=None):
    fig = plt.figure()
    fig.suptitle(titleName)
    curr_plot = fig.add_subplot(111)
#    binBoundaries = np.linspace(minW, maxW, 2**bit + 1)
    if numBin == None:
        curr_plot.hist(flatW[flatW!=0], bins=256, edgecolor='None');
    else:
        curr_plot.hist(flatW[flatW!=0], bins=2**numBin, edgecolor='None');
    curr_plot.set_xlabel('Weight Value')
    curr_plot.set_ylabel('Count')
#    curr_plot.set_xlim(minW, maxW)
    curr_plot.grid(True)
    fig.savefig(titleName + '.pdf')
    plt.close('all')


def pruning(sess, name_and_condition):
    index_w = {}
    for var in tf.trainable_variables():
        if var.name in name_and_condition:
            org_w = sess.run(var)
            ## show information
            print(var.name, "num of non-zero weight before pruning: ", np.count_nonzero(org_w))
            plotData("BefPrune_"+name_dir[var.name], org_w.flatten())
            
            threshold = np.sqrt(0.5 * np.sum(np.power(org_w, 2))) * name_and_condition[var.name]
            #under_threshold = org_w < threshold
            #threshold = np.std(org_w) * name_and_condition[var.name]
            under_threshold = np.absolute(org_w) < threshold
            org_w[under_threshold] = 0

            index_w[var.name] = tf.Variable(tf.constant(-under_threshold, dtype=tf.float32), 
                                            trainable=False, collections=[tf.GraphKeys.PRUNING])
            sess.run(var.assign(tf.convert_to_tensor(org_w)))
            
            ## show information
            print(var.name, "num of non-zero weight after pruning: ", np.count_nonzero(org_w))
            plotData("AftPrune_"+name_dir[var.name], org_w.flatten())
    return index_w


def train():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    # Start running operations on the Graph.
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement))
    #with tf.device("/cpu:0"):

    # Get images and labels for CIFAR-10.
    images, labels = cifar10.distorted_inputs()

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = cifar10.inference(images)

    # Calculate loss.
    loss = cifar10.loss(logits, labels)
    tf.scalar_summary(loss.op.name, loss)

    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver = tf.train.Saver()
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return


    tf.GraphKeys.PRUNING = "PRUNING"
    name_and_condition = {"conv1/weights:0": 0.0001,
                          "conv2/weights:0": 0.0001,
                          "local3/weights:0": 0.0001,
                          "local4/weights:0": 0.0001,
                          "softmax_linear/weights:0": 0.0001}
    # Pruning
    index_w = pruning(sess, name_and_condition)
    sess.run(tf.initialize_variables(tf.get_collection(tf.GraphKeys.PRUNING)))


    global_step = tf.Variable(0, trainable=False, name="global_step")

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = cifar10.train(loss, global_step, name_and_condition, index_w)

    summary_op = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter(FLAGS.prune_dir, sess.graph)

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    for i in tf.all_variables():
      if not sess.run(tf.is_variable_initialized(i)):
        sess.run(tf.initialize_variables([i]))
    
    current_loss = 10.0
    for step in xrange(FLAGS.max_steps):
      start_time = time.time()
      _, loss_value = sess.run([train_op, loss])
      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if (step + 1) % 100 == 0:
        num_examples_per_step = FLAGS.batch_size
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = float(duration)

        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print (format_str % (datetime.now(), step, loss_value,
                             examples_per_sec, sec_per_batch))

        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)

      if loss_value <= current_loss or (step + 1) == FLAGS.max_steps:
          current_loss = loss_value
          checkpoint_path = os.path.join(FLAGS.prune_dir, 'model.ckpt')
          saver.save(sess, checkpoint_path, global_step=step)

    for var in tf.trainable_variables():
        if var.name in name_and_condition:
            org_w = sess.run(var)
            ## show information
            print(var.name, "num of non-zero weight after retaining: ", np.count_nonzero(org_w))
            plotData("AftRetrain_"+name_dir[var.name], org_w.flatten())          
          
          
def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.prune_dir):
    tf.gfile.DeleteRecursively(FLAGS.prune_dir)
  tf.gfile.MakeDirs(FLAGS.prune_dir)
  train()


if __name__ == '__main__':
  tf.app.run()
