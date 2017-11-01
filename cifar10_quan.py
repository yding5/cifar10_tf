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
tf.app.flags.DEFINE_string('quan_dir', './ckpt_quan_6',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 20000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

def plotData(titleName, flatW, numBin=None):
    sortW = np.sort(flatW)
    uniqW = np.unique(sortW)
    print("The # of cluster: " + str(uniqW.shape[0]))
#    for item in uniqW:
#        print("%f : %d" %(item, sortW.tolist().count(item)))

#    fig = plt.figure()
#    fig.suptitle(titleName)
#    curr_plot = fig.add_subplot(111)
##    binBoundaries = np.linspace(minW, maxW, 2**bit + 1)
#    if numBin == None:
#        curr_plot.hist(flatW, bins=256);
#    else:
#        curr_plot.hist(flatW, bins=2**numBin, edgecolor='None');
#    curr_plot.set_xlabel('Weight Value')
#    curr_plot.set_ylabel('Count')
##    curr_plot.set_xlim(minW, maxW)
#    curr_plot.grid(True)
#    fig.savefig(titleName + '.pdf')
#    plt.close('all')


def Quantization(sess, layerAndBit):
    indexWs = {}
    for var in tf.trainable_variables():
        if var.name in layerAndBit:
            #print(var.name)
            orgW = sess.run(var)
            flatW = orgW.flatten()
            indexW = flatW.copy()

            maxW, minW  = np.amax(flatW), np.amin(flatW)
            interval = np.linspace(minW, maxW, 2**layerAndBit[var.name] + 1)
            for i in xrange(2**layerAndBit[var.name]):
                indexE = (flatW >= interval[i]) & (flatW < interval[i+1])
                #if i == (2**layerAndBit[var.name] - 1):
                #    indexE = (flatW >= interval[i]) & (flatW <= interval[i+1])
                indexW[indexE] = i
                if np.any(indexE):
                    flatW[indexE] = np.mean(flatW[indexE])
            #indexWs[var.name] = indexW.astype(np.int32).reshape(var.get_shape())
            ### Reshape index matrix
            indexWs[var.name] = tf.Variable(indexW.astype(np.int32).reshape(var.get_shape()),
                                            trainable=False, collections=[tf.GraphKeys.QUANTABLE])
            ### Modify filters directly
            sess.run(var.assign(tf.convert_to_tensor(flatW.reshape(var.get_shape()))))
    return indexWs


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


    tf.GraphKeys.QUANTABLE = "QUANTABLE"
    layerAndBit = {"conv1/weights:0": 6,
                   "conv2/weights:0": 6}#,
     #             "local3/weights:0": 8,
     #             "local4/weights:0": 8,
     #             "softmax_linear/weights:0": 8}
    # Quantization
    indexWs = Quantization(sess, layerAndBit)
    sess.run(tf.variables_initializer(tf.get_collection(tf.GraphKeys.QUANTABLE)))


    global_step = tf.Variable(0, trainable=False, name="global_step")

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = cifar10.train(loss, global_step, layerAndBit, indexWs)

    summary_op = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter(FLAGS.quan_dir, sess.graph)

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

      #if (step + 1) == FLAGS.max_steps:
        if loss_value <= current_loss:
          current_loss = loss_value
          checkpoint_path = os.path.join(FLAGS.quan_dir, 'model.ckpt')
          saver.save(sess, checkpoint_path, global_step=step)

    for var in tf.trainable_variables():
      if var.name in layerAndBit:
        print(var.name)
        flatW = sess.run(var).flatten()
        plotData("AftReTrain", flatW, layerAndBit[var.name])


def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.quan_dir):
    tf.gfile.DeleteRecursively(FLAGS.quan_dir)
  tf.gfile.MakeDirs(FLAGS.quan_dir)
  train()


if __name__ == '__main__':
  tf.app.run()
