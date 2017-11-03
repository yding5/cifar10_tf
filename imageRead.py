#first we summary the input image by queue

#first file path
file_path='../data/cifar10/cifar-10-batches-bin/data_batch_1.bin'
data_dir='../data/cifar10/cifar-10-batches-bin'
summary_dir='./summary'

#import lib
import tensorflow as tf


IMAGE_SIZE=24
NUM_CLASSES=10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000

#define input image format
class CifarData(object):
    pass

img=CifarData()
img.height = 32
img.width = 32
img.depth = 3

img_bytes=img.height * img.width * img.depth

label_bytes=1

record_bytes=label_bytes+img_bytes

#define filequeue
file_queue=tf.train.string_input_producer(['../data/cifar10/cifar-10-batches-bin/data_batch_1.bin',
                                           '../data/cifar10/cifar-10-batches-bin/data_batch_2.bin'])

#read data from queue
reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
img.key,value = reader.read(file_queue)
num_recs = reader.num_records_produced()
record_bytes=tf.decode_raw(value,tf.uint8)
img.label=tf.cast(tf.slice(record_bytes,[0],[label_bytes]),tf.int32)
depth_major=tf.reshape(tf.slice(record_bytes,[label_bytes],[img_bytes]),[img.depth,img.height,img.width])
img.unit8image=tf.transpose(depth_major,[1,2,0])

img_summary=tf.reshape(img.unit8image, [-1,32,32,3])
tf.image_summary('img',img_summary,10)

reshaped_img=tf.cast(img.unit8image, tf.float32)
height = IMAGE_SIZE
width=IMAGE_SIZE

distorted_img=tf.random_crop(reshaped_img,[height,width,3])
distorted_img=tf.image.random_flip_left_right(distorted_img)
distorted_img=tf.image.random_brightness(distorted_img,max_delta=63)
distorted_img=tf.image.random_contrast(distorted_img,lower=0.2,upper=1.8)
distorted_img_summary=tf.reshape(distorted_img,[-1, height,width,3])
tf.image_summary('img1',distorted_img_summary,10)

float_img= tf.image.per_image_whitening(distorted_img)
min_queue_examples=20000
num_preprocess_threads = 16
images=tf.train.shuffle_batch([float_img],128,num_threads=num_preprocess_threads,capacity=min_queue_examples+3*128,min_after_dequeue=min_queue_examples)
tf.image_summary('images',images,10)

sess = tf.InteractiveSession()

tf.initialize_all_variables().run()

merged=tf.merge_all_summaries()

print('begin to start queue')
tf.train.start_queue_runners(sess=sess)
print('queue has been started')

writer=tf.train.SummaryWriter(summary_dir,sess.graph)

for i in range(100):
    summary, num_, img_, key_ = sess.run([merged,num_recs,value,img.key])
    print("step%d: %s  %d"%(i,key_,num_))
    writer.add_summary(summary,i)
    writer.flush()