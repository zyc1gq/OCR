import datetime
import logging
import os
import time


import numpy as np
import tensorflow as tf
import cnn_lstm_ctc_ocr
import utils
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

FLAGS = utils.FLAGS

logger = logging.getLogger('Traing for OCR using LSTM+CTC')
logger.setLevel(logging.INFO)


def train(train_dir=None, val_dir=None, mode='train'):
    model = cnn_lstm_ctc_ocr.GRUOCR(mode)
    model.build_graph()

    print('loading train data, please wait---------------------')
    train_feeder = utils.DataIterator(img_dir=train_dir,lab_dir= FLAGS.tra_lab,rpath="image/",t_num=8200000)
    print('get image: ', train_feeder.size)

    print('loading validation data, please wait---------------------')
    val_feeder = utils.DataIterator(img_dir=val_dir,lab_dir=FLAGS.val_lab,rpath="image_test/",t_num=8200000)
    print('get image: ', val_feeder.size)

    num_train_samples = train_feeder.size  #
    num_batches_per_epoch = int(num_train_samples / FLAGS.batch_size)  #80

    num_val_samples = val_feeder.size
    num_batches_per_epoch_val = int(num_val_samples / FLAGS.batch_size)  #20
    shuffle_idx_val = np.random.permutation(num_val_samples)

    with tf.device('/gpu:0'):
        config = tf.ConfigProto(inter_op_parallelism_threads = 1,intra_op_parallelism_threads = 4,allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
            train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
            if FLAGS.restore:
                ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
                if ckpt:
                    # the global_step will restore sa well
                    saver.restore(sess, ckpt)
                    print('restore from the checkpoint{0}'.format(ckpt))

            print('=============================begin training=============================')
            for cur_epoch in range(FLAGS.num_epochs):
                epoch_time=time.time()
                print("epoch:" + str(cur_epoch))
                train_cost = 0
                start_time = time.time()
                batch_time = time.time()
                acc_num=0
                step_num=0


                # the tracing part
                for cur_batch in range(num_batches_per_epoch):#
                    step_num+=1
                    if (cur_batch + 1) % 100 == 0:
                        print('batch', cur_batch, ': time', time.time() - batch_time)
                    batch_time = time.time()
                    indexs = [i for i in
                              range(cur_batch * FLAGS.batch_size, (cur_batch + 1) * FLAGS.batch_size)]

                    batch_inputs, batch_seq_len,batch_labels = \
                        train_feeder.input_index_generate_batch(indexs)
                    print(batch_labels)
                    feed = {model.inputs: batch_inputs,
                            model.labels: batch_labels,
                            model.seq_len: batch_seq_len}

                    train_acc,summary_str, batch_cost, step, _ = \
                        sess.run([model.accuracy,model.merged_summay, model.cost, model.global_step,
                                  model.train_op], feed)

                    acc_num+=train_acc*FLAGS.batch_size

                    # calculate the cost
                    train_cost += batch_cost * FLAGS.batch_size

                    train_writer.add_summary(summary_str, step)

                    # save the checkpoint
                    if step % FLAGS.save_steps == 1:
                        if not os.path.isdir(FLAGS.checkpoint_dir):
                            os.mkdir(FLAGS.checkpoint_dir)
                        logger.info('save the checkpoint of{0}', format(step))
                        saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'ocr-model'),
                                   global_step=step)

                    if step % FLAGS.validation_steps == 0:
                        val_acc_num=0
                        step_val=0
                        lr = 0
                        for j in range(num_batches_per_epoch_val):
                            step_val+=1
                            indexs_val = [i % num_val_samples for i in
                                          range(j * FLAGS.batch_size, (j + 1) * FLAGS.batch_size)]
                            #print(indexs_val)

                            val_inputs, val_seq_len, val_labels = \
                                val_feeder.input_index_generate_batch(indexs_val)
                            val_feed = {model.inputs: val_inputs,
                                        model.labels: val_labels,
                                        model.seq_len: val_seq_len}

                            val_acc,loss,lr= \
                                sess.run([model.accuracy,model.cost,model.lrn_rate],
                                         val_feed)

                            val_acc_num+=FLAGS.batch_size*val_acc

                        avg_train_cost = train_cost / ((cur_batch + 1) * FLAGS.batch_size)

                        accuracy=val_acc_num/(FLAGS.batch_size*step_val)

                        now = datetime.datetime.now()
                        log = "{}/{} {}:{}:{} Epoch {}/{}, " \
                              "accuracy = {:.3f},avg_train_cost = {:.3f}, " \
                              " time = {:.3f},lr={:.8f}"
                        pri=log.format(now.month, now.day, now.hour, now.minute, now.second,
                                         cur_epoch + 1, FLAGS.num_epochs, accuracy, avg_train_cost,
                                          time.time() - start_time, lr)
                        print(pri)
                        with open("result.txt", 'a+', encoding='utf-8') as f:
                            f.write(pri+"\n")
                train_acc=acc_num/(step_num*FLAGS.batch_size)
                tri=str(cur_epoch)+"  used  "+str(time.time()-epoch_time)+"  train acc="+str(train_acc)
                print(tri)
                with open("result.txt", 'a+', encoding='utf-8') as f:
                    f.write(tri + "\n")


def main(_):
    if FLAGS.num_gpus == 0:
        dev = '/cpu:0'
    elif FLAGS.num_gpus == 1:
        dev = '/gpu:0'
    else:
        raise ValueError('Only support 0 or 1 gpu.')

    with tf.device(dev):
        if FLAGS.mode == 'train':
            train(FLAGS.train_dir, FLAGS.val_dir, FLAGS.mode)









if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()





