import os
import numpy as np
import tensorflow as tf
import cv2
from sklearn.decomposition import PCA
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
tf.device('/gpu:0')

# +-* + () + 10 digit + blank + space


maxPrintLen = 100
#
tf.app.flags.DEFINE_boolean('restore', True, 'whether to restore from the latest checkpoint')
tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoint/', 'the checkpoint dir')
tf.app.flags.DEFINE_float('initial_learning_rate', 1e-3, 'inital lr')

tf.app.flags.DEFINE_integer('image_height',16, 'image height')
tf.app.flags.DEFINE_integer('image_width',16, 'image width')
tf.app.flags.DEFINE_integer('image_channel', 1, 'image channels as input')

tf.app.flags.DEFINE_integer('max_stepsize', 256, 'max stepsize in lstm, as well as '
                                                'the output channels of last layer in CNN')
tf.app.flags.DEFINE_integer('num_hidden', 128, 'number of hidden units in lstm')
tf.app.flags.DEFINE_integer('num_epochs', 200, 'maximum epochs')
tf.app.flags.DEFINE_integer('batch_size', 512, 'the batch_size')
tf.app.flags.DEFINE_integer('save_steps', 747, 'the step to save checkpoint')
tf.app.flags.DEFINE_integer('validation_steps', 747, 'the step to validation')

tf.app.flags.DEFINE_float('decay_rate', 0.90, 'the lr decay rate')
tf.app.flags.DEFINE_float('beta1', 0.9, 'parameter of adam optimizer beta1')
tf.app.flags.DEFINE_float('beta2', 0.999, 'adam parameter beta2')

tf.app.flags.DEFINE_integer('decay_steps', 747*2, 'the lr decay_step for optimizer')
tf.app.flags.DEFINE_float('momentum', 0.9, 'the momentum')

tf.app.flags.DEFINE_string('train_dir', 'image/', 'the train data dir')
tf.app.flags.DEFINE_string('val_dir', 'image_test/', 'the val data dir')
tf.app.flags.DEFINE_string('infer_dir', './imgs/infer/', 'the infer data dir')
tf.app.flags.DEFINE_string('val_lab', 'stat/label_val.txt', 'the label of test')
tf.app.flags.DEFINE_string('tra_lab', 'stat/label.txt', 'the infer data dir')
tf.app.flags.DEFINE_string('log_dir', './log', 'the logging dir')
tf.app.flags.DEFINE_string('mode', 'train', 'train, val or infer')
tf.app.flags.DEFINE_integer('num_gpus', 1, 'num of gpus')


FLAGS = tf.app.flags.FLAGS
lstr=""
with open("chinese_common.txt",'r+',encoding='utf-8') as f:
    lines=f.readlines()
    for li in lines:
        lstr=lstr+li.strip("\n")
lstr=lstr.strip("\n")
#lstr=lstr[0:120]
charset=lstr+"123456789AaBbCDdEeFfGgHhIiJjKLMmNnOPQqRrSTtUVWXYyZ,.<>/?[]{}|\\!@#$%^&*()!"

num_classes = len(charset)+1

encode_maps = {}
decode_maps = {}
for i, char in enumerate(charset, 1):
    encode_maps[char] = i
    decode_maps[i] = char

SPACE_INDEX = 0
SPACE_TOKEN = ''
encode_maps[SPACE_TOKEN] = SPACE_INDEX
decode_maps[SPACE_INDEX] = SPACE_TOKEN


class DataIterator:
    def __init__(self, img_dir,lab_dir,rpath,t_num):
        self.image = []
        self.labels = []
        img_list=os.listdir(img_dir)
        for i_turn in range(len(img_list)):
            img_list[i_turn]=str(i_turn)+".jpg"
        lab_list=list(open(lab_dir,encoding="utf-8").readlines())
        t_num=min(t_num,len(img_list))
        for turn,(img,lab) in enumerate(zip(img_list,lab_list)):
            if(turn>=t_num):
                break
            image_name=rpath+img
            print(image_name)
            im = cv2.imread(image_name, 0).astype(np.float32) / 255.
            #im = np.reshape(im, [FLAGS.image_height, FLAGS.image_width, FLAGS.image_channel])
            im = cv2.resize(im, (FLAGS.image_width, FLAGS.image_height), interpolation=cv2.INTER_CUBIC)
            im = np.resize(im, [FLAGS.image_height, FLAGS.image_width])
            im = np.resize(im, [FLAGS.image_height, FLAGS.image_width, FLAGS.image_channel])

            self.image.append(im)
            #print(im)
            lab=lab.strip("\n")
            print(lab)
            #code = [SPACE_INDEX if lab == SPACE_TOKEN else encode_maps[c] for c in list(lab)]
            code=SPACE_INDEX if lab == SPACE_TOKEN else encode_maps[lab]
            tmp=[0 for i in range(num_classes)]
            tmp[code]=1
            self.labels.append(tmp)
            print(tmp)

    @property
    def size(self):
        return len(self.labels)

    def input_index_generate_batch(self, index=None):
        #print(index)
        image=[self.image[i] for i in index]
        labels=[self.labels[i] for i in index]
        #print(labels)
        seq=np.asarray([FLAGS.max_stepsize for _ in range(len(index))], dtype=np.int32)
        return image,seq,labels