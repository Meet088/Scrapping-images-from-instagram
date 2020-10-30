#!/usr/bin/python
import tensorflow as tf

from config import Config
from model import CaptionGenerator
from dataset import prepare_train_data, prepare_eval_data, prepare_test_data, prepare_eval_new_data

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string('phase', 'train',
                       'The phase can be train, eval or test')

tf.flags.DEFINE_boolean('load', False,
                        'Turn on to load a pretrained model from either \
                        the latest checkpoint or a specified file')

tf.flags.DEFINE_string('model_file', None,
                       'If sepcified, load a pretrained model from this file')

tf.flags.DEFINE_boolean('load_cnn', False,
                        'Turn on to load a pretrained CNN model')

tf.flags.DEFINE_string('cnn_model_file', './vgg16_no_fc.npy',
                       'The file containing a pretrained CNN model')

tf.flags.DEFINE_boolean('train_cnn', False,
                        'Turn on to train both CNN and RNN. \
                         Otherwise, only RNN is trained')

tf.flags.DEFINE_integer('beam_size', 3,
                        'The size of beam search for caption generation')

def main(argv):
    config = Config()
    config.phase = FLAGS.phase
    config.train_cnn = FLAGS.train_cnn
    config.beam_size = FLAGS.beam_size

    with tf.Session() as sess:
        if FLAGS.phase == 'train':
            # training phase
            data = prepare_train_data(config)
            model = CaptionGenerator(config)
            sess.run(tf.global_variables_initializer())
            if FLAGS.load:
                model.load(sess, FLAGS.model_file)
            if FLAGS.load_cnn:
                model.load_cnn(sess, FLAGS.cnn_model_file)
            tf.get_default_graph().finalize()
            model.train(sess, data)

        elif FLAGS.phase == 'eval':
            # evaluation phase
            coco, data, vocabulary = prepare_eval_data(config)
            model = CaptionGenerator(config)
            model.load(sess, FLAGS.model_file)
            tf.get_default_graph().finalize()
            model.eval(sess, coco, data, vocabulary)

        elif FLAGS.phase == 'test_new_data':
            # evaluation phase
            coco, data, vocabulary = prepare_eval_new_data(config.eval_caption_file_unsplash,config.eval_image_unsplash,config)
            model = CaptionGenerator(config)
            model.load(sess, FLAGS.model_file)
            tf.get_default_graph().finalize()
            model.eval_new_data(sess, coco, data, vocabulary,config.eval_result_dir_unsplash,config.eval_result_file_unsplash)

        elif FLAGS.phase == 'test_new_data_vizwiz':
            # evaluation phase
            coco, data, vocabulary = prepare_eval_new_data(config.eval_caption_file_vizwiz_train,config.eval_image_vizwiz_train,config)
            model = CaptionGenerator(config)
            model.load(sess, FLAGS.model_file)
            tf.get_default_graph().finalize()
            model.eval_new_data(sess, coco, data, vocabulary,config.eval_result_dir_vizwiz_train,config.eval_result_file_vizwiz_train)

        elif FLAGS.phase == 'test_new_data_insta':
            # evaluation phase
            coco, data, vocabulary = prepare_eval_new_data(config.eval_caption_file_insta,config.eval_image_insta,config)
            model = CaptionGenerator(config)
            model.load(sess, FLAGS.model_file)
            tf.get_default_graph().finalize()
            model.eval_new_data(sess, coco, data, vocabulary,config.eval_result_dir_insta,config.eval_result_file_insta)

        elif FLAGS.phase == 'test_new_data_google_top_n':
            # evaluation phase
            coco, data, vocabulary = prepare_eval_new_data(config.eval_caption_file_topN,config.eval_image_topN,config)
            model = CaptionGenerator(config)
            model.load(sess, FLAGS.model_file)
            tf.get_default_graph().finalize()
            model.eval_new_data(sess, coco, data, vocabulary,config.eval_result_dir_topN,config.eval_result_file_topN)


        else:
            # testing phase
            data, vocabulary = prepare_test_data(config)
            model = CaptionGenerator(config)
            model.load(sess, FLAGS.model_file)
            tf.get_default_graph().finalize()
            model.test(sess, data, vocabulary)

if __name__ == '__main__':
    tf.app.run()
