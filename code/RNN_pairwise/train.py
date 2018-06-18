#coding=utf-8
import tensorflow as tf
import numpy as np
import os
import time
import datetime
from helper import get_overlap_dict,batch_gen_with_pair_overlap,load,prepare,batch_gen_with_single
from QA_RNN_pairwise import QA_RNN_extend
import evaluation
import config

now = int(time.time()) 
    
timeArray = time.localtime(now)
timeStamp = time.strftime("%Y%m%d%H%M%S", timeArray)
timeDay = time.strftime("%Y%m%d", timeArray)
print(timeStamp)

from functools import wraps
#print( tf.__version__)
def log_time_delta(func):
    @wraps(func)
    def _deco(*args, **kwargs):
        start = time.time()
        ret = func(*args, **kwargs)
        end = time.time()
        delta = end - start
        print("%s runed %.2f seconds"% (func.__name__,delta))
        return ret
    return _deco

FLAGS = config.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print(("{}={}".format(attr.upper(), value)))
log_dir = 'log/'+ timeDay
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
data_file = log_dir + '/test_' + FLAGS.data + timeStamp
precision = data_file + 'precise.txt'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)

@log_time_delta
def predict(sess, cnn, test, alphabet, batch_size, q_len, a_len, step, type):
    scores = []
    d = get_overlap_dict(test,alphabet,q_len,a_len)
    for data in batch_gen_with_single(test,alphabet,batch_size,q_len,a_len,overlap_dict = d): 
        feed_dict = {
                    cnn.question: data[0],
                    cnn.answer: data[1],
                    cnn.answer_negative: data[1],
                    cnn.q_pos_overlap: data[2],
                    cnn.q_neg_overlap: data[2],
                    cnn.a_pos_overlap: data[3],
                    cnn.a_neg_overlap: data[3],
                    cnn.q_position: data[4],
                    cnn.a_pos_position: data[5],
                    cnn.a_neg_position: data[5]
        }

        score = sess.run(cnn.score12, feed_dict)

        scores.extend(score)
    with open(data_file + '_' + type + "_score_%d.txt" % step, 'w') as ff:
        string_tmp = '\n'.join([str(i) for i in scores])
        ff.write(string_tmp)
    return np.array(scores[:len(test)])


@log_time_delta
def test_pair_wise():
    train, test, dev = load(FLAGS.data, filter = FLAGS.clean)
    q_max_sent_length = max(map(lambda x:len(x),train['question'].str.split()))
    a_max_sent_length = max(map(lambda x:len(x),train['answer'].str.split()))
    print('q_question_length:{} a_question_length:{}'.format(q_max_sent_length, a_max_sent_length))
    print('train question unique:{}'.format(len(train['question'].unique())))
    print('train length', len(train))
    print('test length', len(test))
    print('dev length', len(dev))
    alphabet, embeddings = prepare([train, test, dev], dim=FLAGS.embedding_dim, is_embedding_needed=True, fresh=FLAGS.fresh)
    # alphabet = prepare([train, test, dev], dim = FLAGS.embedding_dim, is_embedding_needed=False)
    print('alphabet:', len(alphabet))
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto()
        session_conf.allow_soft_placement = FLAGS.allow_soft_placement
        session_conf.log_device_placement = FLAGS.log_device_placement
        session_conf.gpu_options.allow_growth = True
        sess = tf.Session(config=session_conf)
        with sess.as_default(), open(precision, "w") as log:
            log.write(str(FLAGS.__flags) + '\n')
            folder = 'runs/' + timeDay + '/' + timeStamp + '/'
            out_dir = folder + FLAGS.data
            if not os.path.exists(folder):
                os.makedirs(folder)
            print("start build model")
            cnn = QA_RNN_extend(
                max_input_left=q_max_sent_length,
                max_input_right=a_max_sent_length,
                batch_size=FLAGS.batch_size,
                vocab_size=len(alphabet),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                dropout_keep_prob=FLAGS.dropout_keep_prob,
                embeddings=embeddings,
                l2_reg_lambda = FLAGS.l2_reg_lambda,
                overlap_needed=FLAGS.overlap_needed,
                learning_rate=FLAGS.learning_rate,
                trainable = FLAGS.trainable,
                extend_feature_dim = FLAGS.extend_feature_dim,
                pooling = FLAGS.pooling,
                position_needed = FLAGS.position_needed,
                conv = FLAGS.conv)
            cnn.build_graph()
           
            saver = tf.train.Saver(tf.global_variables(), max_to_keep = 20)
            train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
            test_writer = tf.summary.FileWriter(log_dir + '/test')
            # Initialize all variables
            print("build over")
            sess.run(tf.global_variables_initializer())
            print("variables_initializer")

            map_max = 0.65
            for i in range(FLAGS.num_epochs):
                d = get_overlap_dict(train, alphabet, q_len=q_max_sent_length, a_len=a_max_sent_length)
                datas = batch_gen_with_pair_overlap(train, alphabet, FLAGS.batch_size,
                                                    q_len=q_max_sent_length, a_len=a_max_sent_length, fresh=FLAGS.fresh,
                                                    overlap_dict=d)
                print("load data")
                for data in datas:
                    feed_dict = {
                        cnn.question: data[0],
                        cnn.answer: data[1],
                        cnn.answer_negative: data[2],
                        cnn.q_pos_overlap: data[3],
                        cnn.q_neg_overlap: data[4],
                        cnn.a_pos_overlap: data[5],
                        cnn.a_neg_overlap: data[6],
                        cnn.q_position: data[7],
                        cnn.a_pos_position: data[8],
                        cnn.a_neg_position: data[9]
                    }
                    _, summary, step, loss, accuracy, score12, score13 = sess.run(
                        [cnn.train_op, cnn.merged, cnn.global_step, cnn.loss, cnn.accuracy, cnn.score12, cnn.score13],
                        feed_dict)
                    train_writer.add_summary(summary, i)
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g} ,positive {:g}, negative {:g}".format(time_str, step, loss,
                                                                                                 accuracy,
                                                                                                 np.mean(score12),
                                                                                                 np.mean(score13)))
                    line = "{}: step {}, loss {:g}, acc {:g} ,positive {:g}, negative {:g}".format(time_str, step, loss,
                                                                                                  accuracy,
                                                                                                  np.mean(score12),
                                                                                                  np.mean(score13))
                if i % 1 == 0:
                    predicted_dev = predict(sess,cnn,dev,alphabet,FLAGS.batch_size,q_max_sent_length,a_max_sent_length, step, 'dev')
                    map_mrr_dev = evaluation.evaluationBypandas(dev, predicted_dev)
                    predicted_test = predict(sess,cnn,test,alphabet,FLAGS.batch_size,q_max_sent_length,a_max_sent_length, step, 'test')
                    map_mrr_test = evaluation.evaluationBypandas(test, predicted_test)

                    print("{}:epoch:dev map mrr {}".format(i,map_mrr_dev))
                    print("{}:epoch:test map mrr {}".format(i,map_mrr_test))
                    line = " {}:epoch: map_dev{}-------map_mrr_test{}".format(i, map_mrr_dev[0], map_mrr_test)
                    if map_mrr_dev[0] > map_max:
                        map_max = map_mrr_dev[0]
                        save_path = saver.save(sess, out_dir)
                        print("Model saved in file: ", save_path)
                log.write(line + '\n')
                log.flush()
            print('train over')
            saver.restore(sess, out_dir)
            predicted = predict(sess,cnn,train,alphabet,FLAGS.batch_size,q_max_sent_length, a_max_sent_length, step,'train')
            train['predicted'] = predicted
            train['predicted'].to_csv('train.QApair.train.score', index=False, sep='\t')
            map_mrr_train = evaluation.evaluationBypandas(train,predicted)

            predicted_dev = predict(sess, cnn, dev, alphabet, FLAGS.batch_size, q_max_sent_length, a_max_sent_length, step, 'dev')
            dev['predicted'] = predicted_dev
            dev['predicted'].to_csv('train.QApair.dev.score',index = False,sep = '\t')
            map_mrr_dev = evaluation.evaluationBypandas(dev,predicted_dev)

            predicted_test = predict(sess, cnn, test, alphabet, FLAGS.batch_size, q_max_sent_length, a_max_sent_length, step, 'test')

            test['predicted'] = predicted_test
            test['predicted'].to_csv('train.QApair.test.score', index=False, sep='\t')
            map_mrr_test = evaluation.evaluationBypandas(test, predicted_test)
    
            print('map_mrr train',map_mrr_train)
            print('map_mrr dev',map_mrr_dev)
            print('map_mrr test',map_mrr_test)
            log.write(str(map_mrr_train) + '\n')
            log.write(str(map_mrr_test) + '\n')
            log.write(str(map_mrr_dev) + '\n')
    return


if __name__ == '__main__':
    test_pair_wise()

