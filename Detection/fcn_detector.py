
import tensorflow as tf
import sys
sys.path.append("../")


class FcnDetector(object):
    #net_factory: which net
    #model_path: where the params'file is
    def __init__(self, net_factory, model_path):
        #create a graph
        graph = tf.Graph()
        with graph.as_default():
            #define tensor and op in graph(-1,1)
            self.image_op = tf.compat.v1.placeholder(tf.float32, name='input_image')
            self.width_op = tf.compat.v1.placeholder(tf.int32, name='image_width')
            self.height_op = tf.compat.v1.placeholder(tf.int32, name='image_height')
            image_reshape = tf.reshape(self.image_op, [1, self.height_op, self.width_op, 3])
            self.cls_prob, self.bbox_pred, _ = net_factory(image_reshape, training=False)
            
            #allow 
            self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True, gpu_options=tf.compat.v1.GPUOptions(allow_growth=True)))
            saver = tf.compat.v1.train.Saver()
            #check whether the dictionary is valid
            model_dict = '/'.join(model_path.split('/')[:-1])
            ckpt = tf.train.get_checkpoint_state(model_dict)
            print(model_path)
            readstate = ckpt and ckpt.model_checkpoint_path
            assert  readstate, "the params dictionary is not valid"
            print("restore models' param")
            saver.restore(self.sess, model_path)

    def predict(self, databatch):
        height, width, _ = databatch.shape
        # print(height, width)
        cls_prob, bbox_pred = self.sess.run([self.cls_prob, self.bbox_pred],
                                                           feed_dict={self.image_op: databatch, self.width_op: width,
                                                                      self.height_op: height})
        return cls_prob, bbox_pred