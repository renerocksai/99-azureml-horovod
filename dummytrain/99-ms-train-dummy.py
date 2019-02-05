import tensorflow as tf
import traceback
import numpy as np

# HOROVOD
import horovod.tensorflow as hvd
hvd.init()

# Pin GPU to be used to process local rank (one GPU per process)
config = tf.ConfigProto()
config.gpu_options.visible_device_list = str(hvd.local_rank())

def train():
    coord = tf.train.Coordinator()
    global_step = tf.Variable(0, name='global_step', trainable=False)
    with tf.variable_scope('model') as scope:
        learning_rate = tf.convert_to_tensor(0.02)

        w = tf.Variable([[0]], trainable=True, dtype=tf.float32)

        x = np.zeros((1,1), dtype=np.float32)
        y = np.ones((1,1), dtype=np.float32)
        x = tf.convert_to_tensor(x)
        y = tf.convert_to_tensor(y)
        y_pred = tf.matmul(x, w)
        mse = tf.losses.mean_squared_error(y, y_pred)

        optimizer = tf.train.AdamOptimizer(learning_rate)
        optimizer = hvd.DistributedOptimizer(optimizer)
        optimize = optimizer.minimize(mse, var_list=w)

  # Train!
    with tf.Session(config=config) as sess:
        try:
            sess.run(tf.global_variables_initializer())
            bcast = hvd.broadcast_global_variables(0)
            bcast.run()
            sess.run([optimize])
        except Exception as e:
            print('Exiting due to exception: %s' % e)
            traceback.print_exc()
            coord.request_stop(e)


def main():
    train()


if __name__ == '__main__':
    main()
