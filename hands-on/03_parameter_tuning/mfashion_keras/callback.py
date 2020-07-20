import tensorflow as tf

class MyMetricCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
         tf.summary.scalar('ccentropy', logs['categorical_crossentropy'], epoch)
