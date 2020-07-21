import tensorflow as tf

class MyMetricCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        tf.summary.scalar(_______, logs['categorical_crossentropy'], epoch) #<TODO>hyperparameterMetricTagを設定する:
