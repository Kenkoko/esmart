
import tensorflow as tf
from keras.engine import data_adapter

# https://stackoverflow.com/questions/66472201/gradient-accumulation-with-custom-model-fit-in-tf-keras
# https://arxiv.org/abs/2102.06171
@tf.keras.utils.register_keras_serializable() 
class EsmartModel(tf.keras.Model):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.gradient_accumulation = [tf.Variable(tf.zeros_like(v, dtype=tf.float32), trainable=False, name="accum_" + str(i)) for i, v in
                                      enumerate(self.trainable_variables)]
    # Accumulate gradients
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.

        # TODO: check if this is correct
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        batch_size = x.shape[0]

        
        max_subbatch_size = int(batch_size / 2)

        for subbatch_start in range(0, batch_size, max_subbatch_size):
            # determine data used for this subbatch
            subbatch_end = min(subbatch_start + max_subbatch_size, batch_size)
            subbatch_slice = slice(subbatch_start, subbatch_end)
            x_subbatch = x[subbatch_slice]
            y_subbatch = y[subbatch_slice]

            with tf.GradientTape() as tape:
                y_pred_subbatch = self(x_subbatch, training=True)
                loss = self.compiled_loss(y_subbatch, y_pred_subbatch, regularization_losses=self.losses) 
                loss = loss / tf.cast(len(range(0, batch_size, max_subbatch_size)), tf.float32)

            gradients = tape.gradient(loss, self.trainable_variables)
            for i in range(len(self.gradient_accumulation)):
                self.gradient_accumulation[i].assign_add(gradients[i])
            self.compiled_metrics.update_state(y_subbatch, y_pred_subbatch)

        # The training happens here.
        self.apply_accu_gradients()
        return {m.name: m.result() for m in self.metrics}


    def apply_accu_gradients(self):
        # apply accumulated gradients
        self.optimizer.apply_gradients(zip(self.gradient_accumulation, self.trainable_variables))

        # reset accumulated gradients
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign(tf.zeros_like(self.trainable_variables[i], dtype=tf.float32))