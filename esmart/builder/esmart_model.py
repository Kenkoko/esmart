
import tensorflow as tf
from keras.engine import data_adapter

# https://stackoverflow.com/questions/66472201/gradient-accumulation-with-custom-model-fit-in-tf-keras
class EsmartModel(tf.keras.Model):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.gradient_accumulation = [tf.zeros_like(this_var) for this_var in self.trainable_variables]
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        batch_size = x.shape[0]

        # max_subbatch_size = (
        #     self._max_subbatch_size if self._max_subbatch_size > 0 else batch_size
        # )

        # max_subbatch_size = 16
        # self.losses
        # # Run forward pass.
        # y_from_subbatch = []
        # with tf.GradientTape() as tape:
        #     for subbatch_start in range(0, batch_size, max_subbatch_size):
        #     # determine data used for this subbatch
        #         subbatch_end = min(subbatch_start + max_subbatch_size, batch_size)
        #         subbatch_slice = slice(subbatch_start, subbatch_end)
        #         x_subbatch = x[subbatch_slice]


        #         y_pred_subbatch = self(x_subbatch, training=True)
        #         y_from_subbatch.append(y_pred_subbatch)
        #     y_pred = tf.concat(y_from_subbatch, 0)
        #     print('y_pred: {}'.format(y_pred.shape))
        #     # y_pred = self(x, training=True)
        #     loss = self.compute_loss(x, y, y_pred, sample_weight)
        #     # print(loss)
        #     loss = loss / batch_size
        # self._validate_target_and_loss(y_pred, loss )
        # # Run backwards pass.
        # self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        max_subbatch_size = 16
        y_pred = []
        accum_gradient = [tf.zeros_like(this_var) for this_var in self.trainable_variables]
        for subbatch_start in range(0, batch_size, max_subbatch_size):
            # determine data used for this subbatch
            subbatch_end = min(subbatch_start + max_subbatch_size, batch_size)
            subbatch_slice = slice(subbatch_start, subbatch_end)
            x_subbatch = x[subbatch_slice]
            y_subbatch = y[subbatch_slice]
            # sample_weight_subbatch = sample_weight[subbatch_slice]
            with tf.GradientTape() as tape:
                y_pred_subbatch = self(x_subbatch, training=True)
                # loss = self.compute_loss(x_subbatch, y_subbatch, y_pred_subbatch)
                loss = self.compiled_loss(y_subbatch, y_pred_subbatch, regularization_losses=self.losses) / len(range(0, batch_size, max_subbatch_size))
                
            y_pred.append(y_pred_subbatch)
            # losses = self.compute_loss(x, y, tf.concat(y_pred, 0))
            # We sum all losses together. (And calculate their mean value.)
            # You might want to split this if you are interested in the separate losses.
            # self.custom_loss_mean.update_state(losses)
            gradients = tape.gradient(loss, self.trainable_variables)
            accum_gradient = [(acum_grad+grad) for acum_grad, grad in zip(accum_gradient, gradients)]
        # The training happens here.
        # losses = losses / batch_size
        accum_gradient = [this_grad  for this_grad in accum_gradient]
        self.optimizer.apply_gradients(zip(accum_gradient, self.trainable_variables))
        # y_pred = self(x, training=True)
        y_pred = tf.concat(y_pred, 0)
        return self.compute_metrics(x, y, y_pred, sample_weight)

        # y_pred = tf.concat(y_pred, 0)
        # self.compiled_metrics.update_state(y, y_pred)
        # return {m.name: m.result() for m in self.metrics}

    # def apply_accu_gradients(self):
    #     # apply accumulated gradients
    #     self.optimizer.apply_gradients(zip(self.gradient_accumulation, self.trainable_variables))

    #     # reset
    #     # self.n_acum_step.assign(0)
    #     for i in range(len(self.gradient_accumulation)):
    #         self.gradient_accumulation[i] = tf.zeros_like(self.trainable_variables[i], dtype=tf.float32)