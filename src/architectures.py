import tensorflow as tf


class NameSpacer:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class QNetArchitecture:
    def __init__(self, obs_shape, n_outputs, learning_rate, name='architecture', reset=True):
        self.obs_shape = obs_shape
        self.n_outputs = n_outputs
        self.learning_rate = learning_rate
        
        # Common initializations
        self.name = name
        if reset:
            tf.reset_default_graph()
        with tf.variable_scope(self.name) as scope:
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.define_computation_graph()

        # Aliases
        self.ph = self.placeholders
        self.op = self.optimizers
        self.summ = self.summaries
        
        # Add ins
        self.variables = {var.name[len(self.name):]:var for var in 
                          tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)}

    def define_computation_graph(self):
        # Reset graph
        self.placeholders = NameSpacer(**self.define_placeholders())
        self.core_model = NameSpacer(**self.define_core_model())
        self.losses = NameSpacer(**self.define_losses())
        self.optimizers = NameSpacer(**self.define_optimizers())
        self.summaries = NameSpacer(**self.define_summaries())

    def define_placeholders(self):
        with tf.variable_scope('Placeholders'):
            X = tf.placeholder(tf.float32, shape=self.obs_shape)
            X_action = tf.placeholder(tf.int32, shape=(None,))
            y = tf.placeholder(tf.float32, shape=(None,1))
        return ({"X": X, "X_action": X_action, "y": y})

    def define_core_model(self):
        with tf.variable_scope('Core_Model'):
            h = tf.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(2,2), padding="SAME",
                                 activation=tf.nn.relu)(self.placeholders.X)
            h = tf.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), padding="SAME",
                                 activation=tf.nn.relu)(h)
            h = tf.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(2,2), padding="SAME",
                                 activation=tf.nn.relu)(h)
            h = tf.layers.Flatten()(h)
            h = tf.layers.Dense(units=128, activation=tf.nn.relu)(h)
            h = tf.layers.Dense(units=64, activation=tf.nn.relu)(h)
            output = tf.layers.Dense(units=self.n_outputs, activation=None)(h)
        return ({"output": output})

    def define_losses(self):
        with tf.variable_scope('Losses'):
            Q_action = tf.reduce_sum(self.core_model.output * \
                                     tf.one_hot(self.placeholders.X_action, self.n_outputs), 
                                     axis=-1, keepdims=True)
            loss=tf.reduce_mean(tf.square(self.placeholders.y-Q_action))
        return ({"loss": loss})

    def define_optimizers(self):
        with tf.variable_scope('Optimization'):
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            training_op=optimizer.minimize(self.losses.loss)
        return ({"step": training_op})

    def define_summaries(self):
        with tf.variable_scope('Summaries'):
            scalar_probes = {"loss": tf.squeeze(self.losses.loss)}
            performance_scalar = [tf.summary.scalar(k, tf.reduce_mean(v), family=self.name)
                                        for k, v in scalar_probes.items()]
            performance_scalar = tf.summary.merge(performance_scalar)
        return ({'scalars': performance_scalar})