import argparse
import io
import os.path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm

from tf_utils import AttrDict, attrdict_from_yaml, lazy_property_with_scope, share_variables

tfd = tfp.distributions
tfl = tf.layers


class Model(object):
    def __init__(self, data, config):
        # Initialize attributes
        self.data = data
        self.data_shape = list(self.data.shape[1:])
        self.config = config

        # Build model
        self.prior
        self.posterior
        self.code
        self.likelihood
        self.sample
        self.samples
        self.log_prob
        self.divergence
        self.elbo
        self.loss
        self.optimiser
        self.gradients
        self.optimise

        # Define summaries
        self.summary
        self.images

    @lazy_property_with_scope
    def prior(self):
        """Standard normal distribution prior."""
        return tfd.MultivariateNormalDiag(
            loc=tf.zeros(self.config.code_size),
            scale_diag=tf.ones(self.config.code_size))

    @lazy_property_with_scope(scope_name="encoder")
    def posterior(self):
        """a.k.a the encoder"""
        x = tfl.Flatten()(self.data)
        x = tfl.Dense(self.config.hidden_size, activation='relu')(x)
        x = tfl.Dense(self.config.hidden_size, activation='relu')(x)
        loc = tfl.Dense(self.config.code_size)(x)
        scale = tfl.Dense(self.config.code_size, activation='softplus')(x)
        return tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale)

    @lazy_property_with_scope
    def code(self):
        """Code sample from the posterior."""
        return self.posterior.sample()

    @lazy_property_with_scope(scope_name="decoder")
    def likelihood(self):
        """a.k.a the decoder."""
        return self._make_decoder(self.code)

    @lazy_property_with_scope
    def sample(self):
        """Sample example."""
        return self._make_decoder(self.prior.sample(1))

    @lazy_property_with_scope
    def samples(self):
        """Generated examples."""
        return self._make_decoder(self.prior.sample(self.config.n_samples)).mean()

    @lazy_property_with_scope
    def log_prob(self):
        """Log. likelihood of data under code sampled from posterior."""
        return self.likelihood.log_prob(self.data)

    @lazy_property_with_scope
    def divergence(self):
        """KL divergence between posterior and prior."""
        return tfd.kl_divergence(self.posterior, self.prior)

    @lazy_property_with_scope
    def elbo(self):
        """Evidence lower bound with a Lagrangian multiplier beta."""
        return self.log_prob - self.config.beta * self.divergence

    @lazy_property_with_scope
    def loss(self):
        """Negative ELBO reduced over the whole batch and every pixel."""
        return -tf.reduce_mean(self.elbo)

    @lazy_property_with_scope
    def optimiser(self):
        """ADAM optimiser."""
        return tf.train.AdamOptimizer(self.config.learning_rate)

    @lazy_property_with_scope
    def gradients(self):
        """Variables values and gradients of the loss (negative ELBO)."""
        return self.optimiser.compute_gradients(self.loss)

    @lazy_property_with_scope
    def optimise(self):
        """Optimise the loss op. (apply gradients)."""
        return self.optimiser.apply_gradients(self.gradients)

    @lazy_property_with_scope
    def summary(self):
        """Merged the model's summaries."""
        return tf.summary.merge(self._define_summaries())

    @lazy_property_with_scope
    def images(self):
        """Image summary of generated examples."""
        images = tf.reshape(self.samples, (-1, self.samples.shape[2])) # Create col. of images
        images = tf.expand_dims(images, axis=0)   # Add batch dim.
        images = tf.expand_dims(images, axis=-1)  # Add channel dim.
        return tf.summary.image("samples", images, max_outputs=1)

    @share_variables
    def _make_decoder(self, code):
        """Build decoder network."""
        x = tfl.Dense(self.config.hidden_size, activation='relu')(code)
        x = tfl.Dense(self.config.hidden_size, activation='relu')(x)
        logits = tfl.Dense(np.product(self.data_shape))(x)
        logits = tf.reshape(logits, [-1] + self.data_shape)
        return tfd.Independent(tfd.Bernoulli(logits), 2)

    def _define_summaries(self):
        """Define the model's summaries."""
        summaries = []

        # Learning rate
        summaries.append(tf.summary.scalar("learning_rate",
                                           self.optimiser._lr))

        # ELBO and loss
        summaries.append(tf.summary.histogram("evidence/lower_bound_log_prob/image",
                                              self.elbo))
        summaries.append(tf.summary.scalar("mean/evidence/lower_bound_log_prob/image",
                                           tf.reduce_mean(self.elbo)))
        summaries.append(tf.summary.scalar("loss",
                                           self.loss))

        # KL divergence
        summaries.append(tf.summary.histogram("divergence",
                                              self.divergence))
        summaries.append(tf.summary.scalar("mean/divergence",
                                           tf.reduce_mean(self.divergence)))

        # Gradients and variables norm
        gradients, variables = list(zip(*self.gradients))
        for gradient, variable in zip(gradients, variables):
            summaries.append(tf.summary.histogram("gradients/batch_norm/" + variable.name,
                                                  tf.norm(gradient, axis=0)))
            summaries.append(tf.summary.histogram("variables/batch_norm/" + variable.name,
                                                  tf.norm(variable, axis=0)))
        summaries.append(tf.summary.scalar("gradients/global_norm",
                                           tf.global_norm(gradients)))
        summaries.append(tf.summary.scalar("variables/global_norm",
                                           tf.global_norm(variables)))

        # Prior and posterior entropy
        summaries.append(tf.summary.histogram("prior/entropy",
                                              self.prior.entropy()))
        summaries.append(tf.summary.scalar("mean/prior/entropy",
                                           tf.reduce_mean(self.prior.entropy())))
        summaries.append(tf.summary.histogram("posterior/entropy",
                                              self.posterior.entropy()))
        summaries.append(tf.summary.scalar("mean/posterior/entropy",
                                           tf.reduce_mean(self.posterior.entropy())))

        # Prior and posterior log_prob
        summaries.append(tf.summary.histogram("prior/log_prob/image",
                                              self.sample.log_prob(self.data)))
        summaries.append(tf.summary.scalar("mean/prior/log_prob/image",
                                           tf.reduce_mean(self.sample.log_prob(self.data))))
        summaries.append(tf.summary.histogram("posterior/log_prob/image",
                                              self.log_prob))
        summaries.append(tf.summary.scalar("mean/posterior/log_prob/image",
                                           tf.reduce_mean(self.log_prob)))

        return summaries


def plot_codes(codes, labels):
    # Scatter plot
    fig, ax = plt.subplots()
    ax.scatter(codes[:, 0], codes[:, 1], s=2, c=labels, alpha=0.1)
    ax.set_aspect('equal')
    ax.set_xlim(codes.min() - .1, codes.max() + .1)
    ax.set_ylim(codes.min() - .1, codes.max() + .1)
    ax.tick_params(
        axis='both', which='both', left=False, bottom=False,
        labelleft=False, labelbottom=False)

    # Save to io buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)

    # Create image summary
    image = tf.Summary.Image(encoded_image_string=buf.getvalue())
    summary = tf.Summary(value=[tf.Summary.Value(tag="images/codes/image", image=image)])
    return summary


def create_datasets(train_set, test_set, config):
    train_dataset = tf.data.Dataset.from_tensor_slices(
        tf.convert_to_tensor(train_set, dtype=tf.float32)) \
        .map(lambda x: x / 255)                            \
        .shuffle(train_set.shape[0])                       \
        .batch(config.batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices(
        tf.convert_to_tensor(test_set, dtype=tf.float32)) \
        .map(lambda x: x / 255)                           \
        .batch(test_set.shape[0])

    iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                               train_dataset.output_shapes)

    next_batch = iterator.get_next()
    train_init_op = iterator.make_initializer(train_dataset)
    test_init_op = iterator.make_initializer(test_dataset)

    return next_batch, train_init_op, test_init_op


def train(model, train_init_op, test_init_op, test_labels, config):
    with tf.train.MonitoredSession() as sess:
        summary_writer_train = tf.summary.FileWriter(
            os.path.join(config.logs_dir, "train"), sess.graph)
        summary_writer_test = tf.summary.FileWriter(
            os.path.join(config.logs_dir, "test"))

        step = 0
        for epoch in tqdm(range(config.epochs)):
            # Test
            sess.run(test_init_op)
            test_summary, test_images, test_codes = sess.run(
                [model.summary, model.images, model.code])
            summary_writer_test.add_summary(test_summary, step)
            summary_writer_test.add_summary(test_images, step)

            # Plot codes
            # TODO: Use TensorBoard projector.
            codes = plot_codes(test_codes, test_labels)
            summary_writer_test.add_summary(codes, step)

            # Train
            # TODO: Add tfu.loop that will run whole epoch, have callbacks and reduce returns.
            sess.run(train_init_op)
            while True:
                try:
                    fetches = AttrDict({"optimise": model.optimise})
                    if step % config.log_every == 0:
                        fetches.summary = model.summary

                    returns = sess.run(fetches)
                    if "summary" in returns:
                        summary_writer_train.add_summary(returns.summary, step)

                    step += 1
                except tf.errors.OutOfRangeError:
                    break

    summary_writer_train.close()
    summary_writer_test.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VAE for MNIST dataset.")
    parser.add_argument('--config', type=str, default="", help="YAML formatted configuration")
    user_config_json = parser.parse_args().config

    default_config = AttrDict({
        "batch_size": 100,
        "epochs": 20,
        "n_samples": 10,
        "hidden_size": 200,
        "code_size": 2,
        "beta": 1.,
        "learning_rate": 0.001,
        "logs_dir": "./logs",
        "log_every": 100
    })
    config = default_config.nested_update(attrdict_from_yaml(user_config_json))

    (train_set, _), (test_set, test_labels) = tf.keras.datasets.mnist.load_data()
    # TODO: Use whole test set, but batch it like train set and average summaries.
    #       https://stackoverflow.com/questions/40788785/how-to-average-summaries-over-multiple-batches
    train_set, test_set, test_labels = train_set[:], test_set[:5000], test_labels[:5000]

    next_batch, train_init_op, test_init_op = create_datasets(train_set, test_set, config)

    model = Model(next_batch, config)
    train(model, train_init_op, test_init_op, test_labels, config)
