use indicatif::ProgressIterator;
use nalgebra::{DMatrix, DVector};
use rand::{seq::SliceRandom, Rng};
use rand_distr::Distribution;
use rayon::prelude::*;
use std::{iter::zip, sync::Arc, time::Instant};

use crate::{
    parallel::{Gradient, ThreadGrads},
    Activation,
};

pub struct Network {
    num_layers: usize,
    sizes: Vec<usize>,
    layers: Arc<Gradient<f32>>,
    biases: Vec<DVector<f32>>,
    weights: Vec<DMatrix<f32>>,
    weight_grad: Vec<DMatrix<f32>>,
    bias_grad: Vec<DVector<f32>>,
    activation: Activation,
    output_activation: Activation,
}

impl Network {
    /// The list ``sizes`` contains the number of neurons in the
    /// respective layers of the network.  For example, if the list
    /// was [2, 3, 1] then it would be a three-layer network, with the
    /// first layer containing 2 neurons, the second layer 3 neurons,
    /// and the third layer 1 neuron.  The biases and weights for the
    /// network are initialized randomly, using a Gaussian
    /// distribution with mean 0, and variance 1.  Note that the first
    /// layer is assumed to be an input layer, and by convention we
    /// won't set any biases for those neurons, since biases are only
    /// ever used in computing the outputs from later layers.
    pub fn new<D>(
        sizes: &[usize],
        activation: Activation,
        output_activation: Activation,
        distr: D,
    ) -> Self
    where
        D: Distribution<f32> + Copy,
    {
        let mut rng = rand::rng();
        Self {
            num_layers: sizes.len(),
            sizes: sizes.to_vec(),
            biases: sizes
                .iter()
                .skip(1)
                .map(|y| DVector::from_fn(*y, |_, _| rng.sample::<f32, D>(distr)))
                .collect(),

            weights: std::iter::zip(&sizes[..sizes.len() - 1], &sizes[1..])
                .map(|(x, y)| DMatrix::from_fn(*y, *x, |_, _| rng.sample(distr)))
                .collect(),
            bias_grad: Vec::new(),

            weight_grad: Vec::new(),
            activation,
            output_activation,
            layers: Arc::new(Gradient::new()),
        }
    }

    //"""Return the output of the network if ``a`` is input."""
    pub fn feedforward(&self, mut a: DVector<f32>) -> DVector<f32> {
        for (i, (b, w)) in zip(self.biases.as_slice(), self.weights.as_slice()).enumerate() {
            a = w * a;
            a += b;
            // a.apply(|v| *v = relu(*v));

            if i == self.biases.len() {
                a.apply(self.output_activation.get_fun32());
            } else {
                a.apply(self.activation.get_fun32());
            }
        }

        a
    }

    pub fn feedforward_layers(&self, mut a: DVector<f32>) -> DVector<f32> {
        for (i, (w, b)) in self.layers.iter().enumerate() {
            a = w * a;
            a += b;
            // a.apply(|v| *v = relu(*v));

            if i == self.layers.len() {
                a.apply(self.output_activation.get_fun32());
            } else {
                a.apply(self.activation.get_fun32());
            }
        }

        a
    }
    /// Train the neural network using mini-batch stochastic
    /// gradient descent.  The ``training_data`` is a list of tuples
    /// ``(x, y)`` representing the training inputs and the desired
    /// outputs.  The other non-optional parameters are
    /// self-explanatory.  If ``test_data`` is provided then the
    /// network will be evaluated against the test data after each
    /// epoch, and partial progress printed out.  This is useful for
    /// tracking progress, but slows things down substantially.
    pub fn sdg(
        &mut self,
        mut training_data: Vec<(DVector<f32>, DVector<f32>)>,
        epochs: usize,
        mini_batch_size: usize,
        eta: f32,
        test_data: Option<&[(DVector<f32>, DVector<f32>)]>,
    ) {
        let style = indicatif::ProgressStyle::with_template(
            "[{elapsed:.green}] [{wide_bar:.cyan/red}] {pos:.red}/{len:.green} ({eta})",
        )
        .unwrap()
        .progress_chars("=> ");

        let n = training_data.len();

        println!(
            "Starting training:\n  epochs: {:>3}\n  mini_batch: {}\n  examples: {}",
            epochs, mini_batch_size, n
        );

        let mut rng = rand::rng();

        self.bias_grad = self
            .sizes
            .iter()
            .skip(1)
            .map(|y| DVector::zeros(*y))
            .collect();

        self.weight_grad = zip(&self.sizes[..self.sizes.len() - 1], &self.sizes[1..])
            .map(|(x, y)| DMatrix::zeros(*y, *x))
            .collect();

        for j in 0..epochs {
            training_data.shuffle(&mut rng);
            let start = Instant::now();
            let mini_batches = training_data.chunks(mini_batch_size);

            for mini_batch in mini_batches.into_iter().progress_with_style(style.clone()) {
                self.update_mini_batch(mini_batch, eta);
            }

            if test_data.is_some() {
                let (correct, loss) = self.evaluate(test_data.as_ref().unwrap());
                let len = test_data.as_ref().unwrap().len();

                println!(
                    "epoch {:>3}: {} / {}: {:#.3}%  loss: {:#.10}  time: {:#?}",
                    j,
                    correct,
                    len,
                    ((correct as f32) / (len as f32)) * 100.0,
                    loss,
                    start.elapsed()
                );
            } else {
                println!("epoch {} complete", j);
            }
        }
    }

    pub fn sdg_par(
        &mut self,
        mut training_data: Vec<(DVector<f32>, DVector<f32>)>,
        epochs: usize,
        mini_batch_size: usize,
        eta: f32,
        test_data: Option<&[(DVector<f32>, DVector<f32>)]>,
        threads: usize,
    ) {
        let style = indicatif::ProgressStyle::with_template(
            "[{elapsed:.green}] [{wide_bar:.cyan/red}] {pos:.red}/{len:.green} ({eta})",
        )
        .unwrap()
        .progress_chars("=> ");

        let n = training_data.len();

        println!(
            "Starting training:\n  epochs: {:>3}\n  mini_batch: {}\n  examples: {}",
            epochs, mini_batch_size, n
        );

        self.layers = Arc::new(zip(self.weights.clone(), self.biases.clone()).collect());

        let mut rng = rand::rng();

        let mut thread_grads = ThreadGrads::new(&self.sizes, threads);

        for j in 0..epochs {
            training_data.shuffle(&mut rng);
            let start = Instant::now();
            let mini_batches = training_data.chunks(mini_batch_size);

            for mini_batch in mini_batches.into_iter().progress_with_style(style.clone()) {
                thread_grads.update_mini_batch(
                    self.layers.clone(),
                    self.activation,
                    self.output_activation,
                    Arc::from(mini_batch),
                );

                thread_grads.apply(
                    Arc::get_mut(&mut self.layers).unwrap(),
                    eta / mini_batch_size as f32,
                );
            }

            if test_data.is_some() {
                let (correct, loss) = self.evaluate_layers(test_data.as_ref().unwrap());
                let len = test_data.as_ref().unwrap().len();

                println!(
                    "epoch {:>3}: {} / {}: {:#.3}%  loss: {:#.10}  time: {:#?}",
                    j,
                    correct,
                    len,
                    ((correct as f32) / (len as f32)) * 100.0,
                    loss,
                    start.elapsed()
                );
            } else {
                println!("epoch {} complete", j);
            }
        }
    }

    /// Update the network's weights and biases by applying
    /// gradient descent using backpropagation to a single mini batch.
    /// The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
    /// is the learning rate.
    pub fn update_mini_batch(&mut self, mini_batch: &[(DVector<f32>, DVector<f32>)], eta: f32) {
        // initialize empty gradients
        // the small coefficient is an experiment based on the fact that the gradients are probably similar enough
        // to the previous batch's that leaving a small remnant from the earlier batch will help.
        self.bias_grad.iter_mut().for_each(|layer| *layer *= 0.1);
        self.weight_grad.iter_mut().for_each(|layer| *layer *= 0.1);

        // update the gradients for every example in the batch
        mini_batch
            .iter()
            .for_each(|(x, y)| self.backprop(x.clone(), y));

        let lr_bs = eta / mini_batch.len() as f32;

        self.weights
            .iter_mut()
            .zip(self.weight_grad.iter())
            .for_each(|(w, nw)| {
                *w -= lr_bs * nw;
            });

        self.biases
            .iter_mut()
            .zip(self.bias_grad.iter())
            .for_each(|(b, nb)| {
                *b -= lr_bs * nb;
            });
    }

    /// Return a tuple ``(nabla_b, nabla_w)`` representing the
    /// gradient for the cost function C_x.  ``nabla_b`` and
    /// ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
    /// to ``self.biases`` and ``self.weights``.
    pub fn backprop(&mut self, x: DVector<f32>, y: &DVector<f32>) {
        // let mut nabla_b: Vec<_> = self
        //     .biases
        //     .iter()
        //     .map(DVector::len)
        //     .map(DVector::<f32>::zeros)
        //     .collect();

        // let mut nabla_w: Vec<_> = self
        //     .weights
        //     .iter()
        //     .map(|layer| DMatrix::zeros(layer.nrows(), layer.ncols()))
        //     .collect();

        // feedforward
        // let mut activation = x.clone();
        // let mut activation = x.clone();
        let mut activations = vec![x]; // list to store all the activations, layer by layer
        let mut zs = Vec::new(); // list to store all the z vectors, layer by layer

        for (i, (b, w)) in zip(self.biases.iter(), self.weights.iter()).enumerate() {
            let mut z = w * activations.last().unwrap() + b;
            zs.push(z.clone());
            // activation = z.apply_into(|z| *z = relu(*z));

            // activation = z.apply_into(A);

            if i == self.biases.len() {
                z.apply(self.output_activation.get_fun32());
            } else {
                z.apply(self.activation.get_fun32())
            }
            activations.push(z);
        }

        // backward pass
        // let mul = if self.output_activation {
        //     zs.last().unwrap().clone().apply_into(O_PRIME)
        // } else {
        //     zs.last().unwrap().clone()
        // };

        let mut delta = Self::cost_derivative(&activations.pop().unwrap(), y).component_mul(
            &zs.pop()
                .unwrap()
                .apply_into(self.output_activation.get_dir32()),
        );

        // .component_mul(&zs.last().unwrap().clone().apply_into(|z| *z = relu(*z)));
        // .component_mul(&zs.last().unwrap().clone().apply_into(A_PRIME));

        *self.bias_grad.last_mut().unwrap() += &delta;

        // *self.weight_grad.last_mut().unwrap() += DMatrix::from_fn(
        //     delta.len(),
        //     activations[activations.len() - 2].len(),
        //     |i, j| delta[i] * activations[activations.len() - 2][j],
        // );

        *self.weight_grad.last_mut().unwrap() += &delta * &activations.pop().unwrap().transpose();

        // Note that the variable l in the loop below is used a little
        // differently to the notation in Chapter 2 of the book.  Here,
        // l = 1 means the last layer of neurons, l = 2 is the
        // second-last layer, and so on.  It's a renumbering of the
        // scheme in the book, used here to take advantage of the fact
        // that Python can use negative indices in lists.
        let b_len = self.bias_grad.len();
        let w_len = self.weight_grad.len();

        for l in 2..self.num_layers {
            // let z = zs[zs_len - l].clone();
            let z = zs.pop().unwrap();
            // let sp = z.apply_into(|z| *z = relu_prime(*z));
            let sp = z.apply_into(self.activation.get_dir32());
            delta = self.weights[w_len - l + 1]
                .tr_mul(&delta)
                .component_mul(&sp);

            self.bias_grad[b_len - l] += &delta;
            self.weight_grad[w_len - l] += &delta * &activations.pop().unwrap().transpose();
        }
        // (nabla_b, nabla_w)
    }

    /// Return the vector of partial derivatives \partial C_x /
    /// \partial a for the output activations.
    pub fn cost_derivative(output_activations: &DVector<f32>, y: &DVector<f32>) -> DVector<f32> {
        // dbg!(output_activations.nrows(), output_activations.ncols());
        // dbg!(y.nrows(), y.ncols());
        output_activations - y
    }

    /// Return the number of test inputs for which the neural
    /// network outputs the correct result. Note that the neural
    /// network's output is assumed to be the index of whichever
    /// neuron in the final layer has the highest activation.
    pub fn evaluate(&self, test_data: &[(DVector<f32>, DVector<f32>)]) -> (usize, f32) {
        let len = test_data.len();
        let values = test_data
            .par_iter()
            .map(|(x, y)| (self.feedforward(x.clone()), y));

        let loss = values
            .clone()
            .map(|(x, y)| {
                (x - y)
                    .apply_into(|z| {
                        *z = z.powi(2);
                    })
                    .sum()
            })
            .sum::<f32>()
            / (len as f32);

        let correct = values
            .map(|(x, y)| {
                (
                    x.iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.total_cmp(b))
                        .unwrap()
                        .0,
                    y.iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.total_cmp(b))
                        .unwrap()
                        .0,
                )
            })
            .filter(|(x, y)| *x == *y)
            .count();

        (correct, loss)
    }

    pub fn evaluate_layers(&self, test_data: &[(DVector<f32>, DVector<f32>)]) -> (usize, f32) {
        let len = test_data.len();
        let values = test_data
            .par_iter()
            .map(|(x, y)| (self.feedforward_layers(x.clone()), y));

        let loss = values
            .clone()
            .map(|(x, y)| {
                (x - y)
                    .apply_into(|z| {
                        *z = z.powi(2);
                    })
                    .sum()
            })
            .sum::<f32>()
            / (len as f32);

        let correct = values
            .map(|(x, y)| {
                (
                    x.iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.total_cmp(b))
                        .unwrap()
                        .0,
                    y.iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.total_cmp(b))
                        .unwrap()
                        .0,
                )
            })
            .filter(|(x, y)| *x == *y)
            .count();

        (correct, loss)
    }
}
