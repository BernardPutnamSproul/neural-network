use rayon::prelude::*;
use std::{iter::zip, time::Instant};

use indicatif::ProgressIterator;
use nalgebra::{DMatrix, DVector};
use ndarray::{Array, Array1, Array2};
use ndarray_rand::{rand_distr::StandardNormal, RandomExt};
use rand::{seq::SliceRandom, Rng};

pub struct Network {
    num_layers: usize,
    _sizes: Vec<usize>,
    biases: Vec<Array1<f64>>,
    weights: Vec<Array2<f64>>,
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
    pub fn new(sizes: &[usize]) -> Self {
        Self {
            num_layers: sizes.len(),
            _sizes: sizes.to_vec(),
            biases: sizes
                .iter()
                .skip(1)
                .map(|y| Array::random(*y, StandardNormal))
                .collect(),

            weights: zip(&sizes[..sizes.len() - 1], &sizes[1..])
                .map(|(x, y)| Array::random((*y, *x), StandardNormal))
                .collect(),
        }
    }

    //"""Return the output of the network if ``a`` is input."""
    pub fn feedforward(&self, mut a: Array1<f64>) -> Array1<f64> {
        for (b, w) in zip(self.biases.as_slice(), self.weights.as_slice()) {
            a = (w.dot(&a) + b).mapv_into(sigmoid);
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
        mut training_data: Vec<(Array1<f64>, Array1<f64>)>,
        epochs: usize,
        mini_batch_size: usize,
        eta: f64,
        test_data: Option<&[(Array1<f64>, usize)]>,
    ) {
        println!(
            "Starting training:\n  epochs: {}\n  mini_batch: {}\n  examples: {}",
            epochs,
            mini_batch_size,
            training_data.len()
        );
        let n = training_data.len();
        let mut rng = rand::rng();
        for j in 0..epochs {
            training_data.shuffle(&mut rng);
            let mini_batches = training_data.chunks(mini_batch_size);

            for mini_batch in mini_batches {
                self.update_mini_batch(mini_batch, eta);
            }
            if test_data.is_some() && j % 2 == 0 {
                println!(
                    "epoch {}: {} / {}",
                    j,
                    self.evaluate(test_data.as_ref().unwrap()),
                    test_data.as_ref().unwrap().len()
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
    pub fn update_mini_batch(&mut self, mini_batch: &[(Array1<f64>, Array1<f64>)], eta: f64) {
        // initialize empty gradients
        let mut nabla_b: Vec<_> = self
            .biases
            .iter()
            .map(Array::shape)
            .map(Array::<f64, _>::zeros)
            .collect();

        let mut nabla_w: Vec<_> = self
            .weights
            .iter()
            .map(Array::shape)
            .map(Array::<f64, _>::zeros)
            .collect();

        for (x, y) in mini_batch {
            // get the partial cost derivative gradients of each of the weights and biases
            let (delta_nabla_b, delta_nabla_w) = self.backprop(x.clone(), y.clone());

            nabla_b.iter_mut().zip(delta_nabla_b).for_each(|(nb, dnb)| {
                *nb = &*nb + dnb;
            });

            nabla_w.iter_mut().zip(delta_nabla_w).for_each(|(nw, dnw)| {
                *nw = &*nw + dnw;
            });

            self.weights
                .iter_mut()
                .zip(nabla_w.iter())
                .for_each(|(w, nw)| {
                    *w = ((&*w - eta / (mini_batch.len() as f64)) * nw)
                        .into_dimensionality()
                        .unwrap();
                });

            self.biases
                .iter_mut()
                .zip(nabla_b.iter())
                .for_each(|(b, nb)| {
                    *b = ((&*b - eta / (mini_batch.len() as f64)) * nb)
                        .into_dimensionality()
                        .unwrap();
                });
        }
    }

    /// Return a tuple ``(nabla_b, nabla_w)`` representing the
    /// gradient for the cost function C_x.  ``nabla_b`` and
    /// ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
    /// to ``self.biases`` and ``self.weights``.
    pub fn backprop(&self, x: Array1<f64>, y: Array1<f64>) -> (Vec<Array1<f64>>, Vec<Array2<f64>>) {
        let mut nabla_b: Vec<_> = self
            .biases
            .iter()
            .map(|val| Array::zeros(val.shape()).into_dimensionality().unwrap())
            .collect();

        let mut nabla_w: Vec<_> = self
            .weights
            .iter()
            .map(|val| Array::zeros(val.shape()).into_dimensionality().unwrap())
            .collect();

        // feedforward
        let mut activation = x.clone();
        let mut activations = vec![x]; // list to store all the activations, layer by layer
        let mut zs = Vec::new(); // list to store all the z vectors, layer by layer

        for (b, w) in zip(self.biases.iter(), self.weights.iter()) {
            let z = w.dot(&activation) + b;
            zs.push(z.clone());
            activation = z.mapv_into(sigmoid);
            activations.push(activation.clone());
        }

        // backward pass
        let mut delta = self.cost_derivative(activations.last().unwrap(), &y)
            * zs.last().unwrap().clone().mapv_into(sigmoid_prime);

        *nabla_b.last_mut().unwrap() = delta.clone();
        *nabla_w.last_mut().unwrap() = Array::from_shape_fn(
            (delta.len(), activations[activations.len() - 2].len()),
            |(i, j)| delta[i] * activations[activations.len() - 2][j],
        );

        // Note that the variable l in the loop below is used a little
        // differently to the notation in Chapter 2 of the book.  Here,
        // l = 1 means the last layer of neurons, l = 2 is the
        // second-last layer, and so on.  It's a renumbering of the
        // scheme in the book, used here to take advantage of the fact
        // that Python can use negative indices in lists.
        for l in 2..self.num_layers {
            let z = zs[zs.len() - l].clone();
            let sp = z.mapv_into(sigmoid_prime);
            delta = self.weights[self.weights.len() - l + 1].t().dot(&delta) * sp;
            let b_len = nabla_b.len();
            let w_len = nabla_w.len();

            nabla_b[b_len - l] = delta.clone();
            nabla_w[w_len - l] = Array::from_shape_fn(
                (delta.len(), activations[activations.len() - l - 1].len()),
                |(i, j)| delta[i] * activations[activations.len() - l - 1][j],
            );
        }
        (nabla_b, nabla_w)
    }

    /// Return the vector of partial derivatives \partial C_x /
    /// \partial a for the output activations.
    pub fn cost_derivative(
        &self,
        output_activations: &Array1<f64>,
        y: &Array1<f64>,
    ) -> Array1<f64> {
        output_activations - y
    }

    /// Return the number of test inputs for which the neural
    /// network outputs the correct result. Note that the neural
    /// network's output is assumed to be the index of whichever
    /// neuron in the final layer has the highest activation.
    pub fn evaluate(&self, test_data: &[(Array1<f64>, usize)]) -> usize {
        test_data
            .iter()
            .map(|(x, y)| {
                (
                    self.feedforward(x.clone())
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.total_cmp(b))
                        .unwrap()
                        .0,
                    y,
                )
            })
            .filter(|(x, y)| x == *y)
            .count()
    }
}

pub struct Network2 {
    num_layers: usize,
    _sizes: Vec<usize>,
    biases: Vec<DVector<f64>>,
    weights: Vec<DMatrix<f64>>,
    weight_grad: Vec<DMatrix<f64>>,
    bias_grad: Vec<DVector<f64>>,
    output_activation: bool,
}

impl Network2 {
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
    pub fn new(sizes: &[usize]) -> Self {
        let mut rng = rand::rng();
        Self {
            num_layers: sizes.len(),
            _sizes: sizes.to_vec(),
            biases: sizes
                .iter()
                .skip(1)
                .map(|y| {
                    DVector::from_fn(*y, |_, _| {
                        // rng.sample::<f64, rand_distr::StandardUniform>(rand_distr::StandardUniform)
                        //     * 2.
                        //     - 1.

                        rng.sample::<f64, rand_distr::StandardNormal>(rand_distr::StandardNormal)
                    })
                })
                .collect(),

            weights: zip(&sizes[..sizes.len() - 1], &sizes[1..])
                .map(|(x, y)| {
                    DMatrix::from_fn(*y, *x, |_, _| {
                        // rng.sample::<f64, rand_distr::StandardUniform>(rand_distr::StandardUniform)
                        //     * 2.
                        //     - 1.

                        rng.sample::<f64, rand_distr::StandardNormal>(rand_distr::StandardNormal)
                    })
                })
                .collect(),
            bias_grad: sizes.iter().skip(1).map(|y| DVector::zeros(*y)).collect(),

            weight_grad: zip(&sizes[..sizes.len() - 1], &sizes[1..])
                .map(|(x, y)| DMatrix::zeros(*y, *x))
                .collect(),
            output_activation: true,
        }
    }

    //"""Return the output of the network if ``a`` is input."""
    pub fn feedforward(&self, mut a: DVector<f64>) -> DVector<f64> {
        for (i, (b, w)) in zip(self.biases.as_slice(), self.weights.as_slice()).enumerate() {
            a = w * a;
            a += b;
            // a.apply(|v| *v = relu(*v));

            if i == self.biases.len() {
                a.apply(O);
            } else {
                a.apply(A);
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
        mut training_data: Vec<(DVector<f64>, DVector<f64>)>,
        epochs: usize,
        mini_batch_size: usize,
        eta: f64,
        test_data: Option<&[(DVector<f64>, DVector<f64>)]>,
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
                    ((correct as f64) / (len as f64)) * 100.0,
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
    pub fn update_mini_batch(&mut self, mini_batch: &[(DVector<f64>, DVector<f64>)], eta: f64) {
        // initialize empty gradients
        // the small coefficient is an experiment based on the fact that the gradients are probably similar enough
        // to the previous batch's that leaving a small remnant from the earlier batch will help.
        self.bias_grad.iter_mut().for_each(|layer| *layer *= 0.1);
        self.weight_grad.iter_mut().for_each(|layer| *layer *= 0.1);

        // update the gradients for every example in the batch
        mini_batch
            .iter()
            .for_each(|(x, y)| self.backprop(x.clone(), y.clone()));

        let lr_bs = eta / mini_batch.len() as f64;

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
    pub fn backprop(&mut self, x: DVector<f64>, y: DVector<f64>) {
        // let mut nabla_b: Vec<_> = self
        //     .biases
        //     .iter()
        //     .map(DVector::len)
        //     .map(DVector::<f64>::zeros)
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
                z.apply(O);
            } else {
                z.apply(A)
            }
            activations.push(z);
        }

        // backward pass
        // let mul = if self.output_activation {
        //     zs.last().unwrap().clone().apply_into(O_PRIME)
        // } else {
        //     zs.last().unwrap().clone()
        // };

        let mut delta = self
            .cost_derivative(&activations.pop().unwrap(), &y)
            .component_mul(&zs.pop().unwrap().apply_into(O_PRIME));

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
        let a_len = activations.len();
        let zs_len = zs.len();

        for l in 2..self.num_layers {
            // let z = zs[zs_len - l].clone();
            let z = zs.pop().unwrap();
            // let sp = z.apply_into(|z| *z = relu_prime(*z));
            let sp = z.apply_into(A_PRIME);
            delta = (&self.weights[w_len - l + 1].tr_mul(&delta)).component_mul(&sp);

            self.bias_grad[b_len - l] += &delta;
            self.weight_grad[w_len - l] += &delta * &activations.pop().unwrap().transpose();
        }
        // (nabla_b, nabla_w)
    }

    /// Return the vector of partial derivatives \partial C_x /
    /// \partial a for the output activations.
    pub fn cost_derivative(
        &self,
        output_activations: &DVector<f64>,
        y: &DVector<f64>,
    ) -> DVector<f64> {
        output_activations - y
    }

    /// Return the number of test inputs for which the neural
    /// network outputs the correct result. Note that the neural
    /// network's output is assumed to be the index of whichever
    /// neuron in the final layer has the highest activation.
    pub fn evaluate(&self, test_data: &[(DVector<f64>, DVector<f64>)]) -> (usize, f64) {
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
            .sum::<f64>()
            / (len as f64);

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

pub const A: fn(&mut f64) = |x: &mut f64| {
    *x = sigmoid(*x);
    // *x = tanh(*x);
    // *x = relu(*x);
};
pub const A_PRIME: fn(&mut f64) = |x: &mut f64| {
    *x = sigmoid_prime(*x);
    // *x = tanh_prime(*x);
    // *x = relu_prime(*x);
};

pub const O: fn(&mut f64) = |x: &mut f64| {
    *x = sigmoid(*x);
};

pub const O_PRIME: fn(&mut f64) = |x: &mut f64| {
    *x = sigmoid_prime(*x);
};

pub fn relu(x: f64) -> f64 {
    f64::max(0.0, x)
}

pub fn relu_prime(x: f64) -> f64 {
    if x >= 0.0 {
        1.0
    } else {
        0.0
    }
}

pub fn tanh(x: f64) -> f64 {
    f64::tanh(x)
}

pub fn tanh_prime(x: f64) -> f64 {
    1.0 - f64::tanh(x).powi(2)
}

/// The sigmoid function.
pub fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

/// Derivative of the sigmoid function.
pub fn sigmoid_prime(z: f64) -> f64 {
    sigmoid(z) * (1.0 - sigmoid(z))
}
