use std::array;

use nalgebra::{DMatrix, DVector};

use neural_network::{mnist, nn_f32};
use rand::Rng;

fn main() {
    let data = mnist::read("data/").unwrap();
    // for i in 0..8 {
    //     print!(
    //         "[{}]:\n{}",
    //         i,
    //         mnist::render(data.test_images[i].as_slice().unwrap(), 0.95),
    //     );
    // }
    let mut network = nn_f32::Network2::new(&[28 * 28, 16, 16, 16, 10]);

    let training_data: Vec<_> = data
        .training_images
        .clone()
        .into_iter()
        .zip(data.training_labels.clone().as_slice())
        .map(|(v1, v2)| {
            (
                DVector::from_column_slice(v1.as_slice().unwrap()).map(|x| x as f32),
                DVector::from_column_slice(v2.as_slice().unwrap()).map(|x| x as f32),
            )
        })
        .collect();

    let test_data: Vec<_> = data
        .test_images
        .clone()
        .into_iter()
        .zip(data.test_labels.clone().iter())
        .map(|(v1, v2)| {
            (
                DVector::from_column_slice(v1.as_slice().unwrap()).map(|x| x as f32),
                DVector::from_column_slice(v2.as_slice().unwrap()).map(|x| x as f32),
            )
        })
        .collect();

    network.sdg(training_data, 20, 64, 1e-1, Some(&test_data));
}

struct Network<const N: usize> {
    weights: [DMatrix<f64>; N],
    biases: [DVector<f64>; N],
    weight_gradient: [DMatrix<f64>; N],
    bias_gradient: [DVector<f64>; N],
}

impl<const N: usize> Network<N> {
    fn new(inputs: usize, layer_sizes: [usize; N]) -> Self {
        let mut rng = rand::rng();

        let weights: [DMatrix<f64>; N] = array::from_fn(|index| {
            if index == 0 {
                DMatrix::from_fn(layer_sizes[0], inputs, |_, _| rng.random_range(-1.0..=1.0))
            } else {
                DMatrix::from_fn(layer_sizes[index], layer_sizes[index - 1], |_, _| {
                    rng.random_range(-1.0..=1.0)
                })
            }
        });

        let weight_gradient: [DMatrix<f64>; N] = array::from_fn(|index| {
            if index == 0 {
                DMatrix::zeros(layer_sizes[0], inputs)
            } else {
                DMatrix::zeros(layer_sizes[index], layer_sizes[index - 1])
            }
        });

        let biases: [DVector<f64>; N] = array::from_fn(|index| {
            DVector::from_fn(layer_sizes[index], |_, _| rng.random_range(-1.0..=1.0))
        });
        let bias_gradient: [DVector<f64>; N] =
            array::from_fn(|index| DVector::zeros(layer_sizes[index]));

        Self {
            weights,
            bias_gradient,
            biases,
            weight_gradient,
        }
    }

    fn classify(&self, inputs: &DVector<f64>) -> DVector<f64> {
        let mut inputs = inputs.clone();
        for i in 0..N {
            inputs = &self.weights[i] * inputs;
            inputs += &self.biases[i];
            inputs.apply(|val| *val = 1. / (1. + (-*val).exp())); //f64::max(*val, 0.)); 1. / (1. + (-*val).exp())));
        }

        inputs
    }

    fn cost(&self, (input, expected): &(DVector<f64>, DVector<f64>)) -> f64 {
        let error = expected - self.classify(input);
        error.component_mul(&error).sum()
    }

    fn average_cost(&self, data: &[(DVector<f64>, DVector<f64>)]) -> f64 {
        data.iter().map(|datum| self.cost(datum)).sum::<f64>() / data.len() as f64
    }

    fn update_gradients(&mut self, training_data: &[(DVector<f64>, DVector<f64>)]) {
        let h = 0.0001;
        let baseline = self.average_cost(training_data);

        for i in 0..N {
            let (rows, cols) = self.weights[i].shape();
            for row in 0..rows {
                for col in 0..cols {
                    *self.weights[i].index_mut((row, col)) += h;

                    let delta_cost = self.average_cost(training_data) - baseline;

                    *self.weights[i].index_mut((row, col)) -= h;

                    *self.weight_gradient[i].index_mut((row, col)) = delta_cost / h;
                }
            }
            for row in 0..self.biases[i].len() {
                *self.biases[i].index_mut(row) += h;

                let delta_cost = self.average_cost(training_data) - baseline;

                *self.weights[i].index_mut(row) -= h;

                *self.weight_gradient[i].index_mut(row) = delta_cost / h;
            }
        }
    }

    fn apply_gradients(&mut self, learn_rate: f64) {
        for i in 0..N {
            self.weights[i] -= &self.weight_gradient[i] * learn_rate;
            self.biases[i] -= &self.bias_gradient[i] * learn_rate;

            // self.weight_gradient[i] *= 0.0;
            // self.bias_gradient[i] *= 0.0;
        }
    }
}
