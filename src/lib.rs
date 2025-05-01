pub mod mnist;
pub mod nn;
pub mod nn_f32;

use std::iter::zip;

use nalgebra::{DMatrix, DVector};
use rand::Rng;

#[derive(Debug, Clone)]
pub struct DataPoint {
    pub inputs: Vec<f64>,
    pub label: usize,
    pub outs: Vec<f64>,
}

impl DataPoint {
    pub fn new(inputs: Vec<f64>, label: usize, outputs: usize) -> Self {
        let mut outs = vec![0.; outputs];
        outs[label] = 1.;

        Self {
            inputs,
            label,
            outs,
        }
    }
}

enum ActivationType {
    Sigmoid,
    ReLU,
}

impl ActivationType {
    pub fn activation(&self) -> fn(&mut f64) {
        match self {
            ActivationType::Sigmoid => |val: &mut f64| *val = 1. / (1. + (-*val).exp()),
            ActivationType::ReLU => |val: &mut f64| *val = val.max(0.),
        }
    }
}

struct Layer {
    weights: DMatrix<f64>,
    biases: DVector<f64>,
    activation: ActivationType,
    weight_gradient: DMatrix<f64>,
    biase_gradient: DVector<f64>,
}

impl Layer {
    fn new(inputs: usize, outputs: usize) -> Self {
        let mut rng = rand::thread_rng();
        let mut function = |_: usize, _: usize| {
            let x1: f64 = 1. - rng.gen::<f64>();
            let x2: f64 = 1. - rng.gen::<f64>();

            // (-2. * x1.log(10.)) * (2. * PI * x2).cos() / (inputs as f64).sqrt()
            rng.gen_range(-1.0..=1.0) / (inputs as f64).sqrt()
        };
        Self {
            weights: DMatrix::from_fn(outputs, inputs, &mut function),
            weight_gradient: DMatrix::from_fn(outputs, inputs, &mut function),
            biases: DVector::from_fn(outputs, &mut function),
            biase_gradient: DVector::from_fn(outputs, &mut function),
            activation: ActivationType::Sigmoid,
        }
    }
}
pub struct Network {
    layers: Vec<Layer>,
}

impl Network {
    pub fn new(layer_sizes: &[usize]) -> Self {
        let layers = layer_sizes
            .windows(2)
            .map(|vals| Layer::new(vals[0], vals[1]))
            .collect();

        Self { layers }
    }

    pub fn classify(&self, inputs: &[f64]) -> usize {
        self.calculate_outputs(inputs)
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .unwrap()
            .0
    }

    pub fn calculate_outputs(&self, inputs: &[f64]) -> DVector<f64> {
        let mut intermediate = DVector::from_column_slice(inputs);

        for layer in self.layers.iter() {
            intermediate = (&layer.weights) * intermediate;
            intermediate += &layer.biases;
            intermediate.apply(layer.activation.activation());
        }
        intermediate
    }

    pub fn learn<'a>(
        &mut self,
        training_data: impl Iterator<Item = &'a DataPoint> + Clone,
        learn_rate: f64,
    ) {
        let h = 0.0001;

        let start_cost = self.average_cost(training_data.clone());

        for layer in 0..self.layers.len() {
            for row in 0..self.layers[layer].weights.nrows() {
                for column in 0..self.layers[layer].weights.ncols() {
                    let i = (row, column);

                    self.layers[layer].weights[i] -= h;
                    let cost = self.average_cost(training_data.clone()) - start_cost;
                    self.layers[layer].weight_gradient[i] += h;

                    self.layers[layer].weight_gradient[i] -= cost / h;
                }
            }
            for row in 0..self.layers[layer].biases.nrows() {
                self.layers[layer].biases[row] -= h;
                let cost = self.average_cost(training_data.clone()) - start_cost;

                self.layers[layer].biases[row] += h;

                self.layers[layer].biase_gradient[row] -= cost / h;
            }
        }

        self.apply_gradients(learn_rate);
    }

    pub fn cost(&self, datum: &DataPoint) -> f64 {
        let expected = datum.outs.as_slice();
        let outputs = self.calculate_outputs(&datum.inputs);

        zip(expected, &outputs)
            .map(|(expected, output)| (output - expected) * (output - expected))
            .sum::<f64>()
    }

    pub fn average_cost<'a, T: Iterator<Item = &'a DataPoint>>(&self, test_data: T) -> f64 {
        let (len, sum) = test_data
            .map(|datum| self.cost(datum))
            .enumerate()
            .reduce(|(_, a), (i, b)| (i, a + b))
            .unwrap();

        sum / len as f64
    }

    pub fn apply_gradients(&mut self, learn_rate: f64) {
        for layer in self.layers.iter_mut() {
            layer.weights -= &layer.weight_gradient * learn_rate;
            layer.biases -= &layer.biase_gradient * learn_rate;
        }
    }
}

#[cfg(test)]
mod test {
    use nalgebra::{Matrix3x2, Vector2, Vector3};
    use rand::Rng;

    #[test]
    fn matrix_test() {
        let mut rng = rand::thread_rng();
        let mut generator = |_: usize, _: usize| rng.gen_range(0.0..=1.0);
        fn sigmoid(val: &mut f64) {
            *val = 1. / (1. + (-*val).exp())
        }

        let inputs = Vector2::from_fn(&mut generator);
        let weights = Matrix3x2::from_fn(&mut generator);
        let biases = Vector3::from_fn(&mut generator);

        println!("{}", inputs);
        println!("{}", weights);
        println!("{}", biases);

        let mut result = weights * inputs + biases;

        println!("{}", result);
        result.apply(&mut sigmoid);
        println!("{}", result);
        println!("{:.3}", result.transpose());

        panic!()
    }
}
