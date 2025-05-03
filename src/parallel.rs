use std::{
    sync::{Arc, Mutex},
    thread,
};

use nalgebra::{DMatrix, DVector};

use crate::{nn_f32::Network, Activation};

type Arctex<T> = Arc<Mutex<T>>;
pub type Gradient<T> = Vec<(DMatrix<T>, DVector<T>)>;

pub struct ThreadGrads {
    gradients: Vec<Arctex<Gradient<f32>>>,
    threads: usize,
}
impl ThreadGrads {
    pub fn new(sizes: &[usize], threads: usize) -> Self {
        let layer: Vec<_> = std::iter::zip(
            std::iter::zip(&sizes[..sizes.len() - 1], &sizes[1..])
                .map(|(x, y)| DMatrix::zeros(*y, *x)),
            sizes.iter().skip(1).map(|y| DVector::zeros(*y)),
        )
        .collect();

        Self {
            gradients: Vec::from_iter(std::iter::repeat_n(
                Arc::new(Mutex::new(layer.clone())),
                threads,
            )),
            threads,
        }
    }

    pub fn apply(&mut self, layers: &mut Gradient<f32>, lr_bs: f32) {
        for grad in self.gradients.iter() {
            let mut current = grad.lock().unwrap();

            for i in 0..current.len() {
                let (weights, biases) = &mut layers[i];

                *weights -= &current[i].0 * lr_bs;
                *biases -= &current[i].1 * lr_bs;

                current[i].0 *= 0.01;
                current[i].1 *= 0.01;
            }
        }
    }

    pub fn backprop_par(
        net: Arc<Gradient<f32>>,
        grad: Arctex<Gradient<f32>>,
        activation: Activation,
        output_activation: Activation,
        x: DVector<f32>,
        y: &DVector<f32>,
    ) {
        // feedforward
        let mut activations = vec![x]; // list to store all the activations, layer by layer
        let mut zs = Vec::new(); // list to store all the z vectors, layer by layer

        for (i, (w, b)) in net.iter().enumerate() {
            let mut z = w * activations.last().unwrap() + b;
            zs.push(z.clone());

            if i == net.len() {
                z.apply(output_activation.get_fun32());
            } else {
                z.apply(activation.get_fun32())
            }
            activations.push(z);
        }

        // backward pass
        let mut delta = Network::cost_derivative(&activations.pop().unwrap(), y)
            .component_mul(&zs.pop().unwrap().apply_into(output_activation.get_dir32()));

        let mut grad = grad.lock().unwrap();
        grad.last_mut().unwrap().0 += &delta * &activations.pop().unwrap().transpose();
        grad.last_mut().unwrap().1 += &delta;

        // Note that the variable l in the loop below is used a little
        // differently to the notation in Chapter 2 of the book.  Here,
        // l = 1 means the last layer of neurons, l = 2 is the
        // second-last layer, and so on.  It's a renumbering of the
        // scheme in the book, used here to take advantage of the fact
        // that Python can use negative indices in lists.
        let length = grad.len();

        for l in 2..(length + 1) {
            // let z = zs[zs_len - l].clone();
            let z = zs.pop().unwrap();
            // let sp = z.apply_into(|z| *z = relu_prime(*z));
            let sp = z.apply_into(activation.get_dir32());
            delta = net[length - l + 1].0.tr_mul(&delta).component_mul(&sp);

            grad[length - l].0 += &delta * &activations.pop().unwrap().transpose();
            grad[length - l].1 += &delta;
        }
    }

    pub fn update_mini_batch(
        &mut self,
        net: Arc<Gradient<f32>>,
        activation: Activation,
        output_activation: Activation,
        mini_batch: Arc<[(DVector<f32>, DVector<f32>)]>,
    ) {
        // initialize empty gradients
        // the small coefficient is an experiment based on the fact that the gradients are probably similar enough
        // to the previous batch's that leaving a small remnant from the earlier batch will help.

        let size = mini_batch.len() / self.threads;

        let mut indexes = Vec::new();

        for i in 0..self.threads {
            if i == self.threads - 1 {
                indexes.push(i * size..mini_batch.len())
            }
            indexes.push(i * size..(i + 1) * size)
        }

        let mut handles = Vec::new();
        for i in 0..self.threads {
            let index = indexes[i].clone();
            let new_net = net.clone();
            let grad = Arc::clone(&self.gradients[i]);
            let new_batch = mini_batch.clone();

            let handle = thread::spawn(move || {
                for (x, y) in new_batch[index].iter() {
                    Self::backprop_par(
                        new_net.clone(),
                        grad.clone(),
                        activation,
                        output_activation,
                        x.clone(),
                        y,
                    );
                }
            });

            handles.push(handle);
        }

        for i in 0..handles.len() {}

        // update the gradients for every example in the batch
    }
}
