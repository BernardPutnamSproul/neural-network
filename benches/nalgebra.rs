#![allow(unused)]

use criterion::BenchmarkId;
use nalgebra::{ArrayStorage, Const, DMatrix, DVector, Dyn, Matrix, VecStorage, Vector};
use rand::Rng;

use criterion::{black_box, criterion_group, criterion_main, Criterion};

pub fn generate_random_inputs_const_f64() -> (
    Vector<f64, Const<784>, ArrayStorage<f64, 784, 1>>,
    Matrix<f64, Const<512>, Const<784>, ArrayStorage<f64, 512, 784>>,
    Vector<f64, Const<512>, ArrayStorage<f64, 512, 1>>,
) {
    let mut rng = rand::rng();

    (
        Vector::from_array_storage(ArrayStorage::<f64, 784, 1>([std::array::from_fn(
            |_| -> f64 { rng.random_range(0.0..=1.0) },
        )])),
        Matrix::from_array_storage(ArrayStorage::<f64, 512, 784>(std::array::from_fn(|_| {
            std::array::from_fn(|_| rng.random_range(0.0..=1.0))
        }))),
        Vector::from_array_storage(ArrayStorage::<f64, 512, 1>([std::array::from_fn(|_| {
            rng.random_range(0.0..=1.0)
        })])),
    )
}

pub fn generate_random_inputs_dyn_f64() -> (
    Vector<f64, Dyn, VecStorage<f64, Dyn, Const<1>>>,
    Matrix<f64, Dyn, Dyn, VecStorage<f64, Dyn, Dyn>>,
    Vector<f64, Dyn, VecStorage<f64, Dyn, Const<1>>>,
) {
    let mut rng = rand::rng();

    (
        DVector::from_fn(784, |_, _| rng.random_range(0.0..=10.0)),
        DMatrix::from_fn(512, 784, |_, _| rng.random_range(0.0..=10.0)),
        DVector::from_fn(512, |_, _| rng.random_range(0.0..=10.0)),
    )
}

pub fn generate_random_inputs_const_f32() -> (
    Vector<f32, Const<784>, ArrayStorage<f32, 784, 1>>,
    Matrix<f32, Const<512>, Const<784>, ArrayStorage<f32, 512, 784>>,
    Vector<f32, Const<512>, ArrayStorage<f32, 512, 1>>,
) {
    let mut rng = rand::rng();

    (
        Vector::from_array_storage(ArrayStorage::<f32, 784, 1>([std::array::from_fn(
            |_| -> f32 { rng.random_range(0.0..=1.0) },
        )])),
        Matrix::from_array_storage(ArrayStorage::<f32, 512, 784>(std::array::from_fn(|_| {
            std::array::from_fn(|_| rng.random_range(0.0..=1.0))
        }))),
        Vector::from_array_storage(ArrayStorage::<f32, 512, 1>([std::array::from_fn(|_| {
            rng.random_range(0.0..=1.0)
        })])),
    )
}

pub fn generate_random_inputs_dyn_f32() -> (
    Vector<f32, Dyn, VecStorage<f32, Dyn, Const<1>>>,
    Matrix<f32, Dyn, Dyn, VecStorage<f32, Dyn, Dyn>>,
    Vector<f32, Dyn, VecStorage<f32, Dyn, Const<1>>>,
) {
    let mut rng = rand::rng();

    (
        DVector::from_fn(784, |_, _| rng.random_range(0.0..=10.0)),
        DMatrix::from_fn(512, 784, |_, _| rng.random_range(0.0..=10.0)),
        DVector::from_fn(512, |_, _| rng.random_range(0.0..=10.0)),
    )
}

pub fn sigmoid_f64(val: &mut f64) {
    *val = 1. / (1. + (-*val).exp())
}

pub fn sigmoid_f32(val: &mut f32) {
    *val = 1. / (1. + (-*val).exp())
}

pub fn relu_f64(val: &mut f64) {
    *val = val.max(0.);
}

pub fn relu_f32(val: &mut f32) {
    *val = val.max(0.);
}

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = rand::rng();

    let mut group = c.benchmark_group("layer calculations");

    // for (i, f) in [sigmoid, relu].iter().enumerate() {
    let i = 0;
    let name = match i {
        0 => "sigmoid",
        1 => "relu",
        _ => "idk",
    };

    group.throughput(criterion::Throughput::Bytes((785 * 513 - 1) as u64));
    group.bench_with_input(
        BenchmarkId::new("784->512 full layer calculation constant f64", name),
        &generate_random_inputs_const_f64(),
        |b, input| {
            let mut output: Vector<f64, Const<512>, ArrayStorage<f64, 512, 1>> =
                Vector::from_array_storage(ArrayStorage([[0.; 512]]));

            b.iter(|| {
                let (inputs, weights, biases) = input;

                // (weights * inputs + biases).apply_into(relu);
                weights.mul_to(inputs, &mut output);
                output += biases;
                output.apply_into(sigmoid_f64);
            })
        },
    );

    group.bench_with_input(
        BenchmarkId::new("784->512 full layer calculation dynamic f64", name),
        &generate_random_inputs_dyn_f64(),
        |b, input| {
            b.iter(|| {
                let (inputs, weights, biases) = input;
                let mut output = DVector::zeros(512);
                weights.mul_to(inputs, &mut output);
                output += biases;
                output.apply_into(sigmoid_f64);
            })
        },
    );

    // group.bench_with_input(
    //     BenchmarkId::new("784->512 full layer calculation constant f32", name),
    //     &generate_random_inputs_const_f32(),
    //     |b, input| {
    //         b.iter(|| {
    //             let (inputs, weights, biases) = input;

    //             // (weights * inputs + biases).apply_into(relu);
    //             let mut output: Vector<f32, Const<512>, ArrayStorage<f32, 512, 1>> =
    //                 Vector::from_array_storage(ArrayStorage([[0.; 512]]));
    //             weights.mul_to(inputs, &mut output);
    //             output += biases;
    //             output.apply(relu_f32);
    //         })
    //     },
    // );

    // group.bench_with_input(
    //     BenchmarkId::new("784->512 full layer calculation dynamic f32", name),
    //     &generate_random_inputs_dyn_f32(),
    //     |b, input| {
    //         b.iter(|| {
    //             let (inputs, weights, biases) = input;
    //             let mut output = DVector::zeros(512);
    //             weights.mul_to(inputs, &mut output);
    //             output += biases;
    //             output.apply(relu_f32);
    //         })
    //     },
    // );

    // }
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
