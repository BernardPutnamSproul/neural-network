use neural_network::{mnist, nn_f32};

fn main() {
    let data = mnist::read("data/").unwrap();
    // for i in 0..8 {
    //     print!(
    //         "[{}]:\n{}",
    //         i,
    //         mnist::render(data.test_images[i].as_slice().unwrap(), 0.95),
    //     );
    // }
    let mut network = nn_f32::Network2::new(
        &[28 * 28, 32, 16, 16, 10],
        neural_network::Activation::Sigmoid,
        neural_network::Activation::Identity,
        rand_distr::StandardNormal,
    );

    let training_data = data.training_data_f32();

    let test_data = data.test_data_f32();

    network.sdg(training_data, 40, 64, 7e-2, Some(&test_data));
}
