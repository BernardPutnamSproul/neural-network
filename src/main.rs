use std::env;

use neural_network::{mnist, nn_f32};

fn main() {
    env::set_var("RUST_BACKTRACE", "1");

    let data = mnist::read("data/").unwrap();
    // for i in 0..8 {
    //     print!(
    //         "[{}]:\n{}",
    //         i,
    //         mnist::render(data.test_images[i].as_slice().unwrap(), 0.95),
    //     );
    // }
    let mut network = nn_f32::Network::new(
        &[28 * 28, 32, 16, 16, 10],
        // &[28 * 28, 512, 128, 32, 16, 10],
        neural_network::Activation::Sigmoid,
        neural_network::Activation::Identity,
        rand_distr::StandardNormal,
    );

    let training_data = data.training_data_f32();

    let test_data = data.test_data_f32();

    // network.sdg_par(training_data, 50, 32, 1e-1, Some(&test_data), 4);
    network.sdg(training_data, 50, 32, 1e-1, Some(&test_data));
}
