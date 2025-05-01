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
    let mut network = nn_f32::Network2::new(&[28 * 28, 16, 16, 16, 10]);

    let training_data: Vec<_> = data
        .training_images
        .clone()
        .into_iter()
        .zip(data.training_labels.clone().as_slice())
        .map(|(v1, v2)| (v1.map(|x| x as f32), v2.map(|x| x as f32)))
        .collect();

    let test_data: Vec<_> = data
        .test_images
        .clone()
        .into_iter()
        .zip(data.test_labels.clone().iter())
        .map(|(v1, v2)| (v1.map(|x| x as f32), v2.map(|x| x as f32)))
        .collect();

    network.sdg(training_data, 20, 64, 1e-1, Some(&test_data));
}
