use braille_rs::BrailleChar;
use nalgebra::DVector;
use std::io::Read;

#[derive(Default, Debug, Clone, PartialEq)]
pub struct Mnist {
    pub test_images: Vec<DVector<f64>>,
    pub test_labels: Vec<DVector<f64>>,
    pub training_images: Vec<DVector<f64>>,
    pub training_labels: Vec<DVector<f64>>,
}

impl Mnist {
    pub fn training_data_f64(&self) -> Vec<(DVector<f64>, DVector<f64>)> {
        self.training_images
            .clone()
            .into_iter()
            .zip(self.training_labels.clone())
            .collect()
    }

    pub fn training_data_f32(&self) -> Vec<(DVector<f32>, DVector<f32>)> {
        self.training_images
            .clone()
            .into_iter()
            .zip(self.training_labels.clone())
            .map(|(v1, v2)| (v1.map(|x| x as f32), v2.map(|x| x as f32)))
            .collect()
    }

    pub fn test_data_f64(&self) -> Vec<(DVector<f64>, DVector<f64>)> {
        self.test_images
            .clone()
            .into_iter()
            .zip(self.test_labels.clone())
            .collect()
    }

    pub fn test_data_f32(&self) -> Vec<(DVector<f32>, DVector<f32>)> {
        self.test_images
            .clone()
            .into_iter()
            .zip(self.test_labels.clone())
            .map(|(v1, v2)| (v1.map(|x| x as f32), v2.map(|x| x as f32)))
            .collect()
    }
}

type UResult<T> = Result<T, Box<dyn std::error::Error>>;

pub fn read(path: impl ToString) -> UResult<Mnist> {
    let test_images = std::fs::File::open(path.to_string() + "t10k-images-idx3-ubyte")?;
    let test_labels = std::fs::File::open(path.to_string() + "t10k-labels-idx1-ubyte")?;
    let training_images = std::fs::File::open(path.to_string() + ("train-images-idx3-ubyte"))?;
    let training_labels = std::fs::File::open(path.to_string() + ("train-labels-idx1-ubyte"))?;

    // let handles = [test_images, test_labels, training_images, training_labels];
    let output = Mnist {
        test_images: parse3(test_images)?,
        test_labels: parse1(test_labels)?,
        training_images: parse3(training_images)?,
        training_labels: parse1(training_labels)?,
    };

    Ok(output)
}

fn parse1(mut handle: std::fs::File) -> UResult<Vec<DVector<f64>>> {
    assert!(read_header(&mut handle, 1)?);

    let mut tmp = vec![0u8; 4];
    handle.read_exact(&mut tmp)?;

    let mut raw_bytes = Vec::new();
    handle.read_to_end(&mut raw_bytes)?;
    let mut labels = Vec::new();

    let empty = DVector::zeros(10);

    for byte in raw_bytes {
        let mut val = empty.clone();
        val[byte as usize] = 1.0;
        labels.push(val)
    }

    Ok(labels)
}

fn parse3(mut handle: std::fs::File) -> UResult<Vec<DVector<f64>>> {
    assert!(read_header(&mut handle, 3)?);

    let mut tmp = vec![0u8; 3 * 4];
    handle.read_exact(&mut tmp)?;

    let mut raw_bytes = Vec::new();
    handle.read_to_end(&mut raw_bytes)?;

    let floats: Vec<f64> = raw_bytes
        .into_iter()
        .map(|byte| byte as f64 / 255.)
        .collect();

    Ok(floats
        .chunks_exact(28 * 28)
        .map(|chunk| DVector::from(chunk.to_vec()))
        .collect())

    // Ok(Array::from_shape_vec(shape, raw)?)
}

fn read_header(handle: &mut std::fs::File, dims: u8) -> UResult<bool> {
    let mut header = [0u8; 4];
    handle.read_exact(&mut header)?;

    match header.map(u8::from_be) {
        [0, 0, 8, d] if d == dims => Ok(true),
        _ => Err("first 3 bytes must be 0x00, 0x00, 0x08")?,
    }
}

// impl Mnist {
//     pub fn training_images_flat_normalized(&self) -> Vec<Array1<f32>> {
//         let images = self.training_images.clone().axis_iter(Axis(0)).clone();

//         images.map(ArrayBase::into_flat).collect()
//     }
// }

pub fn render(data: &[f64], threshhold: f64) -> String {
    let mut canvas = String::new();

    let mut grid = [false; 28 * 28];

    for (i, cell) in grid.iter_mut().enumerate() {
        *cell = data[i] >= threshhold;
    }

    let lines = grid.chunks_exact(28 * 4);

    for line in lines {
        for i in 0..14 {
            let byte = ((line[0 + 2 * i] as u8) << 0)
                | ((line[28 + (2 * i)] as u8) << 1)
                | ((line[28 * 2 + 2 * i] as u8) << 2)
                | ((line[28 * 3 + (2 * i)] as u8) << 3)
                | ((line[0 + (2 * i + 1)] as u8) << 4)
                | ((line[28 + (2 * i + 1)] as u8) << 5)
                | ((line[28 * 2 + (2 * i + 1)] as u8) << 6)
                | ((line[28 * 3 + (2 * i + 1)] as u8) << 7);

            canvas.push(BrailleChar::with_data(byte).into())
        }
        canvas.push('\n')
    }

    canvas
}
