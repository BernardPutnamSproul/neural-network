pub mod mnist;
pub mod nn;
pub mod nn_f32;

#[repr(usize)]
pub enum Activation {
    Sigmoid = 0,
    ReLU = 1,
    Identity = 2,
    SiLU = 3,
}

/// handle little macro so I don't have to write out functions for f32 and f64 types.
macro_rules! twice {
    ($x:expr) => {
        ($x, $x)
    };
}

type A32 = fn(&mut f32);
type A64 = fn(&mut f64);
type Zipped = (A32, A64);

impl Activation {
    const RELU: Zipped = twice!(|x| *x = x.max(0.0));
    const DRELU: Zipped = twice!(|x| *x = if *x > 0. { 1. } else { 0. });

    const SIGMOID: Zipped = twice!(|x| *x = 1.0 / (1.0 + (-*x).exp()));
    const DSIGMOID: Zipped = twice!(|x| {
        let tmp = (-*x).exp();
        *x = (1. - (1.0 / (1.0 + tmp))) / (1.0 + tmp);
    });

    const IDENTITY: Zipped = twice!(|_x| ());
    const DIDENTITY: Zipped = twice!(|x| *x = 1.);

    const SILU: Zipped = twice!(|x| *x = *x / (1. + (-*x).exp()));
    const DSILU: Zipped = twice!(|x| {
        let tmp = (-*x).exp();
        *x = (1. + tmp * (1. + *x)) / (1. + tmp).powi(2);
    });

    const LOOKUP: [Zipped; 4] = [Self::SIGMOID, Self::RELU, Self::IDENTITY, Self::SILU];
    const DLOOKUP: [Zipped; 4] = [Self::DSIGMOID, Self::DRELU, Self::DIDENTITY, Self::DSILU];

    pub const fn get_fun32(&self) -> A32 {
        Self::LOOKUP[self.to_index()].0
    }

    pub const fn get_fun64(&self) -> A64 {
        Self::LOOKUP[self.to_index()].1
    }

    pub const fn get_dir32(&self) -> A32 {
        Self::DLOOKUP[self.to_index()].0
    }

    pub const fn get_dir64(&self) -> A64 {
        Self::DLOOKUP[self.to_index()].1
    }

    const fn to_index(&self) -> usize {
        match self {
            Activation::Sigmoid => 0,
            Activation::ReLU => 1,
            Activation::Identity => 2,
            Activation::SiLU => 3,
        }
    }
}
