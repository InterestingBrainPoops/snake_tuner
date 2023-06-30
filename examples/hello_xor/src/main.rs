use nalgebra::DVector;
use snake_tuner::{
    activation::functions::Sigmoid,
    database::{Database, Entry},
    dataloader::DataLoader,
    net::Net,
    tuner::Tuner,
};

#[derive(Clone)]
struct MyDB {
    inputs: Vec<[f64; 2]>,
    outputs: Vec<f64>,
}

#[derive(Clone)]
struct MyEntry {
    input: [f64; 2],
    output: f64,
}

impl Entry<2> for MyEntry {
    fn get_inputs(&self) -> DVector<f64> {
        DVector::from(self.input.to_vec())
    }

    fn get_expected_output(&self) -> DVector<f64> {
        DVector::from(vec![self.output])
    }
}

impl Database<2, MyEntry> for MyDB {
    fn size(&self) -> usize {
        self.inputs.len()
    }

    fn get(&self, idx: usize) -> MyEntry {
        MyEntry {
            input: self.inputs[idx],
            output: self.outputs[idx],
        }
    }
}

fn main() {
    let database = MyDB {
        inputs: vec![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        outputs: vec![0.0, 1.0, 1.0, 0.0],
    };
    let dataloader = DataLoader::new(database.clone(), 1, true);
    let mut non_rand = DataLoader::new(database, 1, false);

    let mut mynet = Net::new(vec![2, 2, 1]);
    let mut tuner = Tuner::new(dataloader.clone(), 0.1);
    for _ in 0..100000 {
        tuner.step::<Sigmoid>(&mut mynet);
    }
    for _ in 0..4 {
        let sample = non_rand.sample()[0].clone();

        println!(
            "{:?}",
            sample.get_expected_output() - mynet.forward::<Sigmoid>(sample.get_inputs())
        );

        println!(
            "in: {}, out: {}",
            sample.get_inputs(),
            mynet.forward::<Sigmoid>(sample.get_inputs())
        );
    }
}
