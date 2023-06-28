use snake_tuner::{
    database::{Database, Entry},
    dataloader::DataLoader,
};
// Lets start by making a struct to hold the database.
// This could be anything, even digging into an sqlite file.
#[derive(Debug, Clone)]
struct MyDB {
    items: Vec<[f64; 3]>,
    labels: Vec<f64>,
}

#[derive(Debug, Clone)]
struct MyEntry {
    input: [f64; 3],
    output: f64,
}

impl Entry<3> for MyEntry {
    fn get_inputs(&self) -> [f64; 3] {
        self.input
    }

    fn get_expected_output(&self) -> f64 {
        self.output
    }
}

// This is where we implement the `Database<N>` trait
impl Database<3, MyEntry> for MyDB {
    fn size(&self) -> usize {
        return self.items.len();
    }

    fn get(&self, idx: usize) -> MyEntry {
        return MyEntry {
            input: self.items[idx],
            output: self.labels[idx],
        };
    }
}

fn main() {
    let my_db = MyDB {
        items: vec![[1.0, 0.0, 13.0], [17.0, 0.0, 10.0], [15.0, 3.0, 1.0]],
        labels: vec![5.0, 3.0, 0.0],
    };

    // Initialize a dataloader on the database above with a batch size of 1 and shuffle on.
    let mut dataloader = DataLoader::new(my_db, 1, true);

    // Sample the dataloader to get an entry.
    println!("{:?}", dataloader.sample());
}
