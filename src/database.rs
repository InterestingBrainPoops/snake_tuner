pub struct Entry<const N: usize> {
    pub inputs: [f64; N],
    pub expected: f64,
}

pub trait Database<const N: usize> {
    fn new() -> Self;
    fn size(&self) -> usize;
    fn get(&self, idx: usize) -> Entry<N>;
}
