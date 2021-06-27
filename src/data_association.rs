use ndarray::prelude::*;
use std::collections::VecDeque;


pub struct Solution(Vec<usize>);
pub struct Problem(Array2<f64>);



pub fn auction(A: Array2<f64>, eps: f64) -> Solution {
    let (m, n) = A.dim();
    let unassigned_queue: VecDeque<_> = (0..n).collect();
    let assigned_tracks = vec![-1; n];
    todo!()
}