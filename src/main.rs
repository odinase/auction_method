#![allow(non_snake_case)]
use ndarray::prelude::*;

use auction_method::data_association::{auction, Assignment};
use std::time::Instant;


fn main() {
    use std::f64::INFINITY as inf;
    let A = array![
        [-5.69, 5.37, -inf],
        [-inf, -3.8, 6.58],
        [4.78, -inf, -inf],
        [-inf, 5.36, -inf],
        [-0.46, -inf, -inf],
        [-inf, -0.52, -inf],
        [-inf, -inf, -0.60]
    ];
    let start = Instant::now();
    let assigned_tracks = auction(&A, 1e-3);
    let stop = start.elapsed().as_secs_f64()*1e6;
    println!("ran in {:.2} us", stop);
    for (t, j) in assigned_tracks.iter().enumerate() {
        match j {
            Assignment::Assigned(j) => println!("a({}) = {}", t+1, j+1),
            Assignment::Unassigned => println!("a({}) = unassigned", t+1),
        }
    }
}
