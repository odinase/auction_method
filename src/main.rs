#![allow(non_snake_case)]
use ndarray::prelude::*;

use auction_method::data_association::{auction_params, murtys::murtys, auction, Assignment};
use auction_method::problem_solution_pair::{Solution, Problem};
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
    // let problem = Problem::new(A);
    // let start = Instant::now();
    // let assigned_tracks = auction(&problem, auction_params::EPS, auction_params::MAX_ITERATIONS);
    // let stop = start.elapsed().as_secs_f64()*1e6;
    // println!("ran in {:.2} us", stop);
    // match assigned_tracks {   
    //     Ok(assigned_tracks) => {
    //         for (t, j) in assigned_tracks.assignments() {
    //             println!("a({}) = {}", t+1, j+1);
    //         }
    //     },
    //     Err(e) => println!("Auction ended with error {}", e),
    // }
    let start = Instant::now();
    let solutions = murtys(A, 19);
    let stop = start.elapsed().as_secs_f64()*1e6;
    println!("ran in {:.2} us", stop);
    if let Ok(s) = solutions {
        println!("{:#?}", s);
    }
}
