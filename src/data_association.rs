use std::collections::VecDeque;
use crate::argmax::argmax_iter;
use crate::problem_solution_pair::{Solution, Problem, InvalidSolutionError};
use ndarray::prelude::*;

pub mod murtys;


pub mod auction_params {
    pub const EPS: f64 = 1e-3;
    pub const MAX_ITERATIONS: usize = 10_000; 
}
    
#[derive(Copy, Clone, Debug)]
pub enum Assignment {
    Assigned(usize),
    Unassigned,
}

impl std::cmp::PartialEq<usize> for Assignment  {
    fn eq(&self, other: &usize) -> bool {
        match self {
            Self::Assigned(e) => e == other,
            Self::Unassigned => false,
        }
    }
}

pub fn auction<S>(problem: &ArrayBase<S, Ix2>, eps: f64, max_iterations: usize) -> Vec<Assignment> 
where
S: ndarray::RawData<Elem = f64> + ndarray::Data,
{
    use Assignment::{Assigned, Unassigned};
    
    let (m, n) = problem.dim();
    let mut unassigned_queue: VecDeque<_> = (0..n).collect();
    let mut assigned_tracks: Vec<Assignment> = vec![Unassigned; n];
    let mut prices = vec![0f64; m];

    let mut curr_iter = 0;
    
    while let Some(t_star) = unassigned_queue.pop_front() {
        if curr_iter > max_iterations {
            break;
        }
        let (i_star, val_max) = argmax_iter(
            problem.column(t_star)
            .into_iter()
            .zip(prices.iter())
            .map(|(reward, &price)| reward - price),
        );
        let prev_owner = assigned_tracks.iter().position(|&e| e == i_star);
        assigned_tracks[t_star] = Assigned(i_star);
        
        if let Some(prev_owner) = prev_owner {
            // The item has a previous owner
            assigned_tracks[prev_owner] = Unassigned;
            unassigned_queue.push_back(prev_owner);
        }
        
        let y = problem[(i_star, t_star)] - val_max;
        prices[i_star] += y + eps;
        curr_iter += 1;
    }
    
    assigned_tracks
}



#[cfg(test)]
mod test {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_auction() {
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
        let problem = Problem::new(A);
        let start = Instant::now();
        let assigned_tracks = auction(&problem, auction_params::EPS, auction_params::MAX_ITERATIONS);
        let stop = start.elapsed().as_secs_f64()*1e6;
        println!("ran in {:.2} us", stop);
        match assigned_tracks {
            Ok(assigned_tracks) => {
                for (t, j) in assigned_tracks.assignments() {
                    println!("a({}) = {}", t+1, j+1);
                }
            },
            Err(e) => println!("Auction ended with error {}", e),
        };
    }
}