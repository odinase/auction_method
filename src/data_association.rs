use ndarray::prelude::*;
use std::collections::{VecDeque, BinaryHeap};
use crate::argmax::argmax_iter;
use crate::problem_solution_pair::{ProblemSolutionPair, Solution, Problem};

use std::f64::NEG_INFINITY as neg_inf;


// Lets use this at some point :)
#[derive(Copy, Clone, Debug)]
pub enum Assignment {
    Assigned(usize),
    Unassigned,
}

pub fn auction(A: &Array2<f64>, eps: f64) -> Vec<Assignment> {
    use Assignment::{Assigned, Unassigned};
    
    let (m, n) = A.dim();
    let mut unassigned_queue: VecDeque<_> = (0..n).collect();
    let mut assigned_tracks: Vec<Assignment> = vec![Unassigned; n];
    let mut prices = vec![0f64; m];
    
    while let Some(t_star) = unassigned_queue.pop_front() {
        let (i_star, val_max) = argmax_iter(
            A.column(t_star)
            .into_iter()
            .zip(prices.iter())
            .map(|(reward, &price)| reward - price as f64),
        );
        let prev_owner = assigned_tracks.iter().position(|&e| 
            match e {
                Assigned(e) => e == i_star,
                Unassigned => false,
            }
        );
        assigned_tracks[t_star] = Assigned(i_star);
        
        if let Some(prev_owner) = prev_owner {
            // The item has a previous owner
            assigned_tracks[prev_owner] = Unassigned;
            unassigned_queue.push_back(prev_owner);
        }
        
        let y = A[(i_star, t_star)] - val_max;
        prices[i_star] += y + eps;
    }
    
    assigned_tracks
}

pub fn murtys(A: Array2<f64>, N: usize) -> Vec<ProblemSolutionPair> {
    let (m, n) = A.dim();
    let As = auction(&A, 1e-3);
    let problem_solution_pair = ProblemSolutionPair::new(Solution(As), Problem(A)).unwrap(); // TODO: Fix this later
    let mut L = BinaryHeap::new();
    L.push(problem_solution_pair);

    let mut R = Vec::new();

    while let Some(mut problem_solution_pair) = L.pop() {
        // TODO: Be smarter here
        R.push(problem_solution_pair.clone());

        if R.len() == N {
            break;
        }

        let mut P = problem_solution_pair.problem();
        let mut i = problem_solution_pair.solution().0[0];
        
        let mut locked_targets = Vec::new();
        let mut item_idxs = (0..P.0.shape()[0]).collect();

        for t in 0..n {

            P.0[(i, 0)] = neg_inf;
        }
    }

    R
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
}