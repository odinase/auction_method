use ndarray::prelude::*;
use std::collections::{VecDeque, BinaryHeap};
use crate::argmax::argmax_iter;
use crate::problem_solution_pair::{ProblemSolutionPair, Solution, Problem, InvalidSolutionError};

use std::f64::NEG_INFINITY as neg_inf;

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

pub fn auction(problem: &Problem, eps: f64, max_iterations: usize) -> Result<Solution, InvalidSolutionError> {
    use Assignment::{Assigned, Unassigned};
    
    let (m, n) = problem.num_measurements_targets();
    let mut unassigned_queue: VecDeque<_> = (0..n).collect();
    let mut assigned_tracks: Vec<Assignment> = vec![Unassigned; n];
    let mut prices = vec![0f64; m];

    let mut curr_iter = 0;
    
    while let Some(t_star) = unassigned_queue.pop_front() {
        if curr_iter > max_iterations {
            break;
        }
        let (i_star, val_max) = argmax_iter(
            problem.rewards(t_star)
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
        
        let y = problem.reward(i_star, t_star) - val_max;
        prices[i_star] += y + eps;
        curr_iter += 1;
    }
    // We return a Result<Solution> here as we might have terminated early and not assigned all items
    Solution::try_from_unvalidated_assignments(assigned_tracks)
}

pub fn murtys(original_problem: Problem, N: usize) -> Result<Vec<ProblemSolutionPair>, InvalidSolutionError> {
    let (m, n) = original_problem.num_measurements_targets();
    let original_solution = auction(&original_problem, auction_params::EPS, auction_params::MAX_ITERATIONS)?;
    let problem_solution_pair = ProblemSolutionPair::new(original_solution, original_problem.clone())?; // TODO: Fix this later
    let mut L = BinaryHeap::new();
    L.push(problem_solution_pair);

    let mut R = Vec::new();

    while let Some(problem_solution_pair) = L.pop() {
        // TODO: Be smarter here
        R.push(problem_solution_pair.clone());

        if R.len() == N {
            break;
        }

        let P = problem_solution_pair.problem();
        let mut i = problem_solution_pair.solution().first();
        
        let mut locked_targets: Vec<usize> = Vec::new();
        let mut item_idxs: Vec<_> = (0..P.num_measurements()).collect();

        for t in 0..n {
            // println!("make ")
            P.make_association_impossible(i, 0);
            println!("Solving problem: {:?}", P);
            if let Ok(solution) = auction(&P, auction_params::EPS, auction_params::MAX_ITERATIONS) {
                let v = solution.into_vec();
                println!("Solution: {:?}", v);
                let convert_v = get_indexed_vec(item_idxs.as_slice(), v.as_slice());
                println!("\n\nconverted : {:?}\n\n", convert_v);
                let Qs = Solution::concatenate_assignments(&[
                &locked_targets,
                &convert_v
                ]);

                let Qp = original_problem.clone();

                let org_i = item_idxs[i];
                Qp.make_association_impossible(org_i, t);

                if let Ok(pair) = ProblemSolutionPair::new(Qs, Qp) {
                    // Here we should make sure it does not exist already?
                    println!("adding pair:\n{:?}", pair);
                    L.push(pair);
                }
            } else {
                println!("invalid solution!");
            }
            locked_targets.push(item_idxs[i]);
            println!("\n\nlocked_items: {:?}\n\n", locked_targets);
            item_idxs.remove(i);
            println!("\n\nitem_idxs: {:?}\n\n", item_idxs);
            if P.num_targets() > 1{
                println!("About to delete measurement {}, P is {:?}", i, P);
                P.delete_assignment(i);
                println!("P is now {:?}", P);
            } else {
                break;
            }
            // if P.is_empty() {
            //     break;
            // }
            
            i = item_idxs.iter().position(|&i| i == problem_solution_pair.solution().assigned_measurement(t+1).unwrap()).expect("Cant find index??");
        }
    }

    Ok(R)
}


fn get_indexed_vec<T: Copy>(v: &[T], idx: &[usize]) -> Vec<T> {
    let mut index_vec = Vec::with_capacity(idx.len());
    for &i in idx {
        index_vec.push(v[i]);
    }
    index_vec
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