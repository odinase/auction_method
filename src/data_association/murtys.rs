
use crate::problem_solution_pair::{Solution, Problem, ProblemSolutionPair, InvalidSolutionError};
use crate::data_association::{auction, auction_params};

use ndarray::prelude::*;
use std::collections::BinaryHeap;


#[derive(Debug, Clone)]
struct ProblemDuo {
    duo: [Array2<f64>; 2],
    i: usize,
}

impl ProblemDuo {
    pub fn new() -> Self {
        ProblemDuo {
            duo: [Array::from_iter(0..5*5).into_shape((5, 5)).unwrap().mapv(|a| a as f64), Array::from_iter(0..5*5).into_shape((5, 5)).unwrap().mapv(|a| a as f64)],
            i: 0,
        }
    }
    pub fn curr_view(&self) -> ArrayView2<'_, f64> {
        self.duo[self.i].slice(s![..-1, ..-1])
    }
}

pub struct Murtys {

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
        let org_prob = P.clone();
        let mut i = problem_solution_pair.solution().first();
        
        let mut locked_targets: Vec<usize> = Vec::new();
        let mut item_idxs: Vec<_> = (0..m).collect();

        for t in 0..n {
            P.make_association_impossible(i, 0);
            if let Ok(solution) = auction(&P, auction_params::EPS, auction_params::MAX_ITERATIONS) {
                let v = solution.into_vec();
                let convert_v = get_indexed_vec(item_idxs.as_slice(), v.as_slice());
                let Qs = Solution::concatenate_assignments(&[
                &locked_targets,
                &convert_v
                ]);

                let Qp = org_prob.clone();

                let org_i = item_idxs[i];
                Qp.make_association_impossible(org_i, t);

                if let Ok(pair) = ProblemSolutionPair::new(Qs, Qp) {
                    // Here we should make sure it does not exist already?
                    if L.iter().all(|p| p != &pair) {
                        L.push(pair);
                    }
                }
            }

            locked_targets.push(item_idxs[i]);
            item_idxs.remove(i);
            if P.num_targets() > 1{
                P.delete_assignment(i);
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

