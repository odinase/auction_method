use super::Assignment;
use crate::data_association::{auction, auction_params};
use crate::problem_solution_pair::{self, InvalidSolutionError, Problem, Solution};

use ndarray::prelude::*;
use ordered_float::OrderedFloat;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Debug, Clone)]
struct ProblemDuo {
    duo: [Array2<f64>; 2],
    i: usize,
    assignments_removed: i32,
}

impl ProblemDuo {
    fn new(problem: Array2<f64>) -> Self {
        let d = problem.raw_dim();
        ProblemDuo {
            duo: [problem, Array2::zeros(d)],
            i: 0,
            assignments_removed: 0,
        }
    }

    pub fn make_association_impossible(&mut self, measurement: usize, target: usize) {
        let mut prob = self.curr_view_mut();
        prob[(measurement, target)] = std::f64::NEG_INFINITY;
    }

    fn curr_view(&self) -> ArrayView2<'_, f64> {
        self.duo[self.i].slice(s![..-self.assignments_removed, ..-self.assignments_removed])         
    }

    fn curr_view_mut(&mut self) -> ArrayViewMut2<'_, f64> {
        self.duo[self.i].slice_mut(s![..-self.assignments_removed, ..-self.assignments_removed])
    }

    fn next_view_mut(&mut self) -> ArrayViewMut2<'_, f64> {
        self.duo[(self.i + 1) % 2].slice_mut(s![..-self.assignments_removed, ..-self.assignments_removed])
    }

    fn delete_assignment(&mut self, m: usize) {
        let mut next_prob = self.next_view_mut();
        let curr_prob = self.curr_view();
        for (j, mut r) in next_prob
            .rows_mut()
            .into_iter()
            .enumerate()
        {
            let j_offset = !(j < m) as usize;
            r.assign(&curr_prob.row(j + j_offset).slice(s![1..]));
        }
        self.assignments_removed += 1;
        self.i = (self.i + 1) % 2;
    }
}

#[derive(Debug, Clone)]
struct ProblemSolutionPair {
    problem: Array2<f64>,
    solution: Vec<usize>,
    reward: OrderedFloat<f64>,
}

impl Ord for ProblemSolutionPair {
    fn cmp(&self, other: &Self) -> Ordering {
        self.reward.cmp(&other.reward)
    }
}

impl PartialOrd for ProblemSolutionPair {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for ProblemSolutionPair {
    fn eq(&self, other: &Self) -> bool {
        self.reward == other.reward
    }
}

impl Eq for ProblemSolutionPair {}

impl ProblemSolutionPair {
    pub fn new(solution: Vec<usize>, problem: Array2<f64>) -> Result<Self, InvalidSolutionError> {
        let reward = Self::calc_reward(&solution, &problem)?;
        Ok(ProblemSolutionPair {
            solution,
            problem,
            reward: OrderedFloat(reward),
        })
    }

    pub fn reward(&self) -> f64 {
        self.reward.into_inner()
    }

    fn calc_reward(solution: &[usize], problem: &Array2<f64>) -> Result<f64, InvalidSolutionError> {
        solution
            .iter()
            .enumerate()
            .map(|(t, &j)| {
                let r = problem[(j, t)];
                if r.is_finite() {
                    Ok(r)
                } else {
                    Err(InvalidSolutionError::InfiniteReward)
                }
            })
            .sum()
    }

    fn into_tuple(self) -> (Vec<usize>, Array2<f64>) {
        (self.solution, self.problem)
    }
}

struct Murtys {
    item_idxs: Vec<usize>,
    solution_holder: Vec<usize>,
}

impl Murtys {
    fn new(m: usize, n: usize) -> Self {
        Murtys {
            item_idxs: (0..m).collect(),
            solution_holder: vec![0; n],
        }
    }

    fn validate_solution(
        unvalidated_solution: &[Assignment],
        solution_holder: &mut [usize],
    ) -> Result<(), InvalidSolutionError> {
        for (unvalidated_assignement, assignment_slot) in
            unvalidated_solution.iter().zip(solution_holder.iter_mut())
        {
            *assignment_slot = match unvalidated_assignement {
                Assignment::Assigned(i) => *i,
                Assignment::Unassigned => return Err(InvalidSolutionError::UnassignedTarget),
            };
        }
        Ok(())
    }

    fn run(
        mut self,
        original_problem: Array2<f64>,
        N: usize,
    ) -> Result<Vec<ProblemSolutionPair>, InvalidSolutionError> {
        let (m, n) = original_problem.dim();

        let original_solution = auction(
            &original_problem,
            auction_params::EPS,
            auction_params::MAX_ITERATIONS,
        );
        let mut solution_holder = vec![0; n];

        Self::validate_solution(&original_solution, &mut solution_holder)?;

        let problem_solution_pair =
            ProblemSolutionPair::new(solution_holder.clone(), original_problem.clone())?;

        let mut L = BinaryHeap::new();
        L.push(problem_solution_pair);

        let mut best_assignments = Vec::with_capacity(N);
        let mut locked_targets: Vec<usize> = Vec::with_capacity(n);
        let mut item_idxs: Vec<_> = Vec::with_capacity(m);

        while let Some(pair) = L.pop() {
            best_assignments.push(pair.clone());

            // Clear lists for new iteration
            locked_targets.clear();
            item_idxs.clear();
            for k in 0..m {
                item_idxs.push(k)
            }

            let (solution, problem) = pair.into_tuple();

            if best_assignments.len() == N {
                break;
            }

            let org_prob = problem.clone(); // Store for later
            let mut i = solution[0];

            let mut problem_duo = ProblemDuo::new(problem);

            for t in 0..n {
                problem_duo.make_association_impossible(i, 0);
                let sub_solution = auction(
                    &problem_duo.curr_view(),
                    auction_params::EPS,
                    auction_params::MAX_ITERATIONS,
                );
                if let Ok(()) = Self::validate_solution(&sub_solution, &mut solution_holder) {
                    let sol = get_indexed_vec(&item_idxs, &solution_holder);
                    let Qs = locked_targets.iter().chain(sol.iter()).copied().collect();
                    let Qp = org_prob.clone();

                    let org_i = item_idxs[i];
                    Qp[(org_i, t)];

                    if let Ok(pair) = ProblemSolutionPair::new(Qs, Qp) {
                        // Here we should make sure it does not exist already?
                        if L.iter().all(|p| p != &pair) {
                            L.push(pair);
                        }
                    }
                }

                locked_targets.push(item_idxs[i]);
                item_idxs.remove(i);
                if problem_duo.curr_view().shape()[1] > 1 {
                    problem_duo.delete_assignment(i);
                } else {
                    break;
                }

                i = item_idxs
                    .iter()
                    .position(|&i| i == solution[t + 1])
                    .expect("Cant find index??");
            }
        }

        Ok(best_assignments)
    }
}

fn validate_solution(
    unvalidated_solution: &[Assignment],
    solution_holder: &mut [usize],
) -> Result<(), InvalidSolutionError> {
    for (unvalidated_assignement, assignment_slot) in
        unvalidated_solution.iter().zip(solution_holder.iter_mut())
    {
        *assignment_slot = match unvalidated_assignement {
            Assignment::Assigned(i) => *i,
            Assignment::Unassigned => return Err(InvalidSolutionError::UnassignedTarget),
        };
    }
    Ok(())
}

fn murtys(
    original_problem: Array2<f64>,
    N: usize,
) -> Result<Vec<ProblemSolutionPair>, InvalidSolutionError> {
    let (m, n) = original_problem.dim();

    let original_solution = auction(
        &original_problem,
        auction_params::EPS,
        auction_params::MAX_ITERATIONS,
    );
    let mut solution_holder = vec![0; n];
    validate_solution(&original_solution, &mut solution_holder)?;

    let problem_solution_pair =
        ProblemSolutionPair::new(solution_holder.clone(), original_problem.clone())?;

    let mut L = BinaryHeap::new();
    L.push(problem_solution_pair);

    let mut best_assignments = Vec::with_capacity(N);
    let mut locked_targets: Vec<usize> = Vec::with_capacity(n);
    let mut item_idxs: Vec<_> = Vec::with_capacity(m);

    while let Some(pair) = L.pop() {
        best_assignments.push(pair.clone());

        // Clear lists for new iteration
        locked_targets.clear();
        item_idxs.clear();
        for k in 0..m {
            item_idxs.push(k)
        }

        let (solution, problem) = pair.into_tuple();

        if best_assignments.len() == N {
            break;
        }

        let org_prob = problem.clone(); // Store for later
        let mut i = solution[0];
        let mut problem_duo = ProblemDuo::new(problem);

        for t in 0..n {
            problem_duo.make_association_impossible(i, 0);
            let sub_solution = auction(
                &problem_duo.curr_view(),
                auction_params::EPS,
                auction_params::MAX_ITERATIONS,
            );
            if let Ok(()) = validate_solution(&sub_solution, &mut solution_holder) {
                let sol = get_indexed_vec(&item_idxs, &solution_holder);
                let Qs = locked_targets.iter().chain(sol.iter()).copied().collect();
                let Qp = org_prob.clone();

                let org_i = item_idxs[i];
                Qp[(org_i, t)];

                if let Ok(pair) = ProblemSolutionPair::new(Qs, Qp) {
                    // Here we should make sure it does not exist already?
                    if L.iter().all(|p| p != &pair) {
                        L.push(pair);
                    }
                }
            }

            locked_targets.push(item_idxs[i]);
            item_idxs.remove(i);
            if problem_duo.curr_view().shape()[1] > 1 {
                problem_duo.delete_assignment(i);
            } else {
                break;
            }

            i = item_idxs
                .iter()
                .position(|&i| i == solution[t + 1])
                .expect("Cant find index??");
        }
    }

    Ok(best_assignments)
}

// pub fn murtys(original_problem: Problem, N: usize) -> Result<Vec<ProblemSolutionPair>, InvalidSolutionError> {
//     let (m, n) = original_problem.num_measurements_targets();
//     let original_solution = auction(&original_problem, auction_params::EPS, auction_params::MAX_ITERATIONS)?;
//     let problem_solution_pair = problem_solution_pair::ProblemSolutionPair::new(original_solution, original_problem.clone())?; // TODO: Fix this later
//     let mut L = BinaryHeap::new();
//     L.push(problem_solution_pair);

//     let mut R = Vec::new();

//     let mut locked_targets: Vec<usize> = Vec::new();
//     let mut item_idxs: Vec<_> = (0..m).collect();

//     while let Some(problem_solution_pair) = L.pop() {
//         // TODO: Be smarter here
//         R.push(problem_solution_pair.clone());

//         if R.len() == N {
//             break;
//         }
//         let P = problem_solution_pair.problem();
//         let org_prob = P.clone();
//         let mut i = problem_solution_pair.solution().first();

//         for t in 0..n {
//             P.make_association_impossible(i, 0);
//             if let Ok(solution) = auction(&P, auction_params::EPS, auction_params::MAX_ITERATIONS) {
//                 let v = solution.into_vec();
//                 let convert_v = get_indexed_vec(item_idxs.as_slice(), v.as_slice());
//                 let Qs = Solution::concatenate_assignments(&[
//                 &locked_targets,
//                 &convert_v
//                 ]);

//                 let Qp = org_prob.clone();

//                 let org_i = item_idxs[i];
//                 Qp.make_association_impossible(org_i, t);

//                 if let Ok(pair) = problem_solution_pair::ProblemSolutionPair::new(Qs, Qp) {
//                     // Here we should make sure it does not exist already?
//                     if L.iter().all(|p| p != &pair) {
//                         L.push(pair);
//                     }
//                 }
//             }

//             locked_targets.push(item_idxs[i]);
//             item_idxs.remove(i);
//             if P.num_targets() > 1{
//                 P.delete_assignment(i);
//             } else {
//                 break;
//             }
//             // if P.is_empty() {
//             //     break;
//             // }
//             i = item_idxs.iter().position(|&i| i == problem_solution_pair.solution().assigned_measurement(t+1).unwrap()).expect("Cant find index??");
//         }
//     }

//     Ok(R)
// }

fn get_indexed_vec<T: Copy>(v: &[T], idx: &[usize]) -> Vec<T> {
    let mut index_vec = Vec::with_capacity(idx.len());
    for &i in idx {
        index_vec.push(v[i]);
    }
    index_vec
}
