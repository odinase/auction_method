use crate::data_association::Assignment;
use ndarray::prelude::*;
use ordered_float::OrderedFloat;
use std::cell::RefCell;
use std::cmp::Ordering;

#[derive(Debug, Clone)]
pub struct Solution(Vec<usize>);

impl Solution {
    pub fn try_from_unvalidated_assignments(
        unvalidated_assignments: Vec<Assignment>,
    ) -> Result<Self, InvalidSolutionError> {
        let valid_assignments = unvalidated_assignments
            .iter()
            .map(|&e| match e {
                Assignment::Assigned(i) => Ok(i),
                Assignment::Unassigned => Err(InvalidSolutionError::UnassignedTarget),
            })
            .collect::<Result<Vec<usize>, InvalidSolutionError>>()?;
        Ok(Solution(valid_assignments))
    }

    pub fn from_assignments(assignments: Vec<usize>) -> Self {
        Solution(assignments)
    }

    pub fn concatenate_assignments(assignment_set: &[&Vec<usize>]) -> Self {
        let assignments = assignment_set.iter().copied().flatten().copied().collect();
        Solution(assignments)
    }

    pub fn into_vec(self) -> Vec<usize> {
        self.0
    }

    pub fn assigned_measurement(&self, target: usize) -> Option<usize> {
        if target < self.0.len() {
            Some(self.0[target])
        } else {
            None
        }
    }

    pub fn assigned_target(&self, measurement: usize) -> Option<usize> {
        self.0.iter().position(|&m| m == measurement)
    }

    pub fn iter(&self) -> std::slice::Iter<'_, usize> {
        self.0.iter()
    }

    pub fn assignments(&self) -> impl IntoIterator<Item = (usize, usize)> + '_ {
        self.iter().copied().enumerate()
    }

    pub fn first(&self) -> usize {
        self.0[0]
    }
}

impl IntoIterator for Solution {
    type Item = usize;
    type IntoIter = std::vec::IntoIter<Self::Item>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

#[derive(Debug, Clone)]
pub struct Problem(RefCell<Array2<f64>>);

impl Problem {
    pub fn new(A: Array2<f64>) -> Self {
        Problem(RefCell::new(A))
    }

    pub fn reward(&self, measurement: usize, target: usize) -> f64 {
        self.0.borrow()[(measurement, target)]
    }

    pub fn make_association_impossible(&self, measurement: usize, target: usize) {
        let mut A = self.0.borrow_mut();
        A[(measurement, target)] = std::f64::NEG_INFINITY;
    }

    pub fn rewards(&self, target: usize) -> impl IntoIterator<Item = &f64> + '_ {
        // TODO: Hacky?
        // Safety: We now that the data in self is living for exactly as long as self, so this is safe. Pointer is no doubt valid, aligned and all that
        unsafe { (*self.0.as_ptr()).column(target) }
    }

    pub fn num_measurements(&self) -> usize {
        self.0.borrow().shape()[0]
    }

    pub fn num_targets(&self) -> usize {
        self.0.borrow().shape()[1]
    }

    pub fn num_measurements_targets(&self) -> (usize, usize) {
        self.0.borrow().dim()
    }

    pub fn delete_assignment(&self, measurement: usize) {
        let aa = {
            let a = self.0.borrow();
            let (rows, cols) = a.dim();
            let ass = a.slice(s![.., 1..]); // Remove first column to delete target
            let i = ass
                .rows() // Iterate over remaining rows
                .into_iter()
                .enumerate()
                .filter(|&(k, _)| k != measurement) // Remove row with measurement
                .map(|(_, r)| r) // Convert back to iterating over rows
                .flat_map(|r| r.into_iter().copied()); // Squeeze all data into a long iterator
            Array::from_iter(i) // Build new matrix with row and column deleted
                .into_shape((rows - 1, cols - 1))
                .unwrap()
        };
        let r = RefCell::new(aa);
        self.0.swap(&r);
    }

    pub fn is_empty(&self) -> bool {
        self.0.borrow().is_empty()
    }
}

#[derive(Debug, Clone)]
pub struct ProblemSolutionPair {
    solution: Solution,
    problem: Problem,
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

impl std::error::Error for InvalidSolutionError {}

impl ProblemSolutionPair {
    pub fn new(solution: Solution, problem: Problem) -> Result<Self, InvalidSolutionError> {
        let reward = Self::calc_reward(&solution, &problem)?;
        Ok(ProblemSolutionPair {
            solution,
            problem,
            reward: OrderedFloat(reward),
        })
    }
    pub fn solution(&self) -> &Solution {
        &self.solution
    }
    pub fn problem(&self) -> &Problem {
        &self.problem
    }
    pub fn reward(&self) -> f64 {
        self.reward.into_inner()
    }
    fn calc_reward(solution: &Solution, problem: &Problem) -> Result<f64, InvalidSolutionError> {
        solution
            .iter()
            .enumerate()
            .map(|(t, &j)| {
                let r = problem.reward(j, t);
                if r.is_finite() {
                    Ok(r)
                } else {
                    Err(InvalidSolutionError::InfiniteReward)
                }
            })
            .sum()
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum InvalidSolutionError {
    InfiniteReward,
    UnassignedTarget,
}

impl std::fmt::Display for InvalidSolutionError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::InfiniteReward => {
                write!(f, "Solution is invalid - calculated reward is infinite")
            }
            Self::UnassignedTarget => {
                write!(f, "Solution is invalid - There are unassigned targets")
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::data_association::auction;
    use std::collections::BinaryHeap;
    #[test]
    fn test_binary_heap_problem_solution_pair() {
        use std::f64::INFINITY as inf;
        const EPS: f64 = 1e-3;
        const MAX_ITERATIONS: usize = 10_000;
        let A1 = array![
            [-5.69, 5.37, -inf],
            [-inf, -3.8, 6.58],
            [4.78, -inf, -inf],
            [-inf, 5.36, -inf],
            [-0.46, -inf, -inf],
            [-inf, -0.52, -inf],
            [-inf, -inf, -0.60]
        ];
        let problem1 = Problem::new(A1);
        let solution1 = auction(&problem1, EPS, MAX_ITERATIONS).unwrap();

        let pair1 = ProblemSolutionPair::new(solution1, problem1).unwrap();
        let pair1_ref = pair1.clone();

        let A2 = array![
            [-5.69, -inf, -inf],
            [-inf, -3.8, 6.58],
            [4.78, -inf, -inf],
            [-inf, 5.36, -inf],
            [-0.46, -inf, -inf],
            [-inf, -0.52, -inf],
            [-inf, -inf, -0.60]
        ];
        let problem2 = Problem::new(A2);
        let solution2 = auction(&problem2, EPS, MAX_ITERATIONS).unwrap();
        let pair2 = ProblemSolutionPair::new(solution2, problem2).unwrap();
        let pair2_ref = pair2.clone();

        let A3 = array![
            [-5.69, 5.37, -inf],
            [-inf, -3.8, 6.58],
            [-inf, -inf, -inf],
            [-inf, 5.36, -inf],
            [-0.46, -inf, -inf],
            [-inf, -0.52, -inf],
            [-inf, -inf, -0.60]
        ];
        let problem3 = Problem::new(A3);
        let solution3 = auction(&problem3, EPS, MAX_ITERATIONS).unwrap();

        let pair3 = ProblemSolutionPair::new(solution3, problem3).unwrap();
        let pair3_ref = pair3.clone();

        let mut b = BinaryHeap::new();
        b.push(pair3);
        b.push(pair1);
        b.push(pair2);

        assert_eq!(b.pop().unwrap(), pair1_ref);
        assert_eq!(b.pop().unwrap(), pair2_ref);
        assert_eq!(b.pop().unwrap(), pair3_ref);
    }
}
