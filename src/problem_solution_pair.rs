use ndarray::prelude::*;
use ordered_float::OrderedFloat;
use std::cmp::Ordering;
use crate::data_association::Assignment;


#[derive(Debug, Clone)]
pub struct Solution(pub Vec<Assignment>);

#[derive(Debug, Clone)]
pub struct Problem(pub Array2<f64>);

#[derive(Debug, Clone)]
pub struct ProblemSolutionPair {
    solution: Solution,
    problem: Problem,
    reward: OrderedFloat<f64>,
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum InvalidSolutionError {
    InvalidReward(f64),
    UnassignedTarget,
}

impl std::fmt::Display for InvalidSolutionError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::InvalidReward(r) => write!(f, "Solution is invalid - calculated reward is {}", r),
            Self::UnassignedTarget => write!(f, "Solution is invalid - There are unassigned targets")
        }
    }
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
        Ok(
            ProblemSolutionPair{
                solution,
                problem,
                reward: OrderedFloat(reward)
            }
        )
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
        let Solution(assignments) = solution;
        let Problem(A) = problem;

        let mut reward = 0.0;
        for (t, j) in assignments.iter().enumerate() {
            let &j = match j {
                Assignment::Assigned(j) => j,
                Assignment::Unassigned => return Err(InvalidSolutionError::UnassignedTarget),
            };
            reward += A[(j, t)];
        }
        return Ok(reward)
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

        let A1 = array![
            [-5.69, 5.37, -inf],
            [-inf, -3.8, 6.58],
            [4.78, -inf, -inf],
            [-inf, 5.36, -inf],
            [-0.46, -inf, -inf],
            [-inf, -0.52, -inf],
            [-inf, -inf, -0.60]
        ];

        let solution1 = Solution(auction(&A1, EPS));
        let problem1 = Problem(A1);
        let pair1 = ProblemSolutionPair::new(solution1, problem1).unwrap();

        let A2 = array![
            [-5.69, -inf, -inf],
            [-inf, -3.8, 6.58],
            [4.78, -inf, -inf],
            [-inf, 5.36, -inf],
            [-0.46, -inf, -inf],
            [-inf, -0.52, -inf],
            [-inf, -inf, -0.60]
        ];
        
        let solution2 = Solution(auction(&A2, EPS));
        let problem2 = Problem(A2);
        let pair2 = ProblemSolutionPair::new(solution2, problem2).unwrap();

        let A3 = array![
            [-5.69, 5.37, -inf],
            [-inf, -3.8, 6.58],
            [-inf, -inf, -inf],
            [-inf, 5.36, -inf],
            [-0.46, -inf, -inf],
            [-inf, -0.52, -inf],
            [-inf, -inf, -0.60]
        ];

        let solution3 = Solution(auction(&A3, EPS));
        let problem3 = Problem(A3);
        let pair3 = ProblemSolutionPair::new(solution3, problem3).unwrap();

        let mut b = BinaryHeap::new();
        b.push(pair3);
        b.push(pair1);    
        b.push(pair2);

        while let Some(l) = b.pop() {
            println!("{:?}", l);
        }
     }
    
}