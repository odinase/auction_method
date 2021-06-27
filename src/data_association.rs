use ndarray::prelude::*;
use ordered_float::OrderedFloat;
use std::collections::VecDeque;

pub struct Solution<T>(pub Vec<T>);
pub struct Problem<T>(pub Array2<T>);

fn argmax<T, I>(i: I) -> (usize, T)
where
    T: PartialOrd + Copy,
    I: Iterator<Item = T>,
{
    let mut i = i.peekable();
    let f = *i.peek().unwrap();
    i.enumerate()
        .fold((0, f), |(idx_max, val_max), (idx, val)| {
            if val_max < val {
                (idx, val)
            } else {
                (idx_max, val_max)
            }
        })
}

pub fn auction(A: &Array2<f64>, eps: f64) -> Solution<i32> {
    let (m, n) = A.dim();
    let mut unassigned_queue: VecDeque<_> = (0..n).collect();
    let mut assigned_tracks: Vec<i32> = vec![-1; n];

    let mut prices = vec![0f64; m];
    // let mut values = vec![0; m - 1];

    while let Some(t_star) = unassigned_queue.pop_front() {
        let (i_star, _) = argmax(
            A.column(t_star)
                .into_iter()
                .zip(prices.iter())
                .map(|(a, &p)| a - p as f64),
        );
        let prev_owner = assigned_tracks.iter().position(|&e| e as usize == i_star);
        assigned_tracks[t_star] = i_star as i32;
        // The item has a previous owner
        if let Some(prev_owner) = prev_owner {
            assigned_tracks[prev_owner] = -1;
            unassigned_queue.push_back(prev_owner);
        }

        // Quick hack, should be done better
        let mut values: Vec<_> = A
            .column(t_star)
            .into_iter()
            .zip(prices.iter())
            .map(|(a, &p)| OrderedFloat(a - p as f64))
            .collect();
        values.remove(i_star);
        let val_max = values.iter().max().expect("For some reason values is empty???").into_inner();
        let y = A[(i_star, t_star)] - val_max;
        prices[i_star] += y + eps;
    }

    Solution(assigned_tracks)
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
        let Solution(assigned_tracks) = auction(&A, 1e-3);
        let stop = start.elapsed().as_secs_f64()*1e6;
        println!("ran in {:.2} us", stop);
        for (t, j) in assigned_tracks.iter().enumerate() {
            println!("a({}) = {}", t+1, j+1)
        }
    }
}