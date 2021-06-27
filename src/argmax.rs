pub fn argmax_iter<T, I>(i: I) -> (usize, T)
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

pub fn argmax<T: PartialOrd + Copy>(v: &[T]) -> (usize, T) {
    v.iter()
        .enumerate()
        .fold((0, v[0]), |(idx_max, val_max), (idx, val)| {
            if &val_max < val {
                (idx, *val)
            } else {
                (idx_max, val_max)
            }
        })
}