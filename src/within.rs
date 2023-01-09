use std::cmp::Ordering;

pub fn kd_within_by_cmp<T>(
    kdtree: &[T],
    dim: usize,
    compare: impl Fn(&T, usize) -> Ordering + Copy,
) -> Vec<&T> {
    fn recurse<'a, T>(
        results: &mut Vec<&'a T>,
        kdtree: &'a [T],
        axis: usize,
        dim: usize,
        compare: impl Fn(&T, usize) -> Ordering + Copy,
    ) {
        if kdtree.is_empty() {
            return;
        }
        let axis = axis % dim;
        let (lower, item, upper) = {
            let mid = kdtree.len() / 2;
            (&kdtree[..mid], &kdtree[mid], &kdtree[mid + 1..])
        };
        match compare(item, axis) {
            Ordering::Equal => {
                if (1..dim).all(|k| compare(item, (axis + k) % dim) == Ordering::Equal) {
                    results.push(item);
                }
                recurse(results, lower, axis + 1, dim, compare);
                recurse(results, upper, axis + 1, dim, compare);
            }
            Ordering::Less => {
                recurse(results, upper, axis + 1, dim, compare);
            }
            Ordering::Greater => {
                recurse(results, lower, axis + 1, dim, compare);
            }
        }
    }
    let mut results = Vec::new();
    recurse(&mut results, kdtree, 0, dim, compare);
    results
}
