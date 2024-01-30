use std::cmp::Ordering;

pub fn kd_sort_by<T>(
    items: &mut [T],
    dim: usize,
    kd_compare: impl Fn(&T, &T, usize) -> Ordering + Copy,
) {
    fn recurse<T>(
        items: &mut [T],
        axis: usize,
        dim: usize,
        kd_compare: impl Fn(&T, &T, usize) -> Ordering + Copy,
    ) {
        if items.len() >= 2 {
            items.select_nth_unstable_by(items.len() / 2, |x, y| kd_compare(x, y, axis));
            let mid = items.len() / 2;
            let axis = (axis + 1) % dim;
            recurse(&mut items[..mid], axis, dim, kd_compare);
            recurse(&mut items[mid + 1..], axis, dim, kd_compare);
        }
    }
    recurse(items, 0, dim, kd_compare);
}

#[cfg(feature = "rayon")]
pub fn kd_par_sort_by<T: Send>(
    items: &mut [T],
    dim: usize,
    kd_compare: impl Fn(&T, &T, usize) -> Ordering + Copy + Send,
) {
    fn recurse<T: Send>(
        items: &mut [T],
        axis: usize,
        dim: usize,
        kd_compare: impl Fn(&T, &T, usize) -> Ordering + Copy + Send,
    ) {
        if items.len() >= 2 {
            items.select_nth_unstable_by(items.len() / 2, |x, y| kd_compare(x, y, axis));
            let mid = items.len() / 2;
            let axis = (axis + 1) % dim;
            let (lhs, rhs) = items.split_at_mut(mid);
            rayon::join(
                move || recurse(lhs, axis, dim, kd_compare),
                move || recurse(&mut rhs[1..], axis, dim, kd_compare),
            );
        }
    }
    recurse(items, 0, dim, kd_compare);
}
