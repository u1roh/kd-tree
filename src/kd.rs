use crate::array::Array;

pub fn kd_sort_by<T, A: Ord>(items: &mut [T], dim: usize, get: impl Fn(&T, usize) -> A + Copy) {
    fn recurse<T, A: Ord>(
        items: &mut [T],
        dim: usize,
        axis: usize,
        get: impl Fn(&T, usize) -> A + Copy,
    ) {
        if items.len() >= 2 {
            pdqselect::select_by(items, items.len() / 2, |x, y| {
                get(x, axis).cmp(&get(y, axis))
            });
            let mid = items.len() / 2;
            let axis = (axis + 1) % dim;
            recurse(&mut items[..mid], dim, axis, get);
            recurse(&mut items[mid + 1..], dim, axis, get);
        }
    }
    recurse(items, dim, 0, get);
}

fn distance_squared<P: Array, T>(
    p1: &P,
    p2: &T,
    get: impl Fn(&T, usize) -> P::Element,
) -> P::Element
where
    P::Element: num_traits::NumAssign + Copy,
{
    let mut distance = <P::Element as num_traits::Zero>::zero();
    for i in 0..P::LEN {
        let diff = *p1.at(i) - get(p2, i);
        distance += diff * diff;
    }
    distance
}

pub fn kd_find_nearest<'a, T, P: Array>(
    sorted: &'a [T],
    get: impl Fn(&T, usize) -> P::Element + Copy,
    query: &P,
) -> (&'a T, P::Element)
where
    P::Element: num_traits::NumAssign + Copy + PartialOrd + std::fmt::Display,
{
    assert!(!sorted.is_empty());
    let mut best = &sorted[0];
    let mut nearest_distance = distance_squared(query, best, get);
    kd_search_nearest(sorted, get, query, 0, &mut best, &mut nearest_distance);
    (best, nearest_distance)
}

fn kd_search_nearest<'a, T, P: Array>(
    sorted: &'a [T],
    get: impl Fn(&T, usize) -> P::Element + Copy,
    query: &P,
    axis: usize,
    best: &mut &'a T,
    nearest_distance: &mut P::Element,
) where
    P::Element: num_traits::NumAssign + Copy + PartialOrd,
{
    if sorted.is_empty() {
        return;
    }
    let mid_idx = sorted.len() / 2;
    let distance = distance_squared(query, &sorted[mid_idx], get);
    if distance < *nearest_distance {
        *nearest_distance = distance;
        *best = &sorted[mid_idx];
        use num_traits::Zero;
        if nearest_distance.is_zero() {
            return;
        }
    }
    let mid_pos = get(&sorted[mid_idx], axis);
    let [branch1, branch2] = if *query.at(axis) < mid_pos {
        [&sorted[..mid_idx], &sorted[mid_idx + 1..]]
    } else {
        [&sorted[mid_idx + 1..], &sorted[..mid_idx]]
    };
    kd_search_nearest(
        branch1,
        get,
        query,
        (axis + 1) % P::LEN,
        best,
        nearest_distance,
    );
    let diff = *query.at(axis) - mid_pos;
    if diff * diff < *nearest_distance {
        kd_search_nearest(
            branch2,
            get,
            query,
            (axis + 1) % P::LEN,
            best,
            nearest_distance,
        );
    }
}
