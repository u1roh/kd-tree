use crate::Point;
use std::cmp::Ordering;

pub fn kd_sort_indexables<Scalar: Copy + Ord>(
    items: &mut [impl std::ops::Index<usize, Output = Scalar>],
    dim: usize,
) {
    kd_sort_indexables_by(items, dim, |a, b| a.cmp(&b))
}

pub fn kd_sort_indexables_by<Scalar: Copy>(
    items: &mut [impl std::ops::Index<usize, Output = Scalar>],
    dim: usize,
    compare: impl Fn(Scalar, Scalar) -> std::cmp::Ordering + Copy,
) {
    kd_sort_by(items, dim, |p, k| p[k], compare)
}

pub fn kd_sort_points<P: Point>(points: &mut [P])
where
    P::Scalar: Ord,
{
    kd_sort_points_by(points, |a, b| a.cmp(&b))
}

pub fn kd_sort_points_by<P: Point>(
    points: &mut [P],
    compare: impl Fn(P::Scalar, P::Scalar) -> std::cmp::Ordering + Copy,
) {
    kd_sort_by(points, P::DIM, |p, k| p.at(k), compare)
}

pub fn kd_sort<T, A: Ord>(items: &mut [T], dim: usize, get: impl Fn(&T, usize) -> A + Copy) {
    kd_sort_by(items, dim, get, |a, b| a.cmp(&b))
}

pub fn kd_sort_by<T, A>(
    items: &mut [T],
    dim: usize,
    get: impl Fn(&T, usize) -> A + Copy,
    compare: impl Fn(A, A) -> std::cmp::Ordering + Copy,
) {
    fn recurse<T, A>(
        items: &mut [T],
        dim: usize,
        axis: usize,
        get: impl Fn(&T, usize) -> A + Copy,
        compare: impl Fn(A, A) -> std::cmp::Ordering + Copy,
    ) {
        if items.len() >= 2 {
            pdqselect::select_by(items, items.len() / 2, |x, y| {
                compare(get(x, axis), get(y, axis))
            });
            let mid = items.len() / 2;
            let axis = (axis + 1) % dim;
            recurse(&mut items[..mid], dim, axis, get, compare);
            recurse(&mut items[mid + 1..], dim, axis, get, compare);
        }
    }
    recurse(items, dim, 0, get, compare);
}

fn distance_squared<P: Point, T>(
    p1: &P,
    p2: &T,
    get: impl Fn(&T, usize) -> P::Scalar,
) -> P::Scalar {
    let mut distance = <P::Scalar as num_traits::Zero>::zero();
    for i in 0..P::DIM {
        let diff = p1.at(i) - get(p2, i);
        distance += diff * diff;
    }
    distance
}

pub struct KdTree<'a, T, F, N> {
    pub source: &'a [T],
    pub get: F,
    dim: std::marker::PhantomData<N>,
}

impl<'a, T, F, N, A: Copy> KdTree<'a, T, F, N>
where
    N: typenum::Unsigned + typenum::NonZero,
    F: Fn(&T, usize) -> A + Copy,
{
    pub fn new(source: &'a mut [T], get: F, compare: impl Fn(A, A) -> Ordering + Copy) -> Self {
        kd_sort_by(source, N::to_usize(), get, compare);
        Self {
            source,
            get,
            dim: std::marker::PhantomData,
        }
    }
    pub fn nearest<Q: Point<Dim = N, Scalar = A>>(&self, query: &Q) -> Nearest<'a, T, Q>
    where
        A: num_traits::NumAssign + PartialOrd,
    {
        kd_find_nearest(self.source, self.get, query)
    }
}

#[derive(Debug)]
pub struct Nearest<'a, T, P: Point> {
    pub item: &'a T,
    pub distance: P::Scalar,
}

impl<'a, T, P: Point> Nearest<'a, T, P> {
    fn search(
        &mut self,
        sorted: &'a [T],
        get: impl Fn(&T, usize) -> P::Scalar + Copy,
        query: &P,
        axis: usize,
    ) {
        if sorted.is_empty() {
            return;
        }
        let mid_idx = sorted.len() / 2;
        let distance = distance_squared(query, &sorted[mid_idx], get);
        if distance < self.distance {
            *self = Nearest {
                item: &sorted[mid_idx],
                distance,
            };
            use num_traits::Zero;
            if self.distance.is_zero() {
                return;
            }
        }
        let mid_pos = get(&sorted[mid_idx], axis);
        let [branch1, branch2] = if query.at(axis) < mid_pos {
            [&sorted[..mid_idx], &sorted[mid_idx + 1..]]
        } else {
            [&sorted[mid_idx + 1..], &sorted[..mid_idx]]
        };
        self.search(branch1, get, query, (axis + 1) % P::DIM);
        let diff = query.at(axis) - mid_pos;
        if diff * diff < self.distance {
            self.search(branch2, get, query, (axis + 1) % P::DIM);
        }
    }
}

pub fn kd_find_nearest<'a, T, P: Point>(
    sorted: &'a [T],
    get: impl Fn(&T, usize) -> P::Scalar + Copy,
    query: &P,
) -> Nearest<'a, T, P> {
    assert!(!sorted.is_empty());
    let mut nearest = Nearest {
        item: &sorted[0],
        distance: distance_squared(query, &sorted[0], get),
    };
    nearest.search(sorted, get, query, 0);
    //kd_search_nearest(sorted, get, query, 0, &mut nearest);
    nearest
}

/*
fn kd_search_nearest<'a, T, P: Array>(
    sorted: &'a [T],
    get: impl Fn(&T, usize) -> P::Element + Copy,
    query: &P,
    axis: usize,
    nearest: &mut Nearest<'a, T, P>,
) where
    P::Element: num_traits::NumAssign + Copy + PartialOrd,
{
    if sorted.is_empty() {
        return;
    }
    let mid_idx = sorted.len() / 2;
    let distance = distance_squared(query, &sorted[mid_idx], get);
    if distance < nearest.distance {
        *nearest = Nearest {
            item: &sorted[mid_idx],
            distance,
        };
        use num_traits::Zero;
        if nearest.distance.is_zero() {
            return;
        }
    }
    let mid_pos = get(&sorted[mid_idx], axis);
    let [branch1, branch2] = if *query.at(axis) < mid_pos {
        [&sorted[..mid_idx], &sorted[mid_idx + 1..]]
    } else {
        [&sorted[mid_idx + 1..], &sorted[..mid_idx]]
    };
    kd_search_nearest(branch1, get, query, (axis + 1) % P::LEN, nearest);
    let diff = *query.at(axis) - mid_pos;
    if diff * diff < nearest.distance {
        kd_search_nearest(branch2, get, query, (axis + 1) % P::LEN, nearest);
    }
}

fn distance_squared2<T, A>(
    dim: usize,
    p1: &(impl std::ops::Index<usize, Output = A> + ?Sized),
    p2: &T,
    get: impl Fn(&T, usize) -> A,
) -> A
where
    A: num_traits::NumAssign + Copy,
{
    let mut distance = <A as num_traits::Zero>::zero();
    for i in 0..dim {
        let diff = p1[i] - get(p2, i);
        distance += diff * diff;
    }
    distance
}

pub fn kd_find_nearest2<'a, T, Q, A>(
    sorted: &'a [T],
    get: impl Fn(&T, usize) -> A + Copy,
    query: &impl AsRef<Q>,
    dim: usize,
) -> (&'a T, A)
where
    Q: std::ops::Index<usize, Output = A> + ?Sized,
    A: num_traits::NumAssign + Copy + PartialOrd + std::fmt::Display,
{
    assert!(!sorted.is_empty());
    let mut best = &sorted[0];
    let mut nearest_distance = distance_squared2(dim, query.as_ref(), best, get);
    kd_search_nearest2(
        sorted,
        get,
        query.as_ref(),
        dim,
        0,
        &mut best,
        &mut nearest_distance,
    );
    (best, nearest_distance)
}

fn kd_search_nearest2<'a, T, A>(
    sorted: &'a [T],
    get: impl Fn(&T, usize) -> A + Copy,
    query: &(impl std::ops::Index<usize, Output = A> + ?Sized),
    dim: usize,
    axis: usize,
    best: &mut &'a T,
    nearest_distance: &mut A,
) where
    A: num_traits::NumAssign + Copy + PartialOrd,
{
    if sorted.is_empty() {
        return;
    }
    let mid_idx = sorted.len() / 2;
    let distance = distance_squared2(dim, query, &sorted[mid_idx], get);
    if distance < *nearest_distance {
        *nearest_distance = distance;
        *best = &sorted[mid_idx];
        if nearest_distance.is_zero() {
            return;
        }
    }
    let mid_pos = get(&sorted[mid_idx], axis);
    let [branch1, branch2] = if query[axis] < mid_pos {
        [&sorted[..mid_idx], &sorted[mid_idx + 1..]]
    } else {
        [&sorted[mid_idx + 1..], &sorted[..mid_idx]]
    };
    kd_search_nearest2(
        branch1,
        get,
        query,
        dim,
        (axis + 1) % dim,
        best,
        nearest_distance,
    );
    let diff = query[axis] - mid_pos;
    if diff * diff < *nearest_distance {
        kd_search_nearest2(
            branch2,
            get,
            query,
            dim,
            (axis + 1) % dim,
            best,
            nearest_distance,
        );
    }
}
*/
