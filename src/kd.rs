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

pub fn kd_sort_points_by_key<P: Point, Key: Ord>(
    points: &mut [P],
    key: impl Fn(P::Scalar) -> Key + Copy,
) {
    kd_sort_points_by(points, |a, b| key(a).cmp(&key(b)))
}

pub fn kd_sort_points_by<P: Point>(
    points: &mut [P],
    compare: impl Fn(P::Scalar, P::Scalar) -> Ordering + Copy,
) {
    //kd_sort_by(points, P::DIM, |p, k| p.at(k), compare)
    fn recurse<P: Point>(
        items: &mut [P],
        axis: usize,
        compare: impl Fn(P::Scalar, P::Scalar) -> Ordering + Copy,
    ) {
        if items.len() >= 2 {
            pdqselect::select_by(items, items.len() / 2, |x, y| {
                compare(x.at(axis), y.at(axis))
            });
            let mid = items.len() / 2;
            let axis = (axis + 1) % P::DIM;
            recurse(&mut items[..mid], axis, compare);
            recurse(&mut items[mid + 1..], axis, compare);
        }
    }
    recurse(points, 0, compare);
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

pub fn kd_sort_by2<T>(
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
            pdqselect::select_by(items, items.len() / 2, |x, y| kd_compare(x, y, axis));
            let mid = items.len() / 2;
            let axis = (axis + 1) % dim;
            recurse(&mut items[..mid], axis, dim, kd_compare);
            recurse(&mut items[mid + 1..], axis, dim, kd_compare);
        }
    }
    recurse(items, 0, dim, kd_compare);
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

pub struct KdTree<'a, T, P: Point, F> {
    pub source: &'a [T],
    pub get: F,
    point_type: std::marker::PhantomData<P>,
}

impl<'a, T, P: Point, F> KdTree<'a, T, P, F>
where
    F: Fn(&T, usize) -> P::Scalar + Copy,
{
    fn new(source: &'a [T], get: F) -> Self {
        Self {
            source,
            get,
            point_type: std::marker::PhantomData,
        }
    }
    pub fn sort_by(
        source: &'a mut [T],
        get: F,
        compare: impl Fn(P::Scalar, P::Scalar) -> Ordering + Copy,
    ) -> Self {
        kd_sort_by(source, P::DIM, get, compare);
        Self::new(source, get)
    }
    pub fn sort_by_key<Key: Ord>(
        source: &'a mut [T],
        get: F,
        key: impl Fn(P::Scalar) -> Key + Copy,
    ) -> Self {
        kd_sort_by(source, P::DIM, get, |a, b| key(a).cmp(&key(b)));
        Self::new(source, get)
    }
    pub fn nearest<Q: Point<Dim = P::Dim, Scalar = P::Scalar>>(
        &self,
        query: &Q,
    ) -> Nearest<'a, T, P::Scalar> {
        kd_find_nearest(self.source, self.get, query)
    }
}

#[derive(Debug)]
pub struct Nearest<'a, T, Scalar> {
    pub item: &'a T,
    pub distance: Scalar,
}

impl<'a, T, Scalar> Nearest<'a, T, Scalar>
where
    Scalar: num_traits::NumAssign + Copy + PartialOrd,
{
    fn search<Q: Point<Scalar = Scalar>>(
        &mut self,
        sorted: &'a [T],
        get: impl Fn(&T, usize) -> Scalar + Copy,
        query: &Q,
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
        self.search(branch1, get, query, (axis + 1) % Q::DIM);
        let diff = query.at(axis) - mid_pos;
        if diff * diff < self.distance {
            self.search(branch2, get, query, (axis + 1) % Q::DIM);
        }
    }
    fn search2(
        &mut self,
        sorted: &'a [T],
        get: impl Fn(&T, usize) -> Scalar + Copy,
        query: &[Scalar],
        axis: usize,
    ) {
        if sorted.is_empty() {
            return;
        }
        let mid_idx = sorted.len() / 2;
        let distance = Self::distance_squared(query, &sorted[mid_idx], get);
        if distance < self.distance {
            *self = Nearest {
                item: &sorted[mid_idx],
                distance,
            };
            if self.distance.is_zero() {
                return;
            }
        }
        let mid_pos = get(&sorted[mid_idx], axis);
        let [branch1, branch2] = if query[axis] < mid_pos {
            [&sorted[..mid_idx], &sorted[mid_idx + 1..]]
        } else {
            [&sorted[mid_idx + 1..], &sorted[..mid_idx]]
        };
        self.search2(branch1, get, query, (axis + 1) % query.len());
        let diff = query[axis] - mid_pos;
        if diff * diff < self.distance {
            self.search2(branch2, get, query, (axis + 1) % query.len());
        }
    }
    fn distance_squared(p1: &[Scalar], p2: &T, get: impl Fn(&T, usize) -> Scalar) -> Scalar {
        let mut distance = <Scalar as num_traits::Zero>::zero();
        //for i in 0..dim {
        for (i, a) in p1.iter().enumerate() {
            let diff = *a - get(p2, i);
            distance += diff * diff;
        }
        distance
    }
    fn search3<Q: Point<Scalar = Scalar, Dim = T::Dim>>(
        &mut self,
        sorted: &'a [T],
        query: &Q,
        axis: usize,
    ) where
        T: Point<Scalar = Scalar>,
    {
        if sorted.is_empty() {
            return;
        }
        let mid_idx = sorted.len() / 2;
        let distance = distance_squared(query, &sorted[mid_idx], |p, k| p.at(k));
        if distance < self.distance {
            *self = Nearest {
                item: &sorted[mid_idx],
                distance,
            };
            if self.distance.is_zero() {
                return;
            }
        }
        let mid_pos = sorted[mid_idx].at(axis);
        let [branch1, branch2] = if query.at(axis) < mid_pos {
            [&sorted[..mid_idx], &sorted[mid_idx + 1..]]
        } else {
            [&sorted[mid_idx + 1..], &sorted[..mid_idx]]
        };
        self.search3(branch1, query, (axis + 1) % Q::DIM);
        let diff = query.at(axis) - mid_pos;
        if diff * diff < self.distance {
            self.search3(branch2, query, (axis + 1) % Q::DIM);
        }
    }
    fn search4(
        &mut self,
        sorted: &'a [T],
        axis: usize,
        dim: usize,
        get: impl Fn(&T, usize) -> Scalar + Copy,
        get_distance: impl Fn(&T) -> Scalar + Copy,
        get_component: impl Fn(usize) -> Scalar + Copy,
    ) {
        if sorted.is_empty() {
            return;
        }
        let mid_idx = sorted.len() / 2;
        let distance = get_distance(&sorted[mid_idx]);
        if distance < self.distance {
            *self = Nearest {
                item: &sorted[mid_idx],
                distance,
            };
            if self.distance.is_zero() {
                return;
            }
        }
        let mid_pos = get(&sorted[mid_idx], axis);
        let [branch1, branch2] = if get_component(axis) < mid_pos {
            [&sorted[..mid_idx], &sorted[mid_idx + 1..]]
        } else {
            [&sorted[mid_idx + 1..], &sorted[..mid_idx]]
        };
        self.search4(
            branch1,
            (axis + 1) % dim,
            dim,
            get,
            get_distance,
            get_component,
        );
        let diff = get_component(axis) - mid_pos;
        if diff * diff < self.distance {
            self.search4(
                branch2,
                (axis + 1) % dim,
                dim,
                get,
                get_distance,
                get_component,
            );
        }
    }
}

pub fn kd_find_nearest4<'a, T, Scalar>(
    sorted: &'a [T],
    dim: usize,
    get: impl Fn(&T, usize) -> Scalar + Copy,
    get_component: impl Fn(usize) -> Scalar + Copy,
    get_distance: impl Fn(&T) -> Scalar + Copy,
) -> Nearest<'a, T, Scalar>
where
    Scalar: num_traits::NumAssign + Copy + PartialOrd,
{
    assert!(!sorted.is_empty());
    let mut nearest = Nearest {
        item: &sorted[0],
        distance: get_distance(&sorted[0]),
    };
    nearest.search4(sorted, 0, dim, get, get_distance, get_component);
    nearest
}

pub fn kd_find_nearest3<'a, T: Point>(
    sorted: &'a [T],
    query: &impl Point<Scalar = T::Scalar, Dim = T::Dim>,
) -> Nearest<'a, T, T::Scalar> {
    assert!(!sorted.is_empty());
    let mut nearest = Nearest {
        item: &sorted[0],
        distance: distance_squared(query, &sorted[0], |p, k| p.at(k)),
    };
    nearest.search3(sorted, query, 0);
    nearest
}

pub fn kd_find_nearest2<'a, T, Scalar>(
    sorted: &'a [T],
    get: impl Fn(&T, usize) -> Scalar + Copy,
    query: &[Scalar],
) -> Nearest<'a, T, Scalar>
where
    Scalar: num_traits::NumAssign + Copy + PartialOrd,
{
    assert!(!sorted.is_empty());
    let mut nearest = Nearest {
        item: &sorted[0],
        distance: Nearest::distance_squared(query, &sorted[0], get),
    };
    nearest.search2(sorted, get, query, 0);
    //kd_search_nearest(sorted, get, query, 0, &mut nearest);
    nearest
}

pub fn kd_find_nearest<'a, T, P: Point>(
    sorted: &'a [T],
    get: impl Fn(&T, usize) -> P::Scalar + Copy,
    query: &P,
) -> Nearest<'a, T, P::Scalar> {
    assert!(!sorted.is_empty());
    let mut nearest = Nearest {
        item: &sorted[0],
        distance: distance_squared(query, &sorted[0], get),
    };
    nearest.search(sorted, get, query, 0);
    //kd_search_nearest(sorted, get, query, 0, &mut nearest);
    nearest
}

fn kd_search_nearest<'a, T, P: Point>(
    sorted: &'a [T],
    get: impl Fn(&T, usize) -> P::Scalar + Copy,
    query: &P,
    axis: usize,
    nearest: &mut Nearest<'a, T, P::Scalar>,
) {
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
    let [branch1, branch2] = if query.at(axis) < mid_pos {
        [&sorted[..mid_idx], &sorted[mid_idx + 1..]]
    } else {
        [&sorted[mid_idx + 1..], &sorted[..mid_idx]]
    };
    kd_search_nearest(branch1, get, query, (axis + 1) % P::DIM, nearest);
    let diff = query.at(axis) - mid_pos;
    if diff * diff < nearest.distance {
        kd_search_nearest(branch2, get, query, (axis + 1) % P::DIM, nearest);
    }
}

/*
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
