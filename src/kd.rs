use crate::Point;
use std::cmp::Ordering;

pub fn kd_sort<P: Point>(points: &mut [P])
where
    P::Scalar: Ord,
{
    kd_sort_by_key(points, P::DIM, |item, k| item.at(k))
}

pub fn kd_sort_by_key<T, Key: Ord>(
    items: &mut [T],
    dim: usize,
    kd_key: impl Fn(&T, usize) -> Key + Copy,
) {
    kd_sort_by(items, dim, |item1, item2, k| {
        kd_key(item1, k).cmp(&kd_key(item2, k))
    })
}

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
        kd_sort_by(source, P::DIM, |item1, item2, k| {
            compare(get(item1, k), get(item2, k))
        });
        Self::new(source, get)
    }
    pub fn sort_by_key<Key: Ord>(
        source: &'a mut [T],
        get: F,
        key: impl Fn(P::Scalar) -> Key + Copy,
    ) -> Self {
        kd_sort_by_key(source, P::DIM, |item, k| key(get(item, k)));
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
        kdtree: &'a [T],
        get: impl Fn(&T, usize) -> Scalar + Copy,
        query: &Q,
        axis: usize,
    ) {
        if kdtree.is_empty() {
            return;
        }
        let mid_idx = kdtree.len() / 2;
        let item = &kdtree[mid_idx];
        let distance = distance_squared(query, item, get);
        if distance < self.distance {
            self.item = item;
            self.distance = distance;
            if self.distance.is_zero() {
                return;
            }
        }
        let mid_pos = get(item, axis);
        let [branch1, branch2] = if query.at(axis) < mid_pos {
            [&kdtree[..mid_idx], &kdtree[mid_idx + 1..]]
        } else {
            [&kdtree[mid_idx + 1..], &kdtree[..mid_idx]]
        };
        self.search(branch1, get, query, (axis + 1) % Q::DIM);
        let diff = query.at(axis) - mid_pos;
        if diff * diff < self.distance {
            self.search(branch2, get, query, (axis + 1) % Q::DIM);
        }
    }
    fn search2(
        &mut self,
        kdtree: &'a [T],
        get: impl Fn(&T, usize) -> Scalar + Copy,
        query: &[Scalar],
        axis: usize,
    ) {
        if kdtree.is_empty() {
            return;
        }
        let mid_idx = kdtree.len() / 2;
        let item = &kdtree[mid_idx];
        let distance = Self::distance_squared(query, item, get);
        if distance < self.distance {
            *self = Nearest { item, distance };
            if self.distance.is_zero() {
                return;
            }
        }
        let mid_pos = get(item, axis);
        let [branch1, branch2] = if query[axis] < mid_pos {
            [&kdtree[..mid_idx], &kdtree[mid_idx + 1..]]
        } else {
            [&kdtree[mid_idx + 1..], &kdtree[..mid_idx]]
        };
        self.search2(branch1, get, query, (axis + 1) % query.len());
        let diff = query[axis] - mid_pos;
        if diff * diff < self.distance {
            self.search2(branch2, get, query, (axis + 1) % query.len());
        }
    }
    fn distance_squared(p1: &[Scalar], p2: &T, get: impl Fn(&T, usize) -> Scalar) -> Scalar {
        let mut distance = <Scalar as num_traits::Zero>::zero();
        for (i, a) in p1.iter().enumerate() {
            let diff = *a - get(p2, i);
            distance += diff * diff;
        }
        distance
    }
}

pub fn kd_find_nearest_by<T, Scalar>(
    kdtree: &[T],
    dim: usize,
    kd_difference: impl Fn(&T, usize) -> Scalar + Copy,
) -> Nearest<T, Scalar>
where
    Scalar: num_traits::NumAssign + Copy + PartialOrd,
{
    fn distance<T, Scalar: num_traits::NumAssign + Copy>(
        item: &T,
        dim: usize,
        kd_difference: impl Fn(&T, usize) -> Scalar + Copy,
    ) -> Scalar {
        let mut distance = Scalar::zero();
        for k in 0..dim {
            let diff = kd_difference(item, k);
            distance += diff * diff;
        }
        distance
    }
    fn recurse<'a, T, Scalar>(
        nearest: &mut Nearest<'a, T, Scalar>,
        kdtree: &'a [T],
        axis: usize,
        dim: usize,
        kd_difference: impl Fn(&T, usize) -> Scalar + Copy,
    ) where
        Scalar: num_traits::NumAssign + Copy + PartialOrd,
    {
        if kdtree.is_empty() {
            return;
        }
        let mid_idx = kdtree.len() / 2;
        let mid = &kdtree[mid_idx];
        let distance = distance(mid, dim, kd_difference);
        if distance < nearest.distance {
            *nearest = Nearest {
                item: mid,
                distance,
            };
            if nearest.distance.is_zero() {
                return;
            }
        }
        let [branch1, branch2] = if kd_difference(mid, axis) < Scalar::zero() {
            [&kdtree[..mid_idx], &kdtree[mid_idx + 1..]]
        } else {
            [&kdtree[mid_idx + 1..], &kdtree[..mid_idx]]
        };
        recurse(nearest, branch1, (axis + 1) % dim, dim, kd_difference);
        let diff = kd_difference(mid, axis);
        if diff * diff < nearest.distance {
            recurse(nearest, branch2, (axis + 1) % dim, dim, kd_difference);
        }
    }
    assert!(!kdtree.is_empty());
    let mut nearest = Nearest {
        item: &kdtree[0],
        distance: distance(&kdtree[0], dim, kd_difference),
    };
    recurse(&mut nearest, kdtree, 0, dim, kd_difference);
    nearest
}

pub fn kd_find_nearest3<'a, T: Point>(
    kdtree: &'a [T],
    query: &impl Point<Scalar = T::Scalar, Dim = T::Dim>,
) -> Nearest<'a, T, T::Scalar> {
    kd_find_nearest(kdtree, |item, k| item.at(k), query)
}

pub fn kd_find_nearest2<'a, T, Scalar>(
    kdtree: &'a [T],
    get: impl Fn(&T, usize) -> Scalar + Copy,
    query: &[Scalar],
) -> Nearest<'a, T, Scalar>
where
    Scalar: num_traits::NumAssign + Copy + PartialOrd,
{
    //kd_find_nearest_by(kdtree, query.len(), |item, k| query[k] - get(item, k))
    assert!(!kdtree.is_empty());
    let mut nearest = Nearest {
        item: &kdtree[0],
        distance: Nearest::distance_squared(query, &kdtree[0], get),
    };
    nearest.search2(kdtree, get, query, 0);
    //kd_search_nearest(kdtree, get, query, 0, &mut nearest);
    nearest
}

pub fn kd_find_nearest<'a, T, P: Point>(
    kdtree: &'a [T],
    get: impl Fn(&T, usize) -> P::Scalar + Copy,
    query: &P,
) -> Nearest<'a, T, P::Scalar> {
    assert!(!kdtree.is_empty());
    let mut nearest = Nearest {
        item: &kdtree[0],
        distance: distance_squared(query, &kdtree[0], get),
    };
    nearest.search(kdtree, get, query, 0);
    //kd_search_nearest(kdtree, get, query, 0, &mut nearest);
    nearest
}

fn kd_search_nearest<'a, T, P: Point>(
    kdtree: &'a [T],
    get: impl Fn(&T, usize) -> P::Scalar + Copy,
    query: &P,
    axis: usize,
    nearest: &mut Nearest<'a, T, P::Scalar>,
) {
    if kdtree.is_empty() {
        return;
    }
    let mid_idx = kdtree.len() / 2;
    let distance = distance_squared(query, &kdtree[mid_idx], get);
    if distance < nearest.distance {
        *nearest = Nearest {
            item: &kdtree[mid_idx],
            distance,
        };
        use num_traits::Zero;
        if nearest.distance.is_zero() {
            return;
        }
    }
    let mid_pos = get(&kdtree[mid_idx], axis);
    let [branch1, branch2] = if query.at(axis) < mid_pos {
        [&kdtree[..mid_idx], &kdtree[mid_idx + 1..]]
    } else {
        [&kdtree[mid_idx + 1..], &kdtree[..mid_idx]]
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
    kdtree: &'a [T],
    get: impl Fn(&T, usize) -> A + Copy,
    query: &impl AsRef<Q>,
    dim: usize,
) -> (&'a T, A)
where
    Q: std::ops::Index<usize, Output = A> + ?Sized,
    A: num_traits::NumAssign + Copy + PartialOrd + std::fmt::Display,
{
    assert!(!kdtree.is_empty());
    let mut best = &kdtree[0];
    let mut nearest_distance = distance_squared2(dim, query.as_ref(), best, get);
    kd_search_nearest2(
        kdtree,
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
    kdtree: &'a [T],
    get: impl Fn(&T, usize) -> A + Copy,
    query: &(impl std::ops::Index<usize, Output = A> + ?Sized),
    dim: usize,
    axis: usize,
    best: &mut &'a T,
    nearest_distance: &mut A,
) where
    A: num_traits::NumAssign + Copy + PartialOrd,
{
    if kdtree.is_empty() {
        return;
    }
    let mid_idx = kdtree.len() / 2;
    let distance = distance_squared2(dim, query, &kdtree[mid_idx], get);
    if distance < *nearest_distance {
        *nearest_distance = distance;
        *best = &kdtree[mid_idx];
        if nearest_distance.is_zero() {
            return;
        }
    }
    let mid_pos = get(&kdtree[mid_idx], axis);
    let [branch1, branch2] = if query[axis] < mid_pos {
        [&kdtree[..mid_idx], &kdtree[mid_idx + 1..]]
    } else {
        [&kdtree[mid_idx + 1..], &kdtree[..mid_idx]]
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
