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

pub struct KdTree<'a, T, N, F> {
    source: &'a [T],
    get: F,
    dim: std::marker::PhantomData<N>,
}
pub type KdTree2<'a, T, F> = KdTree<'a, T, typenum::U2, F>;
pub type KdTree3<'a, T, F> = KdTree<'a, T, typenum::U3, F>;
impl<'a, T, N, F> std::ops::Deref for KdTree<'a, T, N, F> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        self.source
    }
}
impl<'a, T, N, F, Scalar> KdTree<'a, T, N, F>
where
    N: typenum::Unsigned + typenum::NonZero,
    F: Fn(&T, usize) -> Scalar + Copy,
    Scalar: Copy + PartialOrd + num_traits::NumAssign,
{
    fn new(source: &'a [T], get: F) -> Self {
        Self {
            source,
            get,
            dim: std::marker::PhantomData,
        }
    }
    pub fn sort(source: &'a mut [T], get: F) -> Self
    where
        Scalar: Ord,
    {
        kd_sort_by_key(source, N::to_usize(), get);
        Self::new(source, get)
    }
    pub fn sort_by(
        source: &'a mut [T],
        get: F,
        compare: impl Fn(Scalar, Scalar) -> Ordering + Copy,
    ) -> Self {
        kd_sort_by(source, N::to_usize(), |item1, item2, k| {
            compare(get(item1, k), get(item2, k))
        });
        Self::new(source, get)
    }
    pub fn sort_by_key<Key: Ord>(
        source: &'a mut [T],
        get: F,
        key: impl Fn(Scalar) -> Key + Copy,
    ) -> Self {
        kd_sort_by_key(source, N::to_usize(), |item, k| key(get(item, k)));
        Self::new(source, get)
    }
    pub fn nearest<Q: Point<Dim = N, Scalar = Scalar>>(&self, query: &Q) -> Nearest<'a, T, Scalar> {
        kd_nearest_by(self.source, query, self.get)
    }
}

#[derive(Debug)]
pub struct Nearest<'a, T, Scalar> {
    pub item: &'a T,
    pub distance: Scalar,
}

pub fn kd_nearest_with<T, Scalar>(
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

pub fn kd_nearest<'a, T: Point>(
    kdtree: &'a [T],
    query: &impl Point<Scalar = T::Scalar, Dim = T::Dim>,
) -> Nearest<'a, T, T::Scalar> {
    kd_nearest_by(kdtree, query, |item, k| item.at(k))
}

pub fn kd_nearest_by<'a, T, P: Point>(
    kdtree: &'a [T],
    query: &P,
    get: impl Fn(&T, usize) -> P::Scalar + Copy,
) -> Nearest<'a, T, P::Scalar> {
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
    fn recurse<'a, T, Q: Point>(
        nearest: &mut Nearest<'a, T, Q::Scalar>,
        kdtree: &'a [T],
        get: impl Fn(&T, usize) -> Q::Scalar + Copy,
        query: &Q,
        axis: usize,
    ) {
        if kdtree.is_empty() {
            return;
        }
        let mid_idx = kdtree.len() / 2;
        let item = &kdtree[mid_idx];
        let distance = distance_squared(query, item, get);
        if distance < nearest.distance {
            nearest.item = item;
            nearest.distance = distance;
            use num_traits::Zero;
            if nearest.distance.is_zero() {
                return;
            }
        }
        let mid_pos = get(item, axis);
        let [branch1, branch2] = if query.at(axis) < mid_pos {
            [&kdtree[..mid_idx], &kdtree[mid_idx + 1..]]
        } else {
            [&kdtree[mid_idx + 1..], &kdtree[..mid_idx]]
        };
        recurse(nearest, branch1, get, query, (axis + 1) % Q::DIM);
        let diff = query.at(axis) - mid_pos;
        if diff * diff < nearest.distance {
            recurse(nearest, branch2, get, query, (axis + 1) % Q::DIM);
        }
    }
    assert!(!kdtree.is_empty());
    let mut nearest = Nearest {
        item: &kdtree[0],
        distance: distance_squared(query, &kdtree[0], get),
    };
    recurse(&mut nearest, kdtree, get, query, 0);
    nearest
}
