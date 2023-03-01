use crate::{ItemAndDistance, KdPoint};

pub fn kd_nearest<'a, T: KdPoint<N>, const N: usize>(
    kdtree: &'a [T],
    query: &impl KdPoint<N, Scalar = T::Scalar>,
) -> ItemAndDistance<'a, T, T::Scalar> {
    kd_nearest_by(kdtree, query, |item, k| item.at(k))
}

pub fn kd_nearest_by<'a, T, P: KdPoint<N>, const N: usize>(
    kdtree: &'a [T],
    query: &P,
    get: impl Fn(&T, usize) -> P::Scalar + Copy,
) -> ItemAndDistance<'a, T, P::Scalar> {
    fn distance_squared<P: KdPoint<N>, T, const N: usize>(
        p1: &P,
        p2: &T,
        get: impl Fn(&T, usize) -> P::Scalar,
    ) -> P::Scalar {
        let mut squared_distance = <P::Scalar as num_traits::Zero>::zero();
        for i in 0..P::dim() {
            let diff = p1.at(i) - get(p2, i);
            squared_distance += diff * diff;
        }
        squared_distance
    }
    fn recurse<'a, T, Q: KdPoint<N>, const N: usize>(
        nearest: &mut ItemAndDistance<'a, T, Q::Scalar>,
        kdtree: &'a [T],
        get: impl Fn(&T, usize) -> Q::Scalar + Copy,
        query: &Q,
        axis: usize,
    ) {
        let mid_idx = kdtree.len() / 2;
        let item = &kdtree[mid_idx];
        let squared_distance = distance_squared(query, item, get);
        if squared_distance < nearest.squared_distance {
            nearest.item = item;
            nearest.squared_distance = squared_distance;
            use num_traits::Zero;
            if nearest.squared_distance.is_zero() {
                return;
            }
        }
        let mid_pos = get(item, axis);
        let [branch1, branch2] = if query.at(axis) < mid_pos {
            [&kdtree[..mid_idx], &kdtree[mid_idx + 1..]]
        } else {
            [&kdtree[mid_idx + 1..], &kdtree[..mid_idx]]
        };
        if !branch1.is_empty() {
            recurse(nearest, branch1, get, query, (axis + 1) % Q::dim());
        }
        if !branch2.is_empty() {
            let diff = query.at(axis) - mid_pos;
            if diff * diff < nearest.squared_distance {
                recurse(nearest, branch2, get, query, (axis + 1) % Q::dim());
            }
        }
    }
    assert!(!kdtree.is_empty());
    let mut nearest = ItemAndDistance {
        item: &kdtree[0],
        squared_distance: distance_squared(query, &kdtree[0], get),
    };
    recurse(&mut nearest, kdtree, get, query, 0);
    nearest
}

#[allow(dead_code)]
pub fn kd_nearest_with<T, Scalar>(
    kdtree: &[T],
    dim: usize,
    kd_difference: impl Fn(&T, usize) -> Scalar + Copy,
) -> ItemAndDistance<T, Scalar>
where
    Scalar: num_traits::NumAssign + Copy + PartialOrd,
{
    fn squared_distance<T, Scalar: num_traits::NumAssign + Copy>(
        item: &T,
        dim: usize,
        kd_difference: impl Fn(&T, usize) -> Scalar + Copy,
    ) -> Scalar {
        let mut squared_distance = Scalar::zero();
        for k in 0..dim {
            let diff = kd_difference(item, k);
            squared_distance += diff * diff;
        }
        squared_distance
    }
    fn recurse<'a, T, Scalar>(
        nearest: &mut ItemAndDistance<'a, T, Scalar>,
        kdtree: &'a [T],
        axis: usize,
        dim: usize,
        kd_difference: impl Fn(&T, usize) -> Scalar + Copy,
    ) where
        Scalar: num_traits::NumAssign + Copy + PartialOrd,
    {
        let mid_idx = kdtree.len() / 2;
        let mid = &kdtree[mid_idx];
        let squared_distance = squared_distance(mid, dim, kd_difference);
        if squared_distance < nearest.squared_distance {
            *nearest = ItemAndDistance {
                item: mid,
                squared_distance,
            };
            if nearest.squared_distance.is_zero() {
                return;
            }
        }
        let [branch1, branch2] = if kd_difference(mid, axis) < Scalar::zero() {
            [&kdtree[..mid_idx], &kdtree[mid_idx + 1..]]
        } else {
            [&kdtree[mid_idx + 1..], &kdtree[..mid_idx]]
        };
        if !branch1.is_empty() {
            recurse(nearest, branch1, (axis + 1) % dim, dim, kd_difference);
        }
        if !branch2.is_empty() {
            let diff = kd_difference(mid, axis);
            if diff * diff < nearest.squared_distance {
                recurse(nearest, branch2, (axis + 1) % dim, dim, kd_difference);
            }
        }
    }
    assert!(!kdtree.is_empty());
    let mut nearest = ItemAndDistance {
        item: &kdtree[0],
        squared_distance: squared_distance(&kdtree[0], dim, kd_difference),
    };
    recurse(&mut nearest, kdtree, 0, dim, kd_difference);
    nearest
}
