use crate::{ItemAndDistance, KdPoint};

pub fn kd_nearests<'a, T: KdPoint>(
    kdtree: &'a [T],
    query: &impl KdPoint<Scalar = T::Scalar, Dim = T::Dim>,
    num: usize,
) -> Vec<ItemAndDistance<'a, T, T::Scalar>> {
    kd_nearests_by(kdtree, query, num, |item, k| item.at(k))
}

pub fn kd_nearests_by<'a, T, P: KdPoint>(
    kdtree: &'a [T],
    query: &P,
    num: usize,
    get: impl Fn(&T, usize) -> P::Scalar + Copy,
) -> Vec<ItemAndDistance<'a, T, P::Scalar>> {
    fn distance_squared<P: KdPoint, T>(
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
    fn recurse<'a, T, Q: KdPoint>(
        nearests: &mut Vec<ItemAndDistance<'a, T, Q::Scalar>>,
        kdtree: &'a [T],
        get: impl Fn(&T, usize) -> Q::Scalar + Copy,
        query: &Q,
        axis: usize,
    ) {
        let mid_idx = kdtree.len() / 2;
        let item = &kdtree[mid_idx];
        let squared_distance = distance_squared(query, item, get);
        if nearests.len() < nearests.capacity()
            || squared_distance < nearests.last().unwrap().squared_distance
        {
            if nearests.len() == nearests.capacity() {
                nearests.pop();
            }
            let i = nearests
                .binary_search_by(|item| {
                    item.squared_distance
                        .partial_cmp(&squared_distance)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or_else(|i| i);
            nearests.insert(
                i,
                ItemAndDistance {
                    item,
                    squared_distance,
                },
            );
        }
        let mid_pos = get(item, axis);
        let [branch1, branch2] = if query.at(axis) < mid_pos {
            [&kdtree[..mid_idx], &kdtree[mid_idx + 1..]]
        } else {
            [&kdtree[mid_idx + 1..], &kdtree[..mid_idx]]
        };
        if !branch1.is_empty() {
            recurse(nearests, branch1, get, query, (axis + 1) % Q::dim());
        }
        if !branch2.is_empty() {
            let diff = query.at(axis) - mid_pos;
            if diff * diff < nearests.last().unwrap().squared_distance {
                recurse(nearests, branch2, get, query, (axis + 1) % Q::dim());
            }
        }
    }
    if num == 0 || kdtree.is_empty() {
        return Vec::new();
    }
    let mut nearests = Vec::with_capacity(num);
    recurse(&mut nearests, kdtree, get, query, 0);
    nearests
}
