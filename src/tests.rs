#![cfg(test)]
use super::*;

#[test]
fn test_nearest() {
    let mut gen3d = random3d_generator();
    let kdtree = KdTree::build_by_ordered_float(vec(10000, |_| gen3d()));
    for _ in 0..100 {
        let query = gen3d();
        let found = kdtree.nearest(&query).unwrap().item;
        let expected = kdtree
            .iter()
            .min_by_key(|p| ordered_float::OrderedFloat(squared_distance(p, &query)))
            .unwrap();
        assert_eq!(found, expected);
    }
}

#[test]
fn test_nearests() {
    let mut gen3d = random3d_generator();
    let kdtree = KdTree::build_by_ordered_float(vec(10000, |_| gen3d()));
    const NUM: usize = 5;
    for _ in 0..100 {
        let query = gen3d();
        let found = kdtree.nearests(&query, NUM);
        assert_eq!(found.len(), NUM);
        for i in 1..found.len() {
            assert!(found[i - 1].squared_distance <= found[i].squared_distance);
        }
        let count = kdtree
            .iter()
            .filter(|p| squared_distance(p, &query) <= found[NUM - 1].squared_distance)
            .count();
        assert_eq!(count, NUM);
    }
}

#[test]
fn test_within() {
    let mut gen3d = random3d_generator();
    let kdtree = KdTree::build_by_ordered_float(vec(10000, |_| gen3d()));
    for _ in 0..100 {
        let mut p1 = gen3d();
        let mut p2 = gen3d();
        for k in 0..3 {
            if p1[k] > p2[k] {
                std::mem::swap(&mut p1[k], &mut p2[k]);
            }
        }
        let found = kdtree.within(&[p1, p2]);
        let count = kdtree
            .iter()
            .filter(|p| (0..3).all(|k| p1[k] <= p[k] && p[k] <= p2[k]))
            .count();
        assert_eq!(found.len(), count);
    }
}

#[test]
fn test_within_radius() {
    let mut gen3d = random3d_generator();
    let kdtree = KdTree::build_by_ordered_float(vec(10000, |_| gen3d()));
    const RADIUS: f64 = 0.1;
    for _ in 0..100 {
        let query = gen3d();
        let found = kdtree.within_radius(&query, RADIUS);
        let count = kdtree
            .iter()
            .filter(|p| squared_distance(p, &query) < RADIUS * RADIUS)
            .count();
        assert_eq!(found.len(), count);
    }
}

fn squared_distance<T: num_traits::Num + Copy>(p1: &[T; 3], p2: &[T; 3]) -> T {
    let dx = p1[0] - p2[0];
    let dy = p1[1] - p2[1];
    let dz = p1[2] - p2[2];
    dx * dx + dy * dy + dz * dz
}

// generates a random number between 0 and 1 with 0.1 step
fn random_10th(rng: &mut impl rand::Rng) -> f64 {
    f64::from(rng.gen_range(0u8, 10u8)) / 10.0
}

fn random3d_generator() -> impl FnMut() -> [f64; 3] {
    let mut rng = rand::thread_rng();
    move || [random_10th(&mut rng), random_10th(&mut rng), random_10th(&mut rng)]
}

fn vec<T>(count: usize, mut f: impl FnMut(usize) -> T) -> Vec<T> {
    let mut items = Vec::with_capacity(count);
    for i in 0..count {
        items.push(f(i));
    }
    items
}
