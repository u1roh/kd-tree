use criterion::{criterion_group, criterion_main, Criterion};
use kd_tree::*;

fn criterion_benchmark(c: &mut Criterion) {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    const NUM_GEN: usize = 10000;
    const NUM_NEAREST: usize = 1000000;
    c.bench_function("gen kdtree::KdTree", |b| {
        let points = gen_points3d(NUM_GEN);
        b.iter(|| {
            let mut kdtree = kdtree::KdTree::new(3);
            for p in &points {
                kdtree.add(&p.coord, p.id).unwrap();
            }
        })
    });
    c.bench_function("gen fux_kdtree", |b| {
        let points = gen_points3d(NUM_GEN);
        b.iter(|| fux_kdtree::kdtree::Kdtree::new(&mut points.clone()));
    });
    c.bench_function("gen KdTree<[f64; 3]>", |b| {
        let points = gen_points3d(NUM_GEN);
        b.iter(|| KdTree::build_by_ordered_float(points.clone()));
    });
    c.bench_function("gen KdTree<[i32; 3]>", |b| {
        let points = gen_points3i(NUM_GEN);
        b.iter(|| KdTree::build(points.clone()));
    });
    c.bench_function("nearest by kdtree::KdTree", |b| {
        let points = gen_points3d(NUM_NEAREST);
        let mut kdtree = kdtree::KdTree::new(3);
        for p in &points {
            kdtree.add(&p.coord, p.id).unwrap();
        }
        b.iter(|| {
            let i = rng.gen::<usize>() % NUM_NEAREST;
            assert_eq!(
                kdtree
                    .nearest(&points[i].coord, 1, &kdtree::distance::squared_euclidean)
                    .unwrap()[0]
                    .1,
                &points[i].id
            );
        })
    });
    c.bench_function("nearest by fux_kdtree", |b| {
        let mut points = gen_points3d(NUM_NEAREST);
        let kdtree = fux_kdtree::kdtree::Kdtree::new(&mut points);
        b.iter(|| {
            let i = rng.gen::<usize>() % NUM_NEAREST;
            assert_eq!(kdtree.nearest_search(&points[i]).coord, points[i].coord);
        });
    });
    c.bench_function("nearest by KdTree<[f64; 3]>", |b| {
        let kdtree = KdTree::build_by_ordered_float(gen_points3d(NUM_NEAREST));
        b.iter(|| {
            let i = rng.gen::<usize>() % kdtree.len();
            assert_eq!(kdtree.nearest(&kdtree[i]).item.coord, kdtree[i].coord);
        });
    });
    c.bench_function("nearest by KdTree<[i32; 3]>", |b| {
        let kdtree = KdTree::build(gen_points3i(NUM_NEAREST));
        b.iter(|| {
            let i = rng.gen::<usize>() % kdtree.len();
            assert_eq!(kdtree.nearest(&kdtree[i]).item.coord, kdtree[i].coord);
        });
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);

#[derive(Debug, Clone, Copy, PartialEq)]
struct TestItem<T> {
    coord: [T; 3],
    id: usize,
}
impl<T: num_traits::NumAssign + Copy + PartialOrd> KdPoint for TestItem<T> {
    type Scalar = T;
    type Dim = typenum::U3;
    fn at(&self, k: usize) -> T {
        self.coord[k]
    }
}
impl fux_kdtree::kdtree::KdtreePointTrait for TestItem<f64> {
    fn dims(&self) -> &[f64] {
        &self.coord
    }
}

fn gen_points3d(count: usize) -> Vec<TestItem<f64>> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut points = Vec::with_capacity(count);
    for id in 0..count {
        let coord = [rng.gen(), rng.gen(), rng.gen()];
        points.push(TestItem { coord, id });
    }
    points
}

fn gen_points3i(count: usize) -> Vec<TestItem<i32>> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut points = Vec::with_capacity(count);
    const N: i32 = 1000;
    for id in 0..count {
        let coord = [
            rng.gen::<i32>() % N,
            rng.gen::<i32>() % N,
            rng.gen::<i32>() % N,
        ];
        points.push(TestItem { coord, id });
    }
    points
}
