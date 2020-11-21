use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use kd_tree::*;

fn bench_kdtree_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("construct");
    for log10n in &[2, 3, 4] {
        group.bench_with_input(
            BenchmarkId::new("kd_tree (f64)", log10n),
            log10n,
            |b, log10n| {
                let points = gen_points3d(10usize.pow(*log10n));
                b.iter(|| KdTree::build_by_ordered_float(points.clone()));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("kd_tree (i32)", log10n),
            log10n,
            |b, log10n| {
                let points = gen_points3i(10usize.pow(*log10n));
                b.iter(|| KdTree::build(points.clone()));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("fux_kdtree", log10n),
            log10n,
            |b, log10n| {
                let points = gen_points3d(10usize.pow(*log10n));
                b.iter(|| fux_kdtree::kdtree::Kdtree::new(&mut points.clone()));
            },
        );
        group.bench_with_input(BenchmarkId::new("kdtree", log10n), log10n, |b, log10n| {
            let points = gen_points3d(10usize.pow(*log10n));
            b.iter(|| {
                let mut kdtree = kdtree::KdTree::new(3);
                for p in &points {
                    kdtree.add(&p.coord, p.id).unwrap();
                }
            })
        });
    }
}

fn bench_kdtree_nearest_search(c: &mut Criterion) {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut group = c.benchmark_group("nearest");
    for log10n in &[2, 3, 4] {
        group.bench_with_input(
            BenchmarkId::new("kd_tree (f64)", log10n),
            log10n,
            |b, log10n| {
                let kdtree = KdTree::build_by_ordered_float(gen_points3d(10usize.pow(*log10n)));
                b.iter(|| {
                    let i = rng.gen::<usize>() % kdtree.len();
                    assert_eq!(
                        kdtree.nearest(&kdtree[i]).unwrap().item.coord,
                        kdtree[i].coord
                    );
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("kd_tree (i32)", log10n),
            log10n,
            |b, log10n| {
                let kdtree = KdTree::build(gen_points3i(10usize.pow(*log10n)));
                b.iter(|| {
                    let i = rng.gen::<usize>() % kdtree.len();
                    assert_eq!(
                        kdtree.nearest(&kdtree[i]).unwrap().item.coord,
                        kdtree[i].coord
                    );
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("fux_kdtree", log10n),
            log10n,
            |b, log10n| {
                let mut points = gen_points3d(10usize.pow(*log10n));
                let kdtree = fux_kdtree::kdtree::Kdtree::new(&mut points);
                b.iter(|| {
                    let i = rng.gen::<usize>() % points.len();
                    assert_eq!(kdtree.nearest_search(&points[i]).coord, points[i].coord);
                });
            },
        );
        group.bench_with_input(BenchmarkId::new("kdtree", log10n), log10n, |b, log10n| {
            let points = gen_points3d(10usize.pow(*log10n));
            let mut kdtree = kdtree::KdTree::new(3);
            for p in &points {
                kdtree.add(&p.coord, p.id).unwrap();
            }
            b.iter(|| {
                let i = rng.gen::<usize>() % points.len();
                assert_eq!(
                    kdtree
                        .nearest(&points[i].coord, 1, &kdtree::distance::squared_euclidean)
                        .unwrap()[0]
                        .1,
                    &points[i].id
                );
            })
        });
    }
}

criterion_group!(benches1, bench_kdtree_construction);
criterion_group!(benches2, bench_kdtree_nearest_search);
criterion_main!(benches1, benches2);

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
