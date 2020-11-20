fn main() {
    let kdtree = {
        let now = std::time::Instant::now();
        let kdtree = kd_tree::KdTreeBuf3::build_by_ordered_float(gen_points(1000000));
        println!(
            "KdTreeBuf3::build_by_ordered_float(): elapsed {:?}",
            now.elapsed()
        );
        kdtree
    };
    {
        let now = std::time::Instant::now();
        for p in kdtree.items() {
            let nearest = kdtree.nearest(p);
            assert_eq!(nearest.item, p);
        }
        println!("KdTree::nearest: elapsed {:?}", now.elapsed());
    }
    {
        let now = std::time::Instant::now();
        for q in kdtree.items() {
            let nearest = kdtree.nearest_with(|p, k| q[k] - p[k]);
            assert_eq!(nearest.item, q);
        }
        println!("kd_nearest_with: elapsed {:?}", now.elapsed());
    }
}

fn gen_points(count: usize) -> Vec<[f64; 3]> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut points = Vec::with_capacity(count);
    for _ in 0..count {
        points.push([rng.gen(), rng.gen(), rng.gen()]);
    }
    points
}
