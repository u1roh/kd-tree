use ordered_float::OrderedFloat;

fn main() {
    let mut points = gen_points(1000000);
    {
        let now = std::time::Instant::now();
        kd_tree::kd::kd_sort_by_key(&mut points, 3, |p, k| OrderedFloat(p[k]));
        println!("kd_sort_by_key: elapsed {:?}", now.elapsed());
    }
    {
        let now = std::time::Instant::now();
        for p in &points {
            let nearest = kd_tree::kd::kd_nearest_by(&points, p, |p, k| p[k]);
            assert_eq!(nearest.item, p);
        }
        println!("kd_nearest_by: elapsed {:?}", now.elapsed());
    }
    {
        let now = std::time::Instant::now();
        for p in &points {
            let nearest = kd_tree::kd::kd_nearest(&points, p);
            assert_eq!(nearest.item, p);
        }
        println!("kd_nearest: elapsed {:?}", now.elapsed());
    }
    {
        //let kdtree = kd_tree::kd::KdTree::<_, [f64; 3], _>::sort_by_key(
        let kdtree = kd_tree::kd::KdTreeRef3::sort_points_by(&mut points, OrderedFloat);
        let now = std::time::Instant::now();
        for p in kdtree.items() {
            //let nearest = kd_tree::kd::kd_find_nearest(&points, |p, k| p[k], p);
            let nearest = kdtree.nearest(p);
            assert_eq!(nearest.item, p);
        }
        println!("KdTree::nearest: elapsed {:?}", now.elapsed());
    }
    {
        let now = std::time::Instant::now();
        for q in &points {
            let nearest = kd_tree::kd::kd_nearest_with(&points, 3, |p, k| q[k] - p[k]);
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
