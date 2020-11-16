use ordered_float::OrderedFloat;

fn main() {
    let mut points = gen_points(1000000);
    {
        let now = std::time::Instant::now();
        //kd_tree::kd::kd_sort(&mut points, 3, |p, k| OrderedFloat(p[k]));
        kd_tree::kd::kd_sort_points_by_key(&mut points, OrderedFloat);
        println!("construction: elapsed {:?}", now.elapsed());
    }
    {
        let now = std::time::Instant::now();
        for p in &points {
            let nearest = kd_tree::kd::kd_find_nearest(&points, |p, k| p[k], p);
            assert_eq!(nearest.item, p);
        }
        println!("nearest search: elapsed {:?}", now.elapsed());
    }
    {
        let now = std::time::Instant::now();
        for p in &points {
            let nearest = kd_tree::kd::kd_find_nearest2(&points, |p, k| p[k], p);
            assert_eq!(nearest.item, p);
        }
        println!("nearest search: elapsed {:?}", now.elapsed());
    }
    {
        let now = std::time::Instant::now();
        for p in &points {
            let nearest = kd_tree::kd::kd_find_nearest3(&points, p);
            assert_eq!(nearest.item, p);
        }
        println!("nearest search: elapsed {:?}", now.elapsed());
    }
    {
        let now = std::time::Instant::now();
        for q in &points {
            let nearest = kd_tree::kd::kd_find_nearest4(
                &points,
                3,
                |p, k| p[k],
                |k| q[k],
                |p| {
                    let dx = p[0] - q[0];
                    let dy = p[1] - q[1];
                    let dz = p[2] - q[2];
                    dx * dx + dy * dy + dz * dz
                },
            );
            assert_eq!(nearest.item, q);
        }
        println!("nearest search: elapsed {:?}", now.elapsed());
    }
    {
        let kdtree = kd_tree::kd::KdTree::<_, [f64; 3], _>::sort_by_key(
            &mut points,
            |p, k| p[k],
            OrderedFloat,
        );
        let now = std::time::Instant::now();
        for p in kdtree.source {
            //let nearest = kd_tree::kd::kd_find_nearest(&points, |p, k| p[k], p);
            let nearest = kdtree.nearest(p);
            assert_eq!(nearest.item, p);
        }
        println!("nearest search: elapsed {:?}", now.elapsed());
    }
    /*{
        let now = std::time::Instant::now();
        for p in &points {
            let (nearest, _) = kd_tree::kd::kd_find_nearest2(&points, |p, k| p[k], p, 3);
            assert_eq!(nearest, p);
        }
        println!("nearest search: elapsed {:?}", now.elapsed());
    }*/
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
