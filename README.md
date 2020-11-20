# kd-tree
k-dimensional tree.

Fast, simple, and easy to use.

Only nearest point search is available, currently.

## With or without `KdPoint`
`KdPoint` trait represents k-dimensional point.

You can live with or without `KdPoint`.

### With `KdPoint` explicitly
```rust
// define your own item type.
struct Item {
    point: [f64; 2],
    id: usize,
}

// implement `KdPoint` for your item type.
impl kd_tree::KdPoint for Item {
    type Scalar = f64;
    type Dim = typenum::U2; // 2 dimensional tree.
    fn at(&self, k: usize) -> f64 { self.point[k] }
}

// construct kd-tree from `Vec<Item>`.
// Note: you need to use `build_by_ordered_float()` because f64 doesn't implement `Ord` trait.
let kdtree = kd_tree::KdTreeBuf::build_by_ordered_float(vec![
    Item { point: [1.0, 2.0], id: 111 },
    Item { point: [2.0, 3.0], id: 222 },
    Item { point: [3.0, 4.0], id: 333 },
]);

// search nearest item from [1.9, 3.1] 
assert_eq!(kdtree.nearest(&[1.9, 3.1]).item.id, 222);
```

### With `KdPoint` implicitly
`KdPoint` trait is implemented for fixed-sized array of numerical types, such as `[f64; 3]` or `[i32, 2]` etc.
So you can build kd-trees of those types without custom implementation of `KdPoint`.
```rust
let items: Vec<[i32; 3]> = vec![[1, 2, 3], [3, 1, 2], [2, 3, 1]];
let kdtree = kd_tree::KdTreeBuf::build(items);
assert_eq!(kdtree.nearest(&[3, 1, 2]).item, &[3, 1, 2]);
```

### Without `KdPoint`
```rust
use std::collections::HashMap;
let items: HashMap<'static str, [i32; 2]> = vec![
    ("a", [10, 20]),
    ("b", [20, 10]),
    ("c", [20, 20]),
].into_iter().collect();
let kdtree = kd_tree::KdTreeBuf::build_by_key(vec!["a", "b", "c"], |key, k| items[*key][k]);
assert_eq!(kdtree.nearest_by(&[18, 21], |key, k| items[*key][k]).item, &"c");
```

# To own, or not to own
`KdTree<T, N>` and `KdTreeBuf<T, N>` are similar to `Path` and `PathBuf`.

* `KdTree<T, N>` doesn't own its buffer, but `KdTreeBuf<T, N>`.
* `KdTree<T, N>` is not `Sized`, so it must be dealed in reference mannar.
* `KdTreeBuf<T, N>` implements `Deref` to `KdTree<T, N>`.
* But unlike `PathBuf`, which is mutable, `KdTreeBuf<T, N>` is immutable.

`&KdTree<T, N>` can be constructed directly, not via `KdTreeBuf`, as below:
```rust
let mut items: Vec<[i32; 3]> = vec![[1, 2, 3], [3, 1, 2], [2, 3, 1]];
let kdtree = kd_tree::KdTree::sort(&mut items);
assert_eq!(kdtree.nearest(&[3, 1, 2]).item, &[3, 1, 2]);
```    
