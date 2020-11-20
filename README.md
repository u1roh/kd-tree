# kd-tree
k-dimensional tree.

Fast, simple, and easy to use.

## Usage

### Explicit use of `KdPoint` trait
```rust
// define your own item type.
struct Item {
    point: [f64; 3],
    id: usize,
}

// implement `KdPoint` for your item type.
impl kd_tree::KdPoint for Item {
    type Scalar = f64;
    type Dim = typenum::U3;
    fn at(&self, k: usize) -> f64 { self.point[k] }
}

// construct kd-tree from `Vec<Item>`.
// you need to use `build_by_ordered_float()` because f64 doesn't implement `Ord` trait.
let kdtree = kd_tree::KdTreeBuf::build_by_ordered_float(vec![
    Item { point: [1.0, 2.0, 3.0], id: 111 },
    Item { point: [3.0, 1.0, 2.0], id: 222 },
    Item { point: [2.0, 3.0, 1.0], id: 333 },
]);

// search nearest item from [3.1, 0.1, 2.2] 
assert_eq!(kdtree.nearest(&[3.1, 0.1, 2.2]).item.id, 222);
```

### Implicit use of `KdPoint` trait
`KdPoint` trait is implemented for fixed-sized array of numerical types, such as `[f64; 3]` or `[i32, 2]` etc.
So you can build kd-trees of those types without custom implementation of `KdPoint`.
```rust
let items: Vec<[i32; 3]> = vec![[1, 2, 3], [3, 1, 2], [2, 3, 1]];
let kdtree = kd_tree::KdTreeBuf::build(items);
assert_eq!(kdtree.nearest(&[3, 1, 2]).item, &[3, 1, 2]);
```

### No use of `KdPoint` trait
