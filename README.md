# kd-tree

k-dimensional tree in Rust.

Fast, simple, and easy to use.

## Usage

```rust
// construct kd-tree
let kdtree = kd_tree::KdTree::build_by_ordered_float(vec![
    [1.0, 2.0, 3.0],
    [3.0, 1.0, 2.0],
    [2.0, 3.0, 1.0],
]);

// search the nearest neighbor
let found = kdtree.nearest(&[3.1, 0.9, 2.1]).unwrap();
assert_eq!(found.item, &[3.0, 1.0, 2.0]);

// search k-nearest neighbors
let found = kdtree.nearests(&[1.5, 2.5, 1.8], 2);
assert_eq!(found[0].item, &[2.0, 3.0, 1.0]);
assert_eq!(found[1].item, &[1.0, 2.0, 3.0]);

// search points within a sphere
let found = kdtree.within_radius(&[2.0, 1.5, 2.5], 1.5);
assert_eq!(found.len(), 2);
assert!(found.iter().any(|&&p| p == [1.0, 2.0, 3.0]));
assert!(found.iter().any(|&&p| p == [3.0, 1.0, 2.0]));
```

## With or without `KdPoint`

`KdPoint` trait represents k-dimensional point.

You can live with or without `KdPoint`.

### With `KdPoint` explicitly

```rust
use kd_tree::{KdPoint, KdTree};

// define your own item type.
struct Item {
    point: [f64; 2],
    id: usize,
}

// implement `KdPoint` for your item type.
impl KdPoint for Item {
    type Scalar = f64;
    type Dim = typenum::U2; // 2 dimensional tree.
    fn at(&self, k: usize) -> f64 { self.point[k] }
}

// construct kd-tree from `Vec<Item>`.
// Note: you need to use `build_by_ordered_float()` because f64 doesn't implement `Ord` trait.
let kdtree: KdTree<Item> = KdTree::build_by_ordered_float(vec![
    Item { point: [1.0, 2.0], id: 111 },
    Item { point: [2.0, 3.0], id: 222 },
    Item { point: [3.0, 4.0], id: 333 },
]);

// search nearest item from [1.9, 3.1]
assert_eq!(kdtree.nearest(&[1.9, 3.1]).unwrap().item.id, 222);
```

### With `KdPoint` implicitly

`KdPoint` trait is implemented for fixed-sized array of numerical types, such as `[f64; 3]` or `[i32, 2]` etc.
So you can build kd-trees of those types without custom implementation of `KdPoint`.

```rust
let items: Vec<[i32; 3]> = vec![[1, 2, 3], [3, 1, 2], [2, 3, 1]];
let kdtree = kd_tree::KdTree::build(items);
assert_eq!(kdtree.nearest(&[3, 1, 2]).unwrap().item, &[3, 1, 2]);
```

`KdPoint` trait is also implemented for tuple of a `KdPoint` and an arbitrary type, like `(P, T)` where `P: KdPoint`.
And a type alias named `KdMap<P, T>` is defined as `KdTree<(P, T)>`.
So you can build a kd-tree from key-value pairs, as below:

```rust
let kdmap: kd_tree::KdMap<[isize; 3], &'static str> = kd_tree::KdMap::build(vec![
    ([1, 2, 3], "foo"),
    ([2, 3, 1], "bar"),
    ([3, 1, 2], "buzz"),
]);
assert_eq!(kdmap.nearest(&[3, 1, 2]).unwrap().item.1, "buzz");
```

#### `nalgebra` feature
`KdPoint` trait is implemented for `nalgebra`'s vectors and points.

Enable `nalgebra` feature in your Cargo.toml:
```toml
kd-tree = { version = "...", features = ["nalgebra"] }
```
Then, you can use it as follows:
```rust
use nalgebra::Point3;
let items: Vec<Point3<i32>> = vec![
    Point3::new(1, 2, 3),
    Point3::new(3, 1, 2),
    Point3::new(2, 3, 1)
];
let kdtree = kd_tree::KdTree::build(items);
let query = Point3::new(3, 1, 2);
assert_eq!(kdtree.nearest(&query).unwrap().item, &query);
```


### Without `KdPoint`

```rust
use std::collections::HashMap;
let items: HashMap<&'static str, [i32; 2]> = vec![
    ("a", [10, 20]),
    ("b", [20, 10]),
    ("c", [20, 20]),
].into_iter().collect();
let kdtree = kd_tree::KdTree2::build_by_key(items.keys().collect(), |key, k| items[*key][k]);
assert_eq!(kdtree.nearest_by(&[18, 21], |key, k| items[*key][k]).unwrap().item, &&"c");
```

## To own, or not to own

`KdSliceN<T, N>` and `KdTreeN<T, N>` are similar to `str` and `String`, or `Path` and `PathBuf`.

- `KdSliceN<T, N>` doesn't own its buffer, but `KdTreeN<T, N>`.
- `KdSliceN<T, N>` is not `Sized`, so it must be dealed in reference mannar.
- `KdSliceN<T, N>` implements `Deref` to `[T]`.
- `KdTreeN<T, N>` implements `Deref` to `KdSliceN<T, N>`.
- Unlike `PathBuf` or `String`, which are mutable, `KdTreeN<T, N>` is immutable.

`&KdSliceN<T, N>` can be constructed directly, not via `KdTreeN`, as below:

```rust
let mut items: Vec<[i32; 3]> = vec![[1, 2, 3], [3, 1, 2], [2, 3, 1]];
let kdtree = kd_tree::KdSlice::sort(&mut items);
assert_eq!(kdtree.nearest(&[3, 1, 2]).unwrap().item, &[3, 1, 2]);
```

## `KdIndexTreeN`

A `KdIndexTreeN` refers a slice of items, `[T]`, and contains kd-tree of indices to the items, `KdTreeN<usize, N>`.
Unlike [`KdSlice::sort`], [`KdIndexTree::build`] doesn't sort input items.

```rust
let items = vec![[1, 2, 3], [3, 1, 2], [2, 3, 1]];
let kdtree = kd_tree::KdIndexTree::build(&items);
assert_eq!(kdtree.nearest(&[3, 1, 2]).unwrap().item, &1); // nearest() returns an index of found item.
```
