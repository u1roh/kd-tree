//! k-dimensional tree.
//!
//! # Example
//! ```
//! let kdtree = kd_tree::KdTree::build_by_ordered_float(vec![
//!     [1.0, 2.0, 3.0],
//!     [3.0, 1.0, 2.0],
//!     [2.0, 3.0, 1.0],
//! ]);
//! assert_eq!(kdtree.nearest(&[3.1, 0.9, 2.1]).item, &[3.0, 1.0, 2.0]);
//! ```
mod nearest;
mod sort;
use nearest::*;
use sort::*;
use std::cmp::Ordering;
use std::marker::PhantomData;
use typenum::Unsigned;

/// A trait to represent k-dimensional point.
///
/// # Example
/// ```
/// struct MyItem {
///     point: [f64; 3],
///     id: usize,
/// }
/// impl kd_tree::KdPoint for MyItem {
///     type Scalar = f64;
///     type Dim = typenum::U3;
///     fn at(&self, k: usize) -> f64 { self.point[k] }
/// }
/// let kdtree: kd_tree::KdTree<MyItem> = kd_tree::KdTree::build_by_ordered_float(vec![
///     MyItem { point: [1.0, 2.0, 3.0], id: 111 },
///     MyItem { point: [3.0, 1.0, 2.0], id: 222 },
///     MyItem { point: [2.0, 3.0, 1.0], id: 333 },
/// ]);
/// assert_eq!(kdtree.nearest(&[3.1, 0.1, 2.2]).item.id, 222);
/// ```
pub trait KdPoint {
    type Scalar: num_traits::NumAssign + Copy + PartialOrd;
    type Dim: Unsigned;
    fn dim() -> usize {
        <Self::Dim as Unsigned>::to_usize()
    }
    fn at(&self, i: usize) -> Self::Scalar;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ItemAndDistance<'a, T, Scalar> {
    pub item: &'a T,
    pub squared_distance: Scalar,
}

/// A slice of kd-tree.
/// This type implements [`std::ops::Deref`] to `[T]`.
/// This is an unsized type, meaning that it must always be used as a reference.
/// For an owned version of this type, see [`KdTree`].
#[derive(Debug, PartialEq, Eq)]
pub struct KdSliceN<T, N: Unsigned>(PhantomData<N>, [T]);
pub type KdSlice<T> = KdSliceN<T, <T as KdPoint>::Dim>;
impl<T, N: Unsigned> std::ops::Deref for KdSliceN<T, N> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        &self.1
    }
}
impl<T: Clone, N: Unsigned> std::borrow::ToOwned for KdSliceN<T, N> {
    type Owned = KdTreeN<T, N>;
    fn to_owned(&self) -> Self::Owned {
        KdTreeN(PhantomData, self.1.to_vec())
    }
}
impl<T, N: Unsigned> KdSliceN<T, N> {
    pub fn items(&self) -> &[T] {
        &self.1
    }

    unsafe fn new_unchecked(items: &[T]) -> &Self {
        &*(items as *const _ as *const Self)
    }

    /// # Example
    /// ```
    /// struct Item {
    ///     point: [i32; 3],
    ///     id: usize,
    /// }
    /// let mut items: Vec<Item> = vec![
    ///     Item { point: [1, 2, 3], id: 111 },
    ///     Item { point: [3, 1, 2], id: 222 },
    ///     Item { point: [2, 3, 1], id: 333 },
    /// ];
    /// let kdtree = kd_tree::KdSlice3::sort_by(&mut items, |item1, item2, k| item1.point[k].cmp(&item2.point[k]));
    /// assert_eq!(kdtree.nearest_by(&[3, 1, 2], |item, k| item.point[k]).item.id, 222);
    /// ```
    pub fn sort_by<F>(items: &mut [T], compare: F) -> &Self
    where
        F: Fn(&T, &T, usize) -> Ordering + Copy,
    {
        kd_sort_by(items, N::to_usize(), compare);
        unsafe { Self::new_unchecked(items) }
    }

    /// # Example
    /// ```
    /// struct Item {
    ///     point: [f64; 3],
    ///     id: usize,
    /// }
    /// let mut items: Vec<Item> = vec![
    ///     Item { point: [1.0, 2.0, 3.0], id: 111 },
    ///     Item { point: [3.0, 1.0, 2.0], id: 222 },
    ///     Item { point: [2.0, 3.0, 1.0], id: 333 },
    /// ];
    /// use ordered_float::OrderedFloat;
    /// let kdtree = kd_tree::KdSlice3::sort_by_key(&mut items, |item, k| OrderedFloat(item.point[k]));
    /// assert_eq!(kdtree.nearest_by(&[3.1, 0.9, 2.1], |item, k| item.point[k]).item.id, 222);
    /// ```
    pub fn sort_by_key<Key: Ord, F>(items: &mut [T], kd_key: F) -> &Self
    where
        F: Fn(&T, usize) -> Key + Copy,
    {
        Self::sort_by(items, |item1, item2, k| {
            kd_key(item1, k).cmp(&kd_key(item2, k))
        })
    }

    /// # Example
    /// ```
    /// use kd_tree::KdSlice;
    /// let mut items: Vec<[f64; 3]> = vec![[1.0, 2.0, 3.0], [3.0, 1.0, 2.0], [2.0, 3.0, 1.0]];
    /// let kdtree: &KdSlice<[f64; 3]> = KdSlice::sort_by_ordered_float(&mut items);
    /// assert_eq!(kdtree.nearest(&[3.1, 0.9, 2.1]).item, &[3.0, 1.0, 2.0]);
    /// ```
    pub fn sort_by_ordered_float(points: &mut [T]) -> &Self
    where
        T: KdPoint<Dim = N>,
        T::Scalar: num_traits::Float,
    {
        Self::sort_by_key(points, |item, k| ordered_float::OrderedFloat(item.at(k)))
    }

    /// # Example
    /// ```
    /// use kd_tree::KdSlice;
    /// let mut items: Vec<[i32; 3]> = vec![[1, 2, 3], [3, 1, 2], [2, 3, 1]];
    /// let kdtree: &KdSlice<[i32; 3]> = KdSlice::sort(&mut items);
    /// assert_eq!(kdtree.nearest(&[3, 1, 2]).item, &[3, 1, 2]);
    /// ```
    pub fn sort(points: &mut [T]) -> &Self
    where
        T: KdPoint<Dim = N>,
        T::Scalar: Ord,
    {
        Self::sort_by_key(points, |item, k| item.at(k))
    }

    /// # Example
    /// ```
    /// struct Item {
    ///     point: [f64; 3],
    ///     id: usize,
    /// }
    /// let mut items: Vec<Item> = vec![
    ///     Item { point: [1.0, 2.0, 3.0], id: 111 },
    ///     Item { point: [3.0, 1.0, 2.0], id: 222 },
    ///     Item { point: [2.0, 3.0, 1.0], id: 333 },
    /// ];
    /// use ordered_float::OrderedFloat;
    /// let kdtree = kd_tree::KdSlice3::sort_by_key(&mut items, |item, k| OrderedFloat(item.point[k]));
    /// assert_eq!(kdtree.nearest_by(&[3.1, 0.9, 2.1], |item, k| item.point[k]).item.id, 222);
    /// ```
    /// # Panics
    /// Panics if `self.is_empty()`.
    pub fn nearest_by<Q: KdPoint>(
        &self,
        query: &Q,
        coord: impl Fn(&T, usize) -> Q::Scalar + Copy,
    ) -> ItemAndDistance<T, Q::Scalar> {
        kd_nearest_by(self.items(), query, coord)
    }

    /// # Example
    /// ```
    /// let mut items: Vec<[i32; 3]> = vec![[1, 2, 3], [3, 1, 2], [2, 3, 1]];
    /// let kdtree = kd_tree::KdSlice::sort(&mut items);
    /// assert_eq!(kdtree.nearest(&[3, 1, 2]).item, &[3, 1, 2]);
    /// ```
    /// # Panics
    /// Panics if `self.is_empty()`.
    pub fn nearest(
        &self,
        query: &impl KdPoint<Scalar = T::Scalar, Dim = T::Dim>,
    ) -> ItemAndDistance<T, T::Scalar>
    where
        T: KdPoint,
    {
        kd_nearest(self.items(), query)
    }

    /*
    /// # Example
    /// ```
    /// let kdtree = kd_tree::KdTree3::build(vec![[1, 2, 3], [3, 1, 2], [2, 3, 1]]);
    /// let key = [3, 1, 2];
    /// assert_eq!(kdtree.nearest_with(|p, k| key[k] - p[k]).item, &[3, 1, 2]);
    /// ```
    pub fn nearest_with<Scalar>(
        &self,
        kd_difference: impl Fn(&T, usize) -> Scalar + Copy,
    ) -> ItemAndDistance<T, Scalar>
    where
        Scalar: num_traits::NumAssign + Copy + PartialOrd,
    {
        kd_nearest_with(self.items(), N::to_usize(), kd_difference)
    }
    */
}

/// An owned kd-tree.
/// This type implements [`std::ops::Deref`] to [`KdSlice`].
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct KdTreeN<T, N: Unsigned>(PhantomData<N>, Vec<T>);
pub type KdTree<T> = KdTreeN<T, <T as KdPoint>::Dim>;
impl<T, N: Unsigned> std::ops::Deref for KdTreeN<T, N> {
    type Target = KdSliceN<T, N>;
    fn deref(&self) -> &Self::Target {
        unsafe { KdSliceN::new_unchecked(&self.1) }
    }
}
impl<T, N: Unsigned> AsRef<KdSliceN<T, N>> for KdTreeN<T, N> {
    fn as_ref(&self) -> &KdSliceN<T, N> {
        self
    }
}
impl<T, N: Unsigned> std::borrow::Borrow<KdSliceN<T, N>> for KdTreeN<T, N> {
    fn borrow(&self) -> &KdSliceN<T, N> {
        self
    }
}
impl<T, N: Unsigned> Into<Vec<T>> for KdTreeN<T, N> {
    fn into(self) -> Vec<T> {
        self.1
    }
}
impl<T, N: Unsigned> KdTreeN<T, N> {
    pub fn into_vec(self) -> Vec<T> {
        self.1
    }

    /// # Example
    /// ```
    /// struct Item {
    ///     point: [i32; 3],
    ///     id: usize,
    /// }
    /// let kdtree = kd_tree::KdTree3::build_by(
    ///     vec![
    ///         Item { point: [1, 2, 3], id: 111 },
    ///         Item { point: [3, 1, 2], id: 222 },
    ///         Item { point: [2, 3, 1], id: 333 },
    ///     ],
    ///     |item1, item2, k| item1.point[k].cmp(&item2.point[k])
    /// );
    /// assert_eq!(kdtree.nearest_by(&[3, 1, 2], |item, k| item.point[k]).item.id, 222);
    /// ```
    pub fn build_by<F>(mut items: Vec<T>, compare: F) -> Self
    where
        F: Fn(&T, &T, usize) -> Ordering + Copy,
    {
        kd_sort_by(&mut items, N::to_usize(), compare);
        Self(PhantomData, items)
    }

    /// # Example
    /// ```
    /// struct Item {
    ///     point: [f64; 3],
    ///     id: usize,
    /// }
    /// let kdtree = kd_tree::KdTree3::build_by_key(
    ///     vec![
    ///         Item { point: [1.0, 2.0, 3.0], id: 111 },
    ///         Item { point: [3.0, 1.0, 2.0], id: 222 },
    ///         Item { point: [2.0, 3.0, 1.0], id: 333 },
    ///     ],
    ///     |item, k| ordered_float::OrderedFloat(item.point[k])
    /// );
    /// assert_eq!(kdtree.nearest_by(&[3.1, 0.9, 2.1], |item, k| item.point[k]).item.id, 222);
    /// ```
    pub fn build_by_key<Key, F>(items: Vec<T>, kd_key: F) -> Self
    where
        Key: Ord,
        F: Fn(&T, usize) -> Key + Copy,
    {
        Self::build_by(items, |item1, item2, k| {
            kd_key(item1, k).cmp(&kd_key(item2, k))
        })
    }

    /// # Example
    /// ```
    /// use kd_tree::KdTree;
    /// let kdtree: KdTree<[f64; 3]> = KdTree::build_by_ordered_float(vec![
    ///     [1.0, 2.0, 3.0], [3.0, 1.0, 2.0], [2.0, 3.0, 1.0]
    /// ]);
    /// assert_eq!(kdtree.nearest(&[3.1, 0.9, 2.1]).item, &[3.0, 1.0, 2.0]);
    /// ```
    pub fn build_by_ordered_float(points: Vec<T>) -> Self
    where
        T: KdPoint<Dim = N>,
        T::Scalar: num_traits::Float,
    {
        Self::build_by_key(points, |item, k| ordered_float::OrderedFloat(item.at(k)))
    }

    /// # Example
    /// ```
    /// use kd_tree::KdTree;
    /// let kdtree: KdTree<[i32; 3]> = KdTree::build(vec![[1, 2, 3], [3, 1, 2], [2, 3, 1]]);
    /// assert_eq!(kdtree.nearest(&[3, 1, 2]).item, &[3, 1, 2]);
    /// ```
    pub fn build(points: Vec<T>) -> Self
    where
        T: KdPoint<Dim = N>,
        T::Scalar: Ord,
    {
        Self::build_by_key(points, |item, k| item.at(k))
    }
}

/// This type refers a slice of items, `[T]`, and contains kd-tree of indices to the items, `KdTree<usize, N>`.
/// Unlike [`KdSlice::sort`], [`KdIndexTree::build`] doesn't sort input items.
/// ```
/// let items = vec![[1, 2, 3], [3, 1, 2], [2, 3, 1]];
/// let kdtree = kd_tree::KdIndexTree::build(&items);
/// assert_eq!(kdtree.nearest(&[3, 1, 2]).item, &1); // nearest() returns an index of items.
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KdIndexTreeN<'a, T, N: Unsigned> {
    source: &'a [T],
    kdtree: KdTreeN<usize, N>,
}
pub type KdIndexTree<'a, T> = KdIndexTreeN<'a, T, <T as KdPoint>::Dim>;
impl<'a, T, N: Unsigned> KdIndexTreeN<'a, T, N> {
    pub fn source(&self) -> &'a [T] {
        self.source
    }

    pub fn indices(&self) -> &KdSliceN<usize, N> {
        &self.kdtree
    }

    pub fn item(&self, i: usize) -> &'a T {
        &self.source[i]
    }

    pub fn build_by<F>(source: &'a [T], compare: F) -> Self
    where
        F: Fn(&T, &T, usize) -> Ordering + Copy,
    {
        Self {
            source,
            kdtree: KdTreeN::build_by((0..source.len()).collect(), |i1, i2, k| {
                compare(&source[*i1], &source[*i2], k)
            }),
        }
    }

    pub fn build_by_key<Key, F>(source: &'a [T], kd_key: F) -> Self
    where
        Key: Ord,
        F: Fn(&T, usize) -> Key + Copy,
    {
        Self::build_by(source, |item1, item2, k| {
            kd_key(item1, k).cmp(&kd_key(item2, k))
        })
    }

    pub fn build_by_ordered_float(points: &'a [T]) -> Self
    where
        T: KdPoint<Dim = N>,
        T::Scalar: num_traits::Float,
    {
        Self::build_by_key(points, |item, k| ordered_float::OrderedFloat(item.at(k)))
    }

    pub fn build(points: &'a [T]) -> Self
    where
        T: KdPoint<Dim = N>,
        T::Scalar: Ord,
    {
        Self::build_by_key(points, |item, k| item.at(k))
    }

    pub fn nearest_by<Q: KdPoint>(
        &self,
        query: &Q,
        coord: impl Fn(&T, usize) -> Q::Scalar + Copy,
    ) -> ItemAndDistance<usize, Q::Scalar> {
        self.kdtree
            .nearest_by(query, |&index, k| coord(&self.source[index], k))
    }

    /// # Example
    /// ```
    /// let mut items: Vec<[i32; 3]> = vec![[1, 2, 3], [3, 1, 2], [2, 3, 1]];
    /// let kdtree = kd_tree::KdIndexTree3::build(&items);
    /// assert_eq!(kdtree.nearest(&[3, 1, 2]).item, &1);
    /// ```
    pub fn nearest(
        &self,
        query: &impl KdPoint<Scalar = T::Scalar, Dim = T::Dim>,
    ) -> ItemAndDistance<usize, T::Scalar>
    where
        T: KdPoint,
    {
        self.kdtree
            .nearest_by(query, |&index, k| self.source[index].at(k))
    }
}

macro_rules! define_kdtree_aliases {
    ($($dim:literal),*) => {
        $(
            paste::paste! {
                pub type [<KdSlice $dim>]<T> = KdSliceN<T, typenum::[<U $dim>]>;
                pub type [<KdTree $dim>]<T> = KdTreeN<T, typenum::[<U $dim>]>;
                pub type [<KdIndexTree $dim>]<'a, T> = KdIndexTreeN<'a, T, typenum::[<U $dim>]>;
            }
        )*
    };
}
define_kdtree_aliases!(1, 2, 3, 4, 5, 6, 7, 8);

macro_rules! impl_kd_points {
    ($($len:literal),*) => {
        $(
            paste::paste!{
                impl<T: num_traits::NumAssign + Copy + PartialOrd> KdPoint for [T; $len] {
                    type Scalar = T;
                    type Dim = typenum::[<U $len>];
                    fn at(&self, i: usize) -> T { self[i] }
                }
            }
        )*
    };
}
impl_kd_points!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);

impl<P: KdPoint, T> KdPoint for (P, T) {
    type Scalar = P::Scalar;
    type Dim = P::Dim;
    fn at(&self, k: usize) -> Self::Scalar {
        self.0.at(k)
    }
}

/// kd-tree of key-value pairs.
/// ```
/// let kdmap: kd_tree::KdMap<[isize; 3], &'static str> = kd_tree::KdMap::build(vec![
///     ([1, 2, 3], "foo"),
///     ([2, 3, 1], "bar"),
///     ([3, 1, 2], "buzz"),
/// ]);
/// assert_eq!(kdmap.nearest(&[3, 1, 2]).item.1, "buzz");
/// ```
pub type KdMap<P, T> = KdTree<(P, T)>;

/// kd-tree slice of key-value pairs.
/// ```
/// let mut items: Vec<([isize; 3], &'static str)> = vec![
///     ([1, 2, 3], "foo"),
///     ([2, 3, 1], "bar"),
///     ([3, 1, 2], "buzz"),
/// ];
/// let kdmap = kd_tree::KdMapSlice::sort(&mut items);
/// assert_eq!(kdmap.nearest(&[3, 1, 2]).item.1, "buzz");
/// ```
pub type KdMapSlice<P, T> = KdSlice<(P, T)>;
