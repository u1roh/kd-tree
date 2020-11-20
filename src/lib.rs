//! k-dimensional tree.
//!
//! # Example
//! ```
//! let kdtree = kd_tree::KdTreeBuf::build_by_ordered_float(vec![
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
/// let kdtree = kd_tree::KdTreeBuf::build_by_ordered_float(vec![
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

/// A slice of kd-tree.
/// This is an unsized type, meaning that it must always be used as a reference. For an owned version of this type, see [`KdTreeBuf`].
#[derive(Debug, PartialEq, Eq)]
pub struct KdTree<T, N: Unsigned>(PhantomData<N>, [T]);
impl<T, N: Unsigned> std::ops::Deref for KdTree<T, N> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        &self.1
    }
}
impl<T: Clone, N: Unsigned> std::borrow::ToOwned for KdTree<T, N> {
    type Owned = KdTreeBuf<T, N>;
    fn to_owned(&self) -> Self::Owned {
        KdTreeBuf(PhantomData, self.1.to_vec())
    }
}
impl<T, N: Unsigned> KdTree<T, N> {
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
    /// let kdtree = kd_tree::KdTree3::sort_by(&mut items, |item1, item2, k| item1.point[k].cmp(&item2.point[k]));
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
    /// let kdtree = kd_tree::KdTree3::sort_by_key(&mut items, |item, k| OrderedFloat(item.point[k]));
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
    /// let mut items: Vec<[f64; 3]> = vec![[1.0, 2.0, 3.0], [3.0, 1.0, 2.0], [2.0, 3.0, 1.0]];
    /// let kdtree = kd_tree::KdTree3::sort_by_ordered_float(&mut items);
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
    /// let mut items: Vec<[i32; 3]> = vec![[1, 2, 3], [3, 1, 2], [2, 3, 1]];
    /// let kdtree = kd_tree::KdTree3::sort(&mut items);
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
    /// let kdtree = kd_tree::KdTree3::sort_by_key(&mut items, |item, k| OrderedFloat(item.point[k]));
    /// assert_eq!(kdtree.nearest_by(&[3.1, 0.9, 2.1], |item, k| item.point[k]).item.id, 222);
    /// ```
    pub fn nearest_by<Q: KdPoint>(
        &self,
        query: &Q,
        coord: impl Fn(&T, usize) -> Q::Scalar + Copy,
    ) -> Nearest<T, Q::Scalar> {
        kd_nearest_by(self.items(), query, coord)
    }

    /// # Example
    /// ```
    /// let mut items: Vec<[i32; 3]> = vec![[1, 2, 3], [3, 1, 2], [2, 3, 1]];
    /// let kdtree = kd_tree::KdTree3::sort(&mut items);
    /// assert_eq!(kdtree.nearest(&[3, 1, 2]).item, &[3, 1, 2]);
    /// ```
    pub fn nearest(
        &self,
        query: &impl KdPoint<Scalar = T::Scalar, Dim = T::Dim>,
    ) -> Nearest<T, T::Scalar>
    where
        T: KdPoint,
    {
        kd_nearest(self.items(), query)
    }

    /// # Example
    /// ```
    /// let kdtree = kd_tree::KdTreeBuf3::build(vec![[1, 2, 3], [3, 1, 2], [2, 3, 1]]);
    /// let key = [3, 1, 2];
    /// assert_eq!(kdtree.nearest_with(|p, k| key[k] - p[k]).item, &[3, 1, 2]);
    /// ```
    pub fn nearest_with<Scalar>(
        &self,
        kd_difference: impl Fn(&T, usize) -> Scalar + Copy,
    ) -> Nearest<T, Scalar>
    where
        Scalar: num_traits::NumAssign + Copy + PartialOrd,
    {
        kd_nearest_with(self.items(), N::to_usize(), kd_difference)
    }
}

/// An owned kd-tree.
/// This type implements [`std::ops::Deref`] to [`KdTree`].
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct KdTreeBuf<T, N: Unsigned>(PhantomData<N>, Vec<T>);
impl<T, N: Unsigned> std::ops::Deref for KdTreeBuf<T, N> {
    type Target = KdTree<T, N>;
    fn deref(&self) -> &Self::Target {
        unsafe { KdTree::new_unchecked(&self.1) }
    }
}
impl<T, N: Unsigned> AsRef<KdTree<T, N>> for KdTreeBuf<T, N> {
    fn as_ref(&self) -> &KdTree<T, N> {
        self
    }
}
impl<T, N: Unsigned> std::borrow::Borrow<KdTree<T, N>> for KdTreeBuf<T, N> {
    fn borrow(&self) -> &KdTree<T, N> {
        self
    }
}
impl<T, N: Unsigned> Into<Vec<T>> for KdTreeBuf<T, N> {
    fn into(self) -> Vec<T> {
        self.1
    }
}
impl<T, N: Unsigned> KdTreeBuf<T, N> {
    pub fn into_vec(self) -> Vec<T> {
        self.1
    }

    /// # Example
    /// ```
    /// struct Item {
    ///     point: [i32; 3],
    ///     id: usize,
    /// }
    /// let kdtree = kd_tree::KdTreeBuf3::build_by(
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
    /// let kdtree = kd_tree::KdTreeBuf3::build_by_key(
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
    /// let kdtree = kd_tree::KdTreeBuf3::build_by_ordered_float(vec![
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
    /// let kdtree = kd_tree::KdTreeBuf3::build(vec![[1, 2, 3], [3, 1, 2], [2, 3, 1]]);
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

macro_rules! define_kdtree_aliases {
    ($($dim:literal),*) => {
        $(
            paste::paste! {
                pub type [<KdTree $dim>]<T> = KdTree<T, typenum::[<U $dim>]>;
                pub type [<KdTreeBuf $dim>]<T> = KdTreeBuf<T, typenum::[<U $dim>]>;
            }
        )*
    };
}
define_kdtree_aliases!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);

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
