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
pub use nearest::*;
pub use sort::*;
use std::cmp::Ordering;
use std::marker::PhantomData;
use typenum::Unsigned;

pub trait KdPoint {
    type Scalar: num_traits::NumAssign + Copy + PartialOrd;
    type Dim: Unsigned;
    fn dim() -> usize {
        <Self::Dim as Unsigned>::to_usize()
    }
    fn at(&self, i: usize) -> Self::Scalar;
}

pub trait KdCollection {
    type Item;
    fn items(&self) -> &[Self::Item];
}
impl<'a, T> KdCollection for &'a [T] {
    type Item = T;
    fn items(&self) -> &[Self::Item] {
        self
    }
}
impl<T> KdCollection for Vec<T> {
    type Item = T;
    fn items(&self) -> &[Self::Item] {
        self
    }
}

pub struct KdTree<T, N: Unsigned>(PhantomData<N>, [T]);
impl<T, N: Unsigned> std::ops::Deref for KdTree<T, N> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        &self.1
    }
}
impl<T, N: Unsigned> KdTree<T, N> {
    pub fn items(&self) -> &[T] {
        &self.1
    }

    unsafe fn new_unchecked(items: &[T]) -> &Self {
        &*(items as *const _ as *const Self)
    }
    pub fn sort_by<F>(items: &mut [T], compare: F) -> &Self
    where
        F: Fn(&T, &T, usize) -> Ordering + Copy,
    {
        kd_sort_by(items, N::to_usize(), compare);
        unsafe { Self::new_unchecked(items) }
    }

    pub fn sort_by_key<Key: Ord, F>(items: &mut [T], kd_key: F) -> &Self
    where
        F: Fn(&T, usize) -> Key + Copy,
    {
        Self::sort_by(items, |item1, item2, k| {
            kd_key(item1, k).cmp(&kd_key(item2, k))
        })
    }

    pub fn sort_by_ordered_float(points: &mut [T]) -> &Self
    where
        T: KdPoint<Dim = N>,
        T::Scalar: num_traits::Float,
    {
        Self::sort_by_key(points, |item, k| ordered_float::OrderedFloat(item.at(k)))
    }

    pub fn sort(points: &mut [T]) -> &Self
    where
        T: KdPoint<Dim = N>,
        T::Scalar: Ord,
    {
        Self::sort_by_key(points, |item, k| item.at(k))
    }

    pub fn nearest_by<Q: KdPoint>(
        &self,
        query: &Q,
        coord: impl Fn(&T, usize) -> Q::Scalar + Copy,
    ) -> Nearest<T, Q::Scalar> {
        kd_nearest_by(self.items(), query, coord)
    }

    pub fn nearest(
        &self,
        query: &impl KdPoint<Scalar = T::Scalar, Dim = T::Dim>,
    ) -> Nearest<T, T::Scalar>
    where
        T: KdPoint,
    {
        kd_nearest(self.items(), query)
    }
}

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
impl<T, N: Unsigned> Into<Vec<T>> for KdTreeBuf<T, N> {
    fn into(self) -> Vec<T> {
        self.1
    }
}
impl<T, N: Unsigned> KdTreeBuf<T, N> {
    pub fn into_vec(self) -> Vec<T> {
        self.1
    }

    pub fn build_by<F>(mut items: Vec<T>, compare: F) -> Self
    where
        F: Fn(&T, &T, usize) -> Ordering + Copy,
    {
        kd_sort_by(&mut items, N::to_usize(), compare);
        Self(PhantomData, items)
    }

    pub fn build_by_key<Key, F>(items: Vec<T>, kd_key: F) -> Self
    where
        Key: Ord,
        F: Fn(&T, usize) -> Key + Copy,
    {
        Self::build_by(items, |item1, item2, k| {
            kd_key(item1, k).cmp(&kd_key(item2, k))
        })
    }

    pub fn build_by_ordered_float(points: Vec<T>) -> Self
    where
        T: KdPoint<Dim = N>,
        T::Scalar: num_traits::Float,
    {
        Self::build_by_key(points, |item, k| ordered_float::OrderedFloat(item.at(k)))
    }

    pub fn build(points: Vec<T>) -> Self
    where
        T: KdPoint<Dim = N>,
        T::Scalar: Ord,
    {
        Self::build_by_key(points, |item, k| item.at(k))
    }
}

/*
pub struct KdTree<Collection: KdCollection, N: Unsigned>(Collection, PhantomData<N>);

impl<C: KdCollection, N: Unsigned> KdTree<C, N> {
    pub fn items(&self) -> &[C::Item] {
        self.0.items()
    }

    pub fn nearest(
        &self,
        query: &impl KdPoint<Scalar = <C::Item as KdPoint>::Scalar, Dim = <C::Item as KdPoint>::Dim>,
    ) -> Nearest<C::Item, <C::Item as KdPoint>::Scalar>
    where
        C::Item: KdPoint,
    {
        kd_nearest(self.items(), query)
    }

    pub fn nearest_by<Q: KdPoint>(
        &self,
        query: &Q,
        coord: impl Fn(&C::Item, usize) -> Q::Scalar + Copy,
    ) -> Nearest<C::Item, Q::Scalar> {
        kd_nearest_by(self.items(), query, coord)
    }
}

impl<'a, T, N: Unsigned> KdTree<&'a [T], N> {
    pub fn into_slice(self) -> &'a [T] {
        self.0
    }

    pub fn sort_by<F>(items: &'a mut [T], compare: F) -> Self
    where
        F: Fn(&T, &T, usize) -> Ordering + Copy,
    {
        kd_sort_by(items, N::to_usize(), compare);
        Self(items, PhantomData)
    }

    pub fn sort_by_key<Key: Ord, F>(items: &'a mut [T], kd_key: F) -> Self
    where
        F: Fn(&T, usize) -> Key + Copy,
    {
        Self::sort_by(items, |item1, item2, k| {
            kd_key(item1, k).cmp(&kd_key(item2, k))
        })
    }

    pub fn sort_by_ordered_float(points: &'a mut [T]) -> Self
    where
        T: KdPoint<Dim = N>,
        T::Scalar: num_traits::Float,
    {
        Self::sort_by_key(points, |item, k| ordered_float::OrderedFloat(item.at(k)))
    }

    pub fn sort(points: &'a mut [T]) -> Self
    where
        T: KdPoint<Dim = N>,
        T::Scalar: Ord,
    {
        Self::sort_by_key(points, |item, k| item.at(k))
    }
}

impl<T, N: Unsigned> KdTree<Vec<T>, N> {
    pub fn into_vec(self) -> Vec<T> {
        self.0
    }

    pub fn construct_by<F>(mut items: Vec<T>, compare: F) -> Self
    where
        F: Fn(&T, &T, usize) -> Ordering + Copy,
    {
        kd_sort_by(&mut items, N::to_usize(), compare);
        Self(items, PhantomData)
    }

    pub fn construct_by_key<Key, F>(items: Vec<T>, kd_key: F) -> Self
    where
        Key: Ord,
        F: Fn(&T, usize) -> Key + Copy,
    {
        Self::construct_by(items, |item1, item2, k| {
            kd_key(item1, k).cmp(&kd_key(item2, k))
        })
    }

    pub fn construct_by_ordered_float(points: Vec<T>) -> Self
    where
        T: KdPoint<Dim = N>,
        T::Scalar: num_traits::Float,
    {
        Self::construct_by_key(points, |item, k| ordered_float::OrderedFloat(item.at(k)))
    }

    pub fn construct(points: Vec<T>) -> Self
    where
        T: KdPoint<Dim = N>,
        T::Scalar: Ord,
    {
        Self::construct_by_key(points, |item, k| item.at(k))
    }
}

pub type KdTreeRef<'a, T, N> = KdTree<&'a [T], N>;
pub type KdTreeVec<T, N> = KdTree<Vec<T>, N>;
*/

macro_rules! define_kdtree_aliases {
    ($($dim:literal),*) => {
        $(
            paste::paste! {
                //pub type [<KdTree $dim>]<Collection> = KdTree<Collection, typenum::[<U $dim>]>;
                //pub type [<KdTreeRef $dim>]<'a, T> = KdTree<&'a [T], typenum::[<U $dim>]>;
                //pub type [<KdTreeVec $dim>]<T> = KdTree<Vec<T>, typenum::[<U $dim>]>;
                pub type [<KdTree $dim>]<T> = KdTree<T, typenum::[<U $dim>]>;
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
