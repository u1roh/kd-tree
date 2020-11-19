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
        <Self::Dim as typenum::Unsigned>::to_usize()
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

    pub fn sort_points_by<Key: Ord, F>(points: &'a mut [T], f: F) -> Self
    where
        T: KdPoint<Dim = N>,
        F: Fn(T::Scalar) -> Key,
    {
        Self::sort_by_key(points, |item, k| f(item.at(k)))
    }

    pub fn sort_points(points: &'a mut [T]) -> Self
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

    pub fn from_points_by<Key: Ord, F>(points: Vec<T>, f: F) -> Self
    where
        T: KdPoint<Dim = N>,
        F: Fn(T::Scalar) -> Key,
    {
        Self::construct_by_key(points, |item, k| f(item.at(k)))
    }

    pub fn from_points(points: Vec<T>) -> Self
    where
        T: KdPoint<Dim = N>,
        T::Scalar: Ord,
    {
        Self::construct_by_key(points, |item, k| item.at(k))
    }
}

pub type KdTreeRef<'a, T, N> = KdTree<&'a [T], N>;
pub type KdTreeVec<T, N> = KdTree<Vec<T>, N>;

macro_rules! define_kdtree_aliases {
    ($($dim:literal),*) => {
        $(
            paste::paste! {
                pub type [<KdTree $dim>]<Collection> = KdTree<Collection, typenum::[<U $dim>]>;
                pub type [<KdTreeRef $dim>]<'a, T> = KdTree<&'a [T], typenum::[<U $dim>]>;
                pub type [<KdTreeVec $dim>]<T> = KdTree<Vec<T>, typenum::[<U $dim>]>;
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
