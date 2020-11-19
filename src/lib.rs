mod nearest;
mod sort;
pub use nearest::*;
pub use sort::*;
use std::cmp::Ordering;
use std::marker::PhantomData;

pub trait KdPoint {
    type Scalar: num_traits::NumAssign + Copy + PartialOrd;
    type Dim: typenum::Unsigned + typenum::NonZero;
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

pub struct KdTree<Collection, N>(Collection, PhantomData<N>);
pub type KdTree2<Collection> = KdTree<Collection, typenum::U2>;
pub type KdTree3<Collection> = KdTree<Collection, typenum::U3>;

impl<C: KdCollection, N> KdTree<C, N> {
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
impl<'a, T, N: typenum::Unsigned + typenum::NonZero> KdTree<&'a [T], N> {
    pub fn sort_points(points: &'a mut [T]) -> Self
    where
        T: KdPoint<Dim = N>,
        T::Scalar: Ord,
    {
        Self::sort_by_key(points, |item, k| item.at(k))
    }
    pub fn sort_points_by<Key: Ord>(points: &'a mut [T], f: impl Fn(T::Scalar) -> Key) -> Self
    where
        T: KdPoint<Dim = N>,
    {
        Self::sort_by_key(points, |item, k| f(item.at(k)))
    }
    pub fn sort_by_key<Key: Ord>(
        items: &'a mut [T],
        kd_key: impl Fn(&T, usize) -> Key + Copy,
    ) -> Self {
        Self::sort_by(items, |item1, item2, k| {
            kd_key(item1, k).cmp(&kd_key(item2, k))
        })
    }
    pub fn sort_by(items: &'a mut [T], compare: impl Fn(&T, &T, usize) -> Ordering + Copy) -> Self {
        kd_sort_by(items, N::to_usize(), compare);
        Self(items, PhantomData)
    }
}

macro_rules! impl_points {
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

impl_points!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
