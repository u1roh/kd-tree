pub mod array;
pub mod kd;
use std::cmp::Ordering;
use std::marker::PhantomData;

pub trait Point {
    type Scalar: num_traits::NumAssign + Copy + PartialOrd;
    type Dim: typenum::Unsigned + typenum::NonZero;
    const DIM: usize;
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

#[derive(Debug)]
pub struct Nearest<'a, T, Scalar> {
    pub item: &'a T,
    pub distance: Scalar,
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
        query: &impl Point<Scalar = <C::Item as Point>::Scalar, Dim = <C::Item as Point>::Dim>,
    ) -> Nearest<C::Item, <C::Item as Point>::Scalar>
    where
        C::Item: Point,
    {
        kd::kd_nearest(self.items(), query)
    }
    pub fn nearest_by<Q: Point>(
        &self,
        query: &Q,
        coord: impl Fn(&C::Item, usize) -> Q::Scalar + Copy,
    ) -> Nearest<C::Item, Q::Scalar> {
        kd::kd_nearest_by(self.items(), query, coord)
    }
}
impl<'a, T, N: typenum::Unsigned + typenum::NonZero> KdTree<&'a [T], N> {
    pub fn sort_points(points: &'a mut [T]) -> Self
    where
        T: Point<Dim = N>,
        T::Scalar: Ord,
    {
        Self::sort_by_key(points, |item, k| item.at(k))
    }
    pub fn sort_points_by<Key: Ord>(points: &'a mut [T], f: impl Fn(T::Scalar) -> Key) -> Self
    where
        T: Point<Dim = N>,
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
        kd::kd_sort_by(items, N::to_usize(), compare);
        Self(items, PhantomData)
    }
}

macro_rules! impl_points {
    ($($len:literal),*) => {
        $(
            paste::paste!{
                impl<T: num_traits::NumAssign + Copy + PartialOrd> Point for [T; $len] {
                    type Scalar = T;
                    type Dim = typenum::[<U $len>];
                    const DIM: usize = $len;
                    fn at(&self, i: usize) -> T { self[i] }
                }
            }
        )*
    };
}

impl_points!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
