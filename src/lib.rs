pub mod array;
pub mod kd;
use std::cmp::Ordering;

pub trait Point {
    type Scalar: num_traits::NumAssign + Copy + PartialOrd;
    type Dim: typenum::Unsigned + typenum::NonZero;
    const DIM: usize;
    fn at(&self, i: usize) -> Self::Scalar;
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

pub struct KdTree<'a, T: Point>(&'a [T]);

impl<'a, T: Point> KdTree<'a, T> {
    pub fn sort_by(
        source: &'a mut [T],
        compare: impl Fn(T::Scalar, T::Scalar) -> Ordering + Copy,
    ) -> Self {
        kd::kd_sort_by(source, T::DIM, |item1, item2, k| {
            compare(item1.at(k), item2.at(k))
        });
        Self(source)
    }
    pub fn sort_by_key<Key: Ord>(
        source: &'a mut [T],
        key: impl Fn(T::Scalar) -> Key + Copy,
    ) -> Self {
        Self::sort_by(source, |a, b| key(a).cmp(&key(b)))
    }
    pub fn sort(source: &'a mut [T]) -> Self
    where
        T::Scalar: Ord,
    {
        kd::kd_sort(source);
        Self(source)
    }
    pub fn nearest(
        &self,
        query: &impl Point<Dim = T::Dim, Scalar = T::Scalar>,
    ) -> kd::Nearest<'a, T, T::Scalar> {
        kd::kd_nearest_by(self.0, query, |p, k| p.at(k))
    }
}
