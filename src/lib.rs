pub mod array;
pub mod kd;

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

pub trait KdPointAccess<T = Self> {
    type Scalar: num_traits::NumAssign + Copy + PartialOrd;
    type Dim: typenum::Unsigned + typenum::NonZero;
    fn at(p: &T, i: usize) -> Self::Scalar;
}

fn kd_sort<T, A: KdPointAccess<T>>(items: &mut [T]) {
    A::at(&items[0], 0);
    unimplemented!()
}
