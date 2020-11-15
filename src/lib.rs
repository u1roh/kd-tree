pub mod array;
pub mod kd;

pub trait Point {
    type Scalar: num_traits::NumAssign + Copy + PartialOrd;
    const DIM: usize;
    fn at(&self, i: usize) -> Self::Scalar;
}

macro_rules! impl_points {
    ($($len:literal),*) => {
        $(
            impl<T: num_traits::NumAssign + Copy + PartialOrd> Point for [T; $len] {
                type Scalar = T;
                const DIM: usize = $len;
                fn at(&self, i: usize) -> T { self[i] }
            }
        )*
    };
}

impl_points!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
