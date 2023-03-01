#![cfg(feature = "nalgebra")]
use super::KdPoint;

macro_rules! impl_kdpoint_for_nalgebra_point {
    ($($dim:literal),*) => {
        $(
            paste::paste! {
                impl<Scalar> KdPoint<$dim> for nalgebra::Point<Scalar, $dim>
                where
                    Scalar: num_traits::NumAssign + Copy + PartialOrd + nalgebra::Scalar,
                {
                    type Scalar = Scalar;

                    fn at(&self, k: usize) -> Scalar {
                        self[k]
                    }
                }
            }
        )*
    };
}

macro_rules! impl_kdpoint_for_nalgebra_vector {
    ($($dim:literal),*) => {
        $(
            paste::paste! {
                impl<Scalar, Storage> KdPoint<$dim> for nalgebra::Vector<Scalar, nalgebra::Const<$dim>, Storage>
                where
                    Scalar: num_traits::NumAssign + Copy + PartialOrd + nalgebra::Scalar,
                    Storage: nalgebra::StorageMut<Scalar, nalgebra::Const<$dim>>
                {
                    type Scalar = Scalar;

                    fn at(&self, k: usize) -> Scalar {
                        self[k]
                    }
                }
            }
        )*
    };
}

impl_kdpoint_for_nalgebra_point!(1, 2, 3, 4, 5, 6, 7, 8);
impl_kdpoint_for_nalgebra_vector!(1, 2, 3, 4, 5, 6, 7, 8);
