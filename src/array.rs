pub trait Array {
    type Element;
    const LEN: usize;
    fn at(&self, i: usize) -> &Self::Element;
}

macro_rules! impl_arrays {
    ($($len:literal),*) => {
        $(
            impl<T: Copy> Array for [T; $len] {
                type Element = T;
                const LEN: usize = $len;
                fn at(&self, i: usize) -> &T { &self[i] }
            }
        )*
    };
}

impl_arrays!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
