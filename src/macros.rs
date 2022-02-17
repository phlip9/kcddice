macro_rules! cfg_test {
    ($($item:item)*) => {
        $(
            #[cfg(test)]
            $item
        )*
    }
}

macro_rules! impl_total_size_static {
    ( $($t:ty),+ $(,)? ) => {
        $( impl $crate::TotalSize for $t {
            #[inline]
            fn static_size() -> Option<usize> {
                Some(std::mem::size_of::<Self>())
            }
        })+
    }
}
