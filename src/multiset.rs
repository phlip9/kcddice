use crate::u32_sum_all_nibs;
#[cfg(test)]
use proptest::{
    arbitrary::{any, Arbitrary},
    strategy::{BoxedStrategy, Strategy},
};
use std::{
    cmp, fmt,
    hash::{Hash, Hasher},
    ops,
};

/// A compressed representation of a multiset (a set with potential duplicates
/// of the same item), that can store up to 8 entries with individual counts
/// in the range `0..=15`.
///
/// `MultisetU4x8` is laid out like `0x7654_3210`, where each nibble `X` is the
/// number of items at index `0 <= X < 8` in the multiset.
#[repr(transparent)]
#[derive(Copy, Clone, Eq)]
pub struct MultisetU4x8(u32);

impl MultisetU4x8 {
    /// A new empty set of counts.
    #[inline]
    pub const fn new() -> Self {
        Self(0)
    }

    #[inline]
    pub const fn from_counts(cs: [u8; 8]) -> Self {
        Self(u32::from_le_bytes([
            cs[0] | (cs[1] << 4),
            cs[2] | (cs[3] << 4),
            cs[4] | (cs[5] << 4),
            cs[6] | (cs[7] << 4),
        ]))
    }

    #[inline]
    pub const fn from_count(idx: u8, count: u8) -> Self {
        Self((count as u32) << (4 * (idx as u32)))
    }

    #[inline]
    pub const fn into_counts(self) -> [u8; 8] {
        [
            self.get_count(0),
            self.get_count(1),
            self.get_count(2),
            self.get_count(3),
            self.get_count(4),
            self.get_count(5),
            self.get_count(6),
            self.get_count(7),
        ]
    }

    #[inline]
    pub fn as_u32(self) -> u32 {
        self.0
    }

    #[inline]
    pub fn is_empty(self) -> bool {
        self == Self::new()
    }

    #[inline]
    pub fn len(self) -> u8 {
        // the length of a multiset is the sum of the counts of each element.
        // since we represent each elements' count as a nibble packed into a
        // u32, the total length is the sum of all the nibbles.
        u32_sum_all_nibs(self.0) as u8
    }

    #[inline]
    pub const fn get_count(self, idx: u8) -> u8 {
        // debug_assert!((0..8).contains(&idx));

        (self.0 >> (4 * (idx as u32)) & 0x0f) as u8
    }

    #[cfg(test)]
    #[inline]
    pub fn set_count(&mut self, idx: u8, count: u8) {
        debug_assert!((0..8).contains(&idx));
        debug_assert!((0..8).contains(&count));

        self.0 = (self.0 & !(0x0f << (4 * (idx as u32))))
            + (((count as u32) & 0x0f) << (4 * (idx as u32)));
    }

    pub fn is_superset_of(self, other: Self) -> bool {
        (0..8).all(|idx| self.get_count(idx) >= other.get_count(idx))
    }

    #[cfg(test)]
    pub fn into_iter_flat(self) -> impl Iterator<Item = u8> {
        (0..8).flat_map(move |idx| std::iter::repeat(idx).take(self.get_count(idx) as usize))
    }

    pub fn from_iter_flat(iter: impl Iterator<Item = u8>) -> Self {
        iter.map(|idx| Self::from_count(idx, 1))
            .fold(Self::new(), |acc, single| acc + single)
    }

    pub fn into_iter(self) -> impl Iterator<Item = (u8, u8)> {
        self.into_counts()
            .into_iter()
            .enumerate()
            .map(|(idx, count)| (idx as u8, count))
    }
}

impl FromIterator<(u8, u8)> for MultisetU4x8 {
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = (u8, u8)>,
    {
        iter.into_iter().fold(Self::new(), |acc, (idx, count)| {
            acc + MultisetU4x8::from_count(idx, count)
        })
    }
}

impl fmt::Debug for MultisetU4x8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let entries = self.into_iter().filter(|(_idx, count)| count > &0);
        f.debug_map().entries(entries).finish()
    }
}

impl ops::Add for MultisetU4x8 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl ops::AddAssign for MultisetU4x8 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.0.add_assign(rhs.0)
    }
}

impl ops::Sub for MultisetU4x8 {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl ops::SubAssign for MultisetU4x8 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.0.sub_assign(rhs.0)
    }
}

impl cmp::PartialEq for MultisetU4x8 {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_u32().eq(&other.as_u32())
    }
}

#[cfg(test)]
impl cmp::Ord for MultisetU4x8 {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.into_iter_flat().cmp(other.into_iter_flat())
    }
}

#[cfg(test)]
impl cmp::PartialOrd for MultisetU4x8 {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Hash for MultisetU4x8 {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u32(self.as_u32())
    }
}

#[cfg(test)]
impl Arbitrary for MultisetU4x8 {
    type Parameters = ();
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
        any::<u32>().prop_map(Self).boxed()
    }
}

// #[cfg(test)]
// mod test {
//     use super::*;
//     use proptest::prelude::*;
//
//     #[test]
//     fn test_mset_u4x8_ord() {
//         proptest!(|(mset1 in any::<MultisetU4x8>(), mset2 in any::<MultisetU4x8>())| {
//             prop_assert_eq!(mset1.cmp(&mset2), mset1.fast_cmp(&mset2));
//         });
//     }
// }
