//! # kcddice
//!
//! A simple utility for optimally playing the Kingdom Come: Deliverance dice
//! game : )
//!
//! ## Rules
//!
//! https://kingdom-come-deliverance.fandom.com/wiki/Dice
//!
//! ## Explanation
//!
//! More specifically, this tool takes as input the currently rolled dice on the
//! board and outputs all possible actions along with their expected values and
//! bust probabilities.
//!
//! The expected value in this case is the expectation of the turn total assuming
//! the player plays to maximize their expected value.
//!
//! The bust probability is the probability that no scoring dice are rolled in
//! the next turn, assuming the player choose the given action.

#[macro_use]
mod macros;

mod agent;
mod multiset;
pub mod cli;
pub mod dice;
pub mod parse;
pub mod search;

use std::{cmp, ops::Deref, rc::Rc, collections::HashMap};

pub(crate) const DEFAULT_TOTAL_SCORE: u16 = 0;
pub(crate) const DEFAULT_MAX_SCORE: u16 = 4000;

///////////////////
// Combinatorics //
///////////////////

/// The number of factorials to precompute in our static lookup table. Note this
/// number is chosen so as not to overflow a u32.
pub(crate) const NUM_FACTORIALS: usize = 13;

/// A precomputed lookup table of factorials from `0 <= n < NUM_FACTORIALS`.
/// `FACTORIAL_LT[n] = n!`.
const FACTORIAL_LT: [u32; NUM_FACTORIALS] = precompute_factorials();

const fn precompute_factorials() -> [u32; NUM_FACTORIALS] {
    let mut factorials: [u32; NUM_FACTORIALS] = [1; NUM_FACTORIALS];

    // need ghetto for-loop in const fn...
    let mut idx = 1;
    loop {
        if idx >= NUM_FACTORIALS {
            break;
        }
        factorials[idx] = (idx as u32) * factorials[idx - 1];
        idx += 1;
    }

    factorials
}

pub(crate) const fn factorial(n: u32) -> u32 {
    FACTORIAL_LT[n as usize]
}

/// count `n choose k` without replacement.
pub(crate) const fn num_combinations(n: u32, k: u32) -> u32 {
    factorial(n) / (factorial(k) * factorial(n - k))
}

/// count `n choose k` with replacement. also known as `n multichoose k`.
#[inline]
pub(crate) const fn num_multisets(n: u32, k: u32) -> u32 {
    num_combinations(n + k - 1, k)
}

////////////////////////////
// Unstable std functions //
////////////////////////////

/// A total ordering on f64's. Needed until `f64::total_cmp` is stabilized.
/// See: (https://doc.rust-lang.org/stable/std/primitive.f64.html#method.total_cmp)
#[inline]
pub(crate) fn total_cmp_f64(a: &f64, b: &f64) -> cmp::Ordering {
    let mut left = a.to_bits() as i64;
    let mut right = b.to_bits() as i64;

    left ^= (((left >> 63) as u64) >> 1) as i64;
    right ^= (((right >> 63) as u64) >> 1) as i64;

    left.cmp(&right)
}

/// Returns `true` if the iterator `iter` is sorted, according to the comparator
/// function `compare`, i.e., `x_1 <= x2 <= ... <= x_n`.
// TODO(philiphayes): use `std::slice::is_sorted_by` when it stabilizes.
pub(crate) fn is_sorted_by<T, F>(mut iter: impl Iterator<Item = T>, mut compare: F) -> bool
where
    F: FnMut(&T, &T) -> Option<cmp::Ordering>,
{
    let mut prev = match iter.next() {
        Some(first) => first,
        None => return true,
    };

    for next in iter {
        if let Some(cmp::Ordering::Greater) | None = compare(&prev, &next) {
            return false;
        }
        prev = next;
    }

    true
}

/// Returns `true` if the iterator `iter` is totally ordered, according to the
/// comparator function `compare`, i.e., `x_1 < x2 < ... < x_n`.
pub(crate) fn is_total_order_by<T, F>(mut iter: impl Iterator<Item = T>, mut compare: F) -> bool
where
    F: FnMut(&T, &T) -> Option<cmp::Ordering>,
{
    let mut prev = match iter.next() {
        Some(first) => first,
        None => return true,
    };

    for next in iter {
        if let Some(cmp::Ordering::Greater) | Some(cmp::Ordering::Equal) | None =
            compare(&prev, &next)
        {
            return false;
        }
        prev = next;
    }

    true
}

pub(crate) fn is_partitioned<T>(
    mut iter: impl Iterator<Item = T>,
    mut pred: impl FnMut(T) -> bool,
) -> bool {
    iter.all(&mut pred) || !iter.any(pred)
}

// `x / y`, rounded up
pub(crate) const fn usize_div_ceil(x: usize, y: usize) -> usize {
    let d = x / y;
    let r = x % y;
    if r > 0 && y > 0 {
        d + 1
    } else {
        d
    }
}

///////////////
// Bit Hacks //
///////////////

/// Returns `true` if `x` has _any_ nibbles `nb` in the range `m < nb < n`. The
/// nibbles considered are selected according to `mask`, where `mask` has a `1`
/// in each selectable nibble.
///
/// For example, calling this with `mask = 0x0010_1101` will only look at the
/// 1st, 3rd, 4th, and 6th nibbles. Calling with `mask = 0x1111_1111` will look
/// at all nibbles. Technically you can also look at "unaligned" nibbles too :)
///
/// See [`u32_any_nibs_between_ref`] for a human-readable implementation.
#[inline]
pub(crate) fn u32_any_nibs_between(x: u32, mask: u32, m: u32, n: u32) -> bool {
    debug_assert!((0..=7).contains(&m));
    debug_assert!((0..=8).contains(&n));

    let a = mask * 7;
    let b = mask * 8;
    let w = mask * (7 + n);
    let t = mask * (7 - m);

    let u = x & a;
    let z = (w - u) & (!x) & (u + t) & b;

    z != 0
}

/// Sum all nibbles in `x`.
#[inline]
pub(crate) fn u32_sum_all_nibs(x: u32) -> u32 {
    // a mask that selects the lo nibble in each byte.
    const NIBS_0246: u32 = 0x0f0f_0f0f;

    // horizontal sum hi and lo nibbles in each byte, placing in the lo nibble.
    let y = (x & NIBS_0246) + ((x >> 4) & NIBS_0246);

    // if y = [y0, y1, y2, y3] bytes and each byte b is in the range 0 <= b < 64,
    // then multiplying by 0x0101_0101 will yield
    // z = [y0, y0 + y1, y0 + y1 + y2, y0 + y1 + y2 + y3] without any overflows.
    //
    // since each byte in y is the sum of two nibbles nb where 0 <= nb < 16, it
    // follows that 0 <= b = nb_lo + nb_hi < 32 < 64, so we won't have any
    // overflows.
    let z = y.wrapping_mul(0x0101_0101);

    // select the last byte in z, which contains our desired sum:
    // z3 = y0 + y1 + y2 + y3
    //    = (nb0 + nb1) + (nb2 + nb3) + (nb4 + nb5) + (nb6 + nb7)
    z >> 24
}

/// Return the number of leading bytes == 0x00
#[inline]
pub(crate) fn u64_leading_zero_bytes(x: u64) -> u32 {
    x.leading_zeros() >> 3
}

/// Return the number of trailing bytes == 0x00
#[inline]
pub(crate) fn u64_trailing_zero_bytes(x: u64) -> u32 {
    x.trailing_zeros() >> 3
}

/// Do an element-wise less-than comparison across each byte in `x` and `y`.
/// Returns a mask where each byte is `0x80` if `bx < by` and `0x00` otherwise.
/// Requires that each byte in `x` is in the range `0 <= bx <= 0x7f (127)`.
#[inline]
pub(crate) fn u64_elementwise_bytes_lt(x: u64, y: u64) -> u64 {
    // ensure no elements in x are > 0x7f
    debug_assert!((x & 0x8080_8080_8080_8080) == 0);

    let a = 0x7f7f_7f7f_7f7f_7f7f_u64;
    let b = 0x8080_8080_8080_8080_u64;
    let c = a - x;

    // a = (0x7f << 56) + (0x7f << 48) + .. + (0x7f << 0)
    // x = (b7 << 56) + (b6 << 48) + .. + (b0 << 0)
    // c = ((0x7f - b7) << 56) + .. + ((0x7f - b0) << 0)  iff. bi <= 0x7f forall i

    // (((by & 0x7f) + (0x7f - bx) > 0x7f) || (by > 0x7f)
    // by' + 0x7f - bx > 0x7f
    // by' - bx > 0x7f
    // (by' > bx) || (by > 0x7f)

    // convert 0x80 mask z to 0xff mask
    // (z / 0x80) * 0xff (convert 0x80 -> 0x01 mask, then 0x01 -> 0xff mask)
    // = (z >> 7) * 0xff (replace div with shr)

    ((y & a).wrapping_add(c) | y) & b
}

/// Given two u64's, `x` and `y`, where all bytes `bx` in `x` are in the range
/// `0 <= bx <= 0x7f (127)`, return the _byte index_ (i.e., idx = 0 = least
/// significant byte, idx = 7 = most-significant byte)  of the first leading byte
/// `by` in `y` where `by > bx`. (i.e., starting from the most-significant byte).
///
/// Returns `None` if there are no bytes in `y` greater than their corresponding
/// byte in `x`.
#[inline]
pub(crate) fn u64_leading_byte_idx_lt(x: u64, y: u64) -> Option<u8> {
    let mask_0x80 = u64_elementwise_bytes_lt(x, y);

    if mask_0x80 != 0 {
        Some((7 - u64_leading_zero_bytes(mask_0x80)) as u8)
    } else {
        None
    }
}

/// Return the _byte index_ of the first _trailing_ byte `by` in `y` where
/// `n < by`.
#[inline]
pub(crate) fn u64_trailing_byte_idx_lt(n: u8, y: u64) -> Option<u8> {
    // broadcast byte `n` across all bytes
    let x = 0x0101_0101_0101_0101_u64 * (n as u64);
    // set 0x80 in all bytes where `n < by`
    let mask_0x80 = u64_elementwise_bytes_lt(x, y);

    if mask_0x80 != 0 {
        Some(u64_trailing_zero_bytes(mask_0x80) as u8)
    } else {
        None
    }
}

/////////////////////
// TotalSize trait //
/////////////////////

/// A trait for computing the total size of a data structure in memory. That means
/// not just the size on the stack, but also the total size of any owned resources.
pub trait TotalSize {
    /// Types whose total size is known statically, like a `u32`, can return a
    /// constant here. Unsized types or types that contain variable-size resources
    /// must return `None` here.
    fn static_size() -> Option<usize> {
        None
    }
    fn total_size(&self) -> usize {
        Self::static_size().unwrap()
    }
}

impl_total_size_static!(u16, u32, f64);

impl<T> TotalSize for Vec<T>
where
    T: TotalSize,
{
    fn total_size(&self) -> usize {
        let inner_size = if let Some(elt_size) = T::static_size() {
            self.len() * elt_size
        } else {
            self.iter()
                .map(|x| x.total_size())
                .sum()
        };
        std::mem::size_of::<Self>() + inner_size
    }
}

impl<T> TotalSize for Rc<T>
where
    T: TotalSize,
{
    fn total_size(&self) -> usize {
        if let Some(size) = Self::static_size() {
            size
        } else {
            std::mem::size_of::<Self>() + self.deref().total_size()
        }
    }
    fn static_size() -> Option<usize> {
        T::static_size().map(|inner_size| inner_size + std::mem::size_of::<Self>())
    }
}

impl<K, V> TotalSize for HashMap<K, V>
where
    K: TotalSize,
    V: TotalSize,
{
    fn total_size(&self) -> usize {
        let inner_size = match (K::static_size(), V::static_size()) {
            (Some(size_k), Some(size_v)) => self.len() * (size_k + size_v),
            (Some(size_k), None) => {
                (self.len() * size_k)
                    + self.values().map(|v| v.total_size()).sum::<usize>()
            }
            (None, Some(size_v)) => {
                (self.len() * size_v)
                    + self.keys().map(|k| k.total_size()).sum::<usize>()
            }
            (None, None) => self
                .iter()
                .map(|(k, v)| k.total_size() + v.total_size())
                .sum(),
        };
        std::mem::size_of::<Self>() + inner_size
    }
}

//////////
// util //
//////////

// slightly modified matrix transpose, xs.len() can be < nrow * ncol
pub(crate) fn transpose<T: Copy>(xs: &[T], ncol: usize) -> Vec<Option<T>> {
    let nrow = usize_div_ceil(xs.len(), ncol);

    let mut out = vec![None; nrow * ncol];
    for (idx, x) in xs.iter().enumerate() {
        // idx = r + (nrow * c)
        // idx / nrow = r / nrow + c = 0 + c = c
        // idx % nrow = (r % nrow) + 0 = r
        let (c, r) = (idx / nrow, idx % nrow);
        let idx2 = c + (ncol * r);
        out[idx2] = Some(*x);
    }
    out
}

pub fn dice_table(ncol: usize) -> Vec<(&'static str, &'static str)> {
    let mut dice = dice::DieKind::all().to_vec();
    // more compact table, but not alphabetical order... idk if I will use this
    // dice.sort_unstable_by_key(|(memnonic, name)| -((memnonic.len() + name.len()) as isize));
    dice.sort_unstable_by_key(|die_kind| die_kind.as_memnonic());

    transpose(&dice, ncol)
        .into_iter()
        .map(|maybe_die_kind| {
            match maybe_die_kind {
                Some(die_kind) => {
                    if die_kind == dice::DieKind::Standard {
                        ("s", die_kind.as_human_readable())
                    } else {
                        (die_kind.as_memnonic(), die_kind.as_human_readable())
                    }
                }
                None => ("", ""),
            }
        })
        .collect::<Vec<_>>()
}

///////////
// Tests //
///////////

#[cfg(test)]
mod test {
    use super::*;
    use proptest::prelude::*;

    fn factorial_ref(n: u32) -> u32 {
        (1..=n).product()
    }

    #[test]
    fn test_factorial_lt() {
        for n in 0..NUM_FACTORIALS as u32 {
            assert_eq!(factorial_ref(n), factorial(n));
        }
    }

    fn niters(n: u32) -> ProptestConfig {
        ProptestConfig::with_cases(n)
    }

    /// spread `x` into an array of nibbles.
    fn u32_into_nib_le(x: u32) -> [u8; 8] {
        let [b0, b1, b2, b3] = x.to_le_bytes();
        [
            b0 & 0x0f,
            (b0 >> 4) & 0x0f,
            b1 & 0x0f,
            (b1 >> 4) & 0x0f,
            b2 & 0x0f,
            (b2 >> 4) & 0x0f,
            b3 & 0x0f,
            (b3 >> 4) & 0x0f,
        ]
    }

    /// reference implementation of [`u32_any_nibs_between`]
    fn u32_any_nibs_between_ref(x: u32, mask: u32, m: u32, n: u32) -> bool {
        assert!((0..=7).contains(&m));
        assert!((0..=8).contains(&n));

        let m = m as u8;
        let n = n as u8;

        u32_into_nib_le(x)
            .into_iter()
            .zip(u32_into_nib_le(mask).into_iter())
            .any(|(nb_x, nb_mask)| (nb_mask == 0x1) && (m < nb_x) && (nb_x < n))
    }

    // reference implementation of [`std::arch::x86::bmi::_pdep32`] so I don't
    // need to gate tests.
    fn pdep32_fallback(src: u32, mut mask: u32) -> u32 {
        // iterate bit from lsb -> msb
        let mut bit = 1;
        let mut r = 0;
        while mask != 0 {
            if src & bit != 0 {
                // add the lsb from mask to r
                r |= mask & mask.wrapping_neg();
            }
            // clear lsb
            mask &= mask - 1;
            bit <<= 1;
        }
        r
    }

    // Efficiently generate combinations of mask bits that match a "`meta_mask`"
    // (by match I mean `meta_mask & mask == mask`).
    fn arb_mask_u32(meta_mask: u32) -> impl Strategy<Value = u32> {
        let nmasks: u32 = 1 << meta_mask.count_ones();

        (0..=nmasks).prop_map(move |mask_idx| pdep32_fallback(mask_idx, meta_mask))
    }

    #[test]
    fn test_u32_any_nibs_between() {
        assert!(u32_any_nibs_between(0x1234_5678, 0x1010_1010, 0, 3));
        assert!(!u32_any_nibs_between(0x1234_5678, 0x0011_1011, 0, 3));

        proptest!(niters(2000), |(x in any::<u32>(), mask in arb_mask_u32(0x1111_1111), m in (0u32..=7), n in (0u32..=8))| {
            assert_eq!(u32_any_nibs_between_ref(x, mask, m, n), u32_any_nibs_between(x, mask, m, n));
        });
    }

    fn u32_sum_all_nibs_ref(x: u32) -> u32 {
        u32_into_nib_le(x).into_iter().map(|x| x as u32).sum()
    }

    #[test]
    fn test_u32_sum_all_nibs() {
        proptest!(niters(2000), |(x in any::<u32>())| {
            prop_assert_eq!(u32_sum_all_nibs_ref(x), u32_sum_all_nibs(x));
 
        });
    }

    fn u64_leading_zero_bytes_ref(x: u64) -> u32 {
        x.to_be_bytes().into_iter().take_while(|&b| b == 0).count() as u32
    }

    #[test]
    fn test_u64_leading_zero_bytes() {
        proptest!(niters(2000), |(x in any::<u64>())| {
            prop_assert_eq!(u64_leading_zero_bytes_ref(x), u64_leading_zero_bytes(x));
        });
    }

    fn u64_elementwise_bytes_lt_ref(x: u64, y: u64) -> u64 {
        assert!(x & 0x8080_8080_8080_8080_u64 == 0);
        let bytes = x.to_le_bytes()
            .into_iter()
            .zip(y.to_le_bytes().into_iter())
            .map(|(bx, by)| if bx < by { 0x80_u8 } else { 0x00_u8 })
            .collect::<Vec<_>>();
        u64::from_le_bytes(<[u8; 8]>::try_from(&bytes[..]).unwrap())
    }

    #[test]
    fn test_u64_elementwise_bytes_lt() {
        proptest!(niters(2000), |(x in any::<u64>(), y in any::<u64>())| {
            let x = x & 0x7f7f_7f7f_7f7f_7f7f_u64;
            prop_assert_eq!(
                u64_elementwise_bytes_lt_ref(x, y),
                u64_elementwise_bytes_lt(x, y)
            );
        });
    }

    fn u64_leading_byte_idx_lt_ref(x: u64, y: u64) -> Option<u8> {
        x.to_le_bytes().into_iter()
            .zip(y.to_le_bytes().into_iter())
            .rposition(|(bx, by)| bx < by)
            .map(|idx| idx as u8)
    }

    #[test]
    fn test_u64_leading_byte_idx_lt() {
        proptest!(niters(2000), |(x in any::<u64>(), y in any::<u64>())| {
            let x = x & 0x7f7f_7f7f_7f7f_7f7f_u64;
            prop_assert_eq!(
                u64_leading_byte_idx_lt_ref(x, y),
                u64_leading_byte_idx_lt(x, y)
            );
        });
    }

    fn u64_trailing_byte_idx_lt_ref(n: u8, y: u64) -> Option<u8> {
        y.to_le_bytes()
            .into_iter()
            .position(|by| n < by)
            .map(|idx| idx as u8)
    }

    #[test]
    fn test_u64_trailing_byte_idx_lt() {
        proptest!(niters(2000), |(n in (0_u8..=127), y in any::<u64>())| {
            // let x = x & 0x7f7f_7f7f_7f7f_7f7f_u64;
            prop_assert_eq!(
                u64_trailing_byte_idx_lt_ref(n, y),
                u64_trailing_byte_idx_lt(n, y)
            );
        });
    }

    #[test]
    fn test_transpose() {
        let xs = vec![1, 2, 3, 4, 5];
        let expected_c1 = vec![Some(1), Some(2), Some(3), Some(4), Some(5)];
        let expected_c2 = vec![Some(1), Some(4), Some(2), Some(5), Some(3), None];
        let expected_c3 = vec![Some(1), Some(3), Some(5), Some(2), Some(4), None];
        let expected_c4 = vec![Some(1), Some(3), Some(5), None, Some(2), Some(4), None, None];

        assert_eq!(expected_c1, transpose(&xs, 1));
        assert_eq!(expected_c2, transpose(&xs, 2));
        assert_eq!(expected_c3, transpose(&xs, 3));
        assert_eq!(expected_c4, transpose(&xs, 4));
    }
}
