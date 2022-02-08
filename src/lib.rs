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

pub mod cli;
mod dice;
mod search2;

use std::cmp;

pub(crate) const DEFAULT_TARGET_SCORE: u16 = 4000;

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

///////////
// Tests //
///////////

#[cfg(test)]
mod test {
    use super::*;

    fn factorial_ref(n: u32) -> u32 {
        (1..=n).product()
    }

    #[test]
    fn test_factorial_lt() {
        for n in 0..NUM_FACTORIALS as u32 {
            assert_eq!(factorial_ref(n), factorial(n));
        }
    }
}
