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

use approx::relative_eq;
use ndarray::{s, Array1, Array2, ArrayView1};
use pico_args::Arguments;
use std::{
    cell::Cell,
    cmp::{self, min},
    collections::HashMap,
    fmt,
    str::FromStr,
    time::Instant,
};
use tabular::{row, Table};

const DEFAULT_TARGET_SCORE: u16 = 4000;

///////////////////
// Combinatorics //
///////////////////

/// The number of factorials to precompute in our static lookup table. Note this
/// number is chosen so as not to overflow a u32.
const NUM_FACTORIALS: usize = 13;

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

const fn factorial(n: u32) -> u32 {
    FACTORIAL_LT[n as usize]
}

/// `n choose k` without replacement.
#[cfg(test)]
const fn num_combinations(n: u32, k: u32) -> u32 {
    factorial(n) / (factorial(k) * factorial(n - k))
}

/// `n choose k` with replacement.
#[cfg(test)]
#[inline]
const fn num_multisets(n: u32, k: u32) -> u32 {
    num_combinations(n + k - 1, k)
}

/// A total ordering on f64's. Needed until `f64::total_cmp` is stabilized.
/// See: (https://doc.rust-lang.org/stable/std/primitive.f64.html#method.total_cmp)
#[inline]
pub fn total_cmp_f64(a: &f64, b: &f64) -> cmp::Ordering {
    let mut left = a.to_bits() as i64;
    let mut right = b.to_bits() as i64;

    left ^= (((left >> 63) as u64) >> 1) as i64;
    right ^= (((right >> 63) as u64) >> 1) as i64;

    left.cmp(&right)
}

//////////////
// Dice Set //
//////////////

/// A compressed representation of a set of dice, stored as counts of each die
/// packed into a u32. Visually,
///
/// `XXXX 1111 2222 3333 4444 5555 6666 XXXX`, where `XXXX`s are unused nibbles
/// and `NNNN` is the nibble for the n'th die.
///
/// This representation has the added benefit of being order invariant, since
/// the order of dice in a set is not important.
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
struct Counts(pub u32);

impl Counts {
    /// A new empty set of dice rolls.
    #[inline]
    fn new() -> Self {
        Self(0)
    }

    /// A convenience function for constructing a `Counts` set from an unordered
    /// list of dice rolls.
    #[cfg(test)]
    fn from_rolls(rolls: &[u8]) -> Self {
        let mut counts_arr = [0u8; 7];
        for roll in rolls {
            counts_arr[*roll as usize] += 1;
        }

        let mut counts = Self::new();
        for roll in 1..=6 {
            counts.set_count(roll, counts_arr[roll as usize]);
        }
        counts
    }

    #[inline]
    fn is_empty(self) -> bool {
        self.0 == 0
    }

    // The number of dice in this set
    #[inline]
    fn len(self) -> u8 {
        let mut len = 0;
        for roll in 1..=6 {
            len += self.get_count(roll);
        }
        len
    }

    #[inline]
    fn get_count(self, roll: u8) -> u8 {
        ((self.0 >> (4 * (roll as u32))) & 0xf) as u8
    }

    #[inline]
    fn set_count(&mut self, roll: u8, count: u8) {
        self.0 = (self.0 & !(0x7 << (4 * (roll as u32))))
            + (((count as u32) & 0x7) << (4 * (roll as u32)));
    }

    #[inline]
    fn add_count(&mut self, roll: u8, count: u8) {
        self.0 += ((count as u32) & 0x7) << (4 * (roll as u32));
    }

    fn add(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }

    /// Returns the score of a set of dice. will still work with non-scoring dice
    /// mixed in.
    fn score(self) -> u16 {
        if self.is_empty() {
            return 0;
        }

        let c1 = self.get_count(1);
        let c2 = self.get_count(2);
        let c3 = self.get_count(3);
        let c4 = self.get_count(4);
        let c5 = self.get_count(5);
        let c6 = self.get_count(6);

        // check for straights
        if c1 > 0 && c2 > 0 && c3 > 0 && c4 > 0 && c5 > 0 && c6 > 0 {
            return 1500;
        } else if c2 > 0 && c3 > 0 && c4 > 0 && c5 > 0 && c6 > 0 {
            let extra_5 = if c5 == 2 { 50 } else { 0 };
            // can't have an extra 1, since then we would just have a full straight
            return 750 + extra_5;
        } else if c1 > 0 && c2 > 0 && c3 > 0 && c4 > 0 && c5 > 0 {
            let extra_1 = if c1 == 2 { 100 } else { 0 };
            let extra_5 = if c5 == 2 { 50 } else { 0 };
            return 500 + extra_1 + extra_5;
        }

        let mut score: u16 = 0;

        for dice in 1u16..=6 {
            let count = self.get_count(dice as u8) as u16;

            if count == 0 {
                continue;
            }

            // score single or double 1s or 5s
            if count < 3 {
                if dice == 1 {
                    score += 100 * count;
                } else if dice == 5 {
                    score += 50 * count;
                }
            } else {
                let mult: u16 = 1 << (count - 3);
                let base = if dice == 1 { 1000 } else { dice * 100 };
                score += base * mult
            }
        }

        score
    }

    /// Return the score of a set of dice. If there are any non-scoring dice in
    /// the set, then this returns 0.
    fn exact_score(self) -> u16 {
        if self.is_empty() {
            return 0;
        }

        let c1 = self.get_count(1);
        let c2 = self.get_count(2);
        let c3 = self.get_count(3);
        let c4 = self.get_count(4);
        let c5 = self.get_count(5);
        let c6 = self.get_count(6);

        // check for pure straights and straights with 1 added scoring dice
        // [1 1 1 1 1 1]
        if c1 == 1 && c2 == 1 && c3 == 1 && c4 == 1 && c5 == 1 && c6 == 1 {
            return 1500;
        // [0 1 1 1 1 1]
        } else if c1 == 0 && c2 == 1 && c3 == 1 && c4 == 1 && c5 == 1 && c6 == 1 {
            return 750;
        // [0 1 1 1 2 1]
        } else if c1 == 0 && c2 == 1 && c3 == 1 && c4 == 1 && c5 == 2 && c6 == 1 {
            return 750 + 50;
        // [1 1 1 1 1 0]
        } else if c1 == 1 && c2 == 1 && c3 == 1 && c4 == 1 && c5 == 1 && c6 == 0 {
            return 500;
        // [2 1 1 1 1 0]
        } else if c1 == 2 && c2 == 1 && c3 == 1 && c4 == 1 && c5 == 1 && c6 == 0 {
            return 500 + 100;
        // [1 1 1 1 2 0]
        } else if c1 == 1 && c2 == 1 && c3 == 1 && c4 == 1 && c5 == 2 && c6 == 0 {
            return 500 + 50;
        }

        let mut score: u16 = 0;

        for dice in 1u16..=6 {
            let count = self.get_count(dice as u8) as u16;

            if count == 0 {
                continue;
            }

            // score single or double 1s or 5s
            if count < 3 {
                if dice == 1 {
                    score += 100 * count;
                } else if dice == 5 {
                    score += 50 * count;
                } else {
                    // less than 3 _non-scoring_ dice, this is not a valid scoring
                    // dice set.
                    return 0;
                }
            } else {
                let mult: u16 = 1 << (count - 3);
                let base = if dice == 1 { 1000 } else { dice * 100 };
                score += base * mult
            }
        }

        score
    }

    /// Return true if this set has no scoring dice, also called a "bust".
    #[inline]
    fn is_bust(self) -> bool {
        relative_eq!(self.score() as f64, 0.0)
    }

    /// Return the probability of rolling this set of dice.
    ///
    /// let n = number of dice in the set
    ///     P = n! / (6^n * ∏_{i∈[1,6]} c_i!)
    ///         where c_i is the count of the i'th dice in the set
    fn p_roll(self) -> f64 {
        let n = self.len();

        let prod: u32 = (1..=6)
            .map(|roll| factorial(self.get_count(roll) as u32))
            .product();

        let m = 6_u32.pow(n as u32);

        (factorial(n as u32) as f64) / ((prod * m) as f64)
    }

    fn to_rolls(self) -> Vec<u8> {
        let mut rolls = Vec::new();
        for roll in 1..=6 {
            for _ in 0..self.get_count(roll) {
                rolls.push(roll);
            }
        }
        rolls
    }
}

impl fmt::Debug for Counts {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (self.to_rolls()).fmt(f)
    }
}

impl cmp::Ord for Counts {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.to_rolls().cmp(&other.to_rolls())
    }
}

impl cmp::PartialOrd for Counts {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Parse a comma/space/tab separated list of dice into a `Counts` set.
/// Enclosing brackets ('[' or ']') optional.
impl FromStr for Counts {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut counts = Counts::new();

        let s = s.trim_start_matches('[');
        let s = s.trim_end_matches(']');

        let splitters = &[',', ' ', '\n', '\t'];

        for roll_str in s.split(splitters).filter(|s| !s.is_empty()) {
            let roll = roll_str
                .parse::<u8>()
                .map_err(|err| format!("dice roll is not a valid integer: {}", err))?;
            if (1..=6).contains(&roll) {
                counts.add_count(roll, 1);
            } else {
                return Err(format!("roll is out of range [1, 6]: {}", roll));
            }
        }

        Ok(counts)
    }
}

/// An `Iterator` over combinations (with replacement) of _all_ dice roll outcomes.
struct AllDiceMultisetsIter {
    /// the _next_ combination we'll output (unless we're done).
    counts: Counts,
    /// total number of dice rolls per combination.
    total_dice: u8,
    /// set to `true` when we're done generating.
    done: bool,
}

impl AllDiceMultisetsIter {
    fn new(total_dice: u8) -> Self {
        // initialize with our the very first combination: [1, 1, .., 1]
        let mut counts = Counts::new();
        counts.set_count(1, total_dice);

        Self {
            counts,
            total_dice,
            done: false,
        }
    }
}

impl Iterator for AllDiceMultisetsIter {
    type Item = Counts;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        // this is the combination we're about to output
        let current = self.counts;

        // we're done when we finally reach [6, 6, .., 6]
        if self.counts.get_count(6) == self.total_dice {
            self.done = true;
        } else {
            // patch self.counts to be the _next_ combination we'll output
            for i in 1..=5 {
                let count_i = self.counts.get_count(i);
                if count_i > 0 {
                    self.counts.add_count(i + 1, 1);
                    for j in 2..=i {
                        self.counts.set_count(j, 0);
                    }
                    self.counts.set_count(1, count_i - 1);
                    break;
                }
            }
        }

        Some(current)
    }
}

/// Return all possible sub-multisets of size `ndice` of a given set of dice rolls.
fn dice_multisets(set: Counts, ndice: u8) -> Vec<Counts> {
    fn rec(cb: &mut impl FnMut(Counts), mut acc: Counts, mut left: Counts, ndice: u8) {
        // time to return the accumulator
        if ndice == 0 {
            cb(acc);
            return;
        }

        // if ndice == 1 we can just return each unique element
        if ndice == 1 {
            for roll in 1..=6 {
                if left.get_count(roll) > 0 {
                    let mut new_acc = acc;
                    new_acc.add_count(roll, 1);
                    cb(new_acc);
                }
            }
            return;
        }

        let elts_left = left.len();

        // special case: |S| = n means just one possible combination: S
        if elts_left == ndice {
            cb(acc.add(left));
            return;
        }

        // will never have enough; skip
        if elts_left < ndice {
            return;
        }

        // find next non-zero-count roll
        let roll = (1..=6).find(|&roll| left.get_count(roll) > 0).unwrap();
        let count = left.get_count(roll);

        // completely remove the element from the set
        left.set_count(roll, 0);

        // add zero or more of this element
        for to_add in (0..=min(count, ndice)).rev() {
            acc.set_count(roll, to_add);
            rec(cb, acc, left, ndice - to_add);
        }
    }

    let mut out = Vec::new();
    rec(&mut |counts| out.push(counts), Counts::new(), set, ndice);
    out
}

////////////
// Action //
////////////

/// An `Action` represents any of the possible actions the player can take from a
/// given round [`State`]. Note that a "bust" is simply the absence of any possible
/// actions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum Action {
    /// If there is at least one scoring die, the player can pass their turn,
    /// ending their turn and scoring their current round total + the currently
    /// rolled dice score.
    Pass,
    /// If there is at least one scoring die and at least 2 dice left, the player
    /// can choose to hold some non-empty subset of scoring dice and re-roll the
    /// rest. The held dice score is added to their current round total.
    Roll(Counts),
}

////////////////////////
// Evaluation Context //
////////////////////////

/// The `Context` is some additional evaluation state passed along while evaluating
/// `(state, action) -> expected value`. The evaluation should still be correct
/// even without the `Context`; rather, the `Context` should be used only for
/// optimizations or evaluation statistics.
struct Context {
    /// A cache from normalized (State, Action) pairs to their expected value
    /// after evaluation.
    action_value_cache: HashMap<NormalizedStateAction, f64>,
    score_distr_cache: HashMap<NormalizedStateAction, ScorePMF>,
    actions_explored: u64,
    cache_hits: Cell<u64>,
    game_finished_prunes: Cell<u64>,
    /// A recursion depth tracker to limit the search depth. Since it appears you
    /// can actually start a new roll if you score and hold all the dice on the
    /// board, then the game is no longer finite and isn't guaranteed to terminate.
    /// This depth limit is then necessary for our search to terminate.
    depth: u32,
    depth_max: u32,
    depth_prunes: Cell<u64>,
    /// The current joint probability of this evaluation path, i.e., `P(X_1, .., X_n)`
    /// where `X_i` is the i'th random dice set drawn in this evaluation path.
    /// We use this to limit searching of super low probability evaluation paths
    /// (which also tend to explode our branching factor).
    joint_path_prob: f64,
    joint_path_prob_min: f64,
    joint_prob_prunes: Cell<u64>,
}

impl Context {
    fn new() -> Self {
        Context {
            game_finished_prunes: Cell::new(0),
            depth: 0,
            depth_max: 30,
            joint_path_prob: 1.0,
            joint_path_prob_min: 1.0e-10,
            action_value_cache: HashMap::new(),
            score_distr_cache: HashMap::new(),
            actions_explored: 0,
            cache_hits: Cell::new(0),
            depth_prunes: Cell::new(0),
            joint_prob_prunes: Cell::new(0),
        }
    }

    #[inline]
    fn actions_explored(&self) -> u64 {
        self.actions_explored
    }

    #[inline]
    fn inc_actions_explored(&mut self) {
        self.actions_explored += 1;
    }

    #[inline]
    #[allow(unused)]
    fn set_depth_max(&mut self, depth_max: u32) -> &mut Self {
        self.depth_max = depth_max;
        self
    }

    #[inline]
    #[allow(unused)]
    fn set_joint_prob_min(&mut self, joint_path_prob_min: f64) -> &mut Self {
        self.joint_path_prob_min = joint_path_prob_min;
        self
    }

    #[inline]
    fn with_next_depth<T>(&mut self, p_roll: f64, f: impl FnOnce(&mut Self) -> T) -> T {
        self.depth += 1;
        let pre_joint_prob = self.joint_path_prob;
        self.joint_path_prob *= p_roll;

        let res = f(self);

        self.depth -= 1;
        self.joint_path_prob = pre_joint_prob;

        res
    }

    #[inline]
    fn depth_prunes(&self) -> u64 {
        self.depth_prunes.get()
    }

    #[inline]
    fn game_finished_prunes(&self) -> u64 {
        self.game_finished_prunes.get()
    }

    #[inline]
    fn joint_prob_prunes(&self) -> u64 {
        self.joint_prob_prunes.get()
    }

    #[inline]
    fn should_prune(&self, my_round_total: u16, target_score: u16) -> bool {
        // TODO(philiphayes): is this still necessary?
        // if self.depth > self.depth_max {
        //     self.depth_prunes.set(self.depth_prunes() + 1);
        //     return true;
        // }
        //
        if my_round_total >= target_score {
            self.game_finished_prunes
                .set(self.game_finished_prunes.get() + 1);
            return true;
        }
        // TODO: is this still necessary?
        // if self.joint_path_prob < self.joint_path_prob_min {
        //     self.joint_prob_prunes.set(self.joint_prob_prunes() + 1);
        //     return true;
        // }
        false
    }

    #[inline]
    fn peek_cache(&self, key: &NormalizedStateAction) -> Option<f64> {
        let out = self.action_value_cache.get(key).copied();
        if out.is_some() {
            self.cache_hits.set(self.cache_hits() + 1);
        }
        out
    }

    #[inline]
    fn fill_cache(&mut self, key: NormalizedStateAction, value: f64) {
        self.action_value_cache.insert(key, value);
    }

    #[inline]
    fn peek_score_distr_cache(&self, key: &NormalizedStateAction) -> Option<ScorePMF> {
        let out = self.score_distr_cache.get(key).cloned();
        if out.is_some() {
            self.cache_hits.set(self.cache_hits() + 1);
        }
        out
    }

    #[inline]
    fn fill_score_distr_cache(&mut self, key: NormalizedStateAction, pmf: ScorePMF) {
        self.score_distr_cache.insert(key, pmf);
    }

    #[inline]
    fn cache_size(&self) -> usize {
        self.action_value_cache.len()
    }

    #[inline]
    fn cache_hits(&self) -> u64 {
        self.cache_hits.get()
    }

    #[inline]
    fn cache_hit_rate(&self) -> f64 {
        (self.cache_hits() as f64) / (self.actions_explored() as f64 + 1.0)
    }

    #[inline]
    fn depth_prune_rate(&self) -> f64 {
        (self.depth_prunes() as f64) / (self.actions_explored() as f64 + 1.0)
    }

    #[inline]
    fn game_finished_rate(&self) -> f64 {
        (self.game_finished_prunes() as f64) / (self.actions_explored() as f64 + 1.0)
    }

    #[inline]
    fn joint_prob_prune_rate(&self) -> f64 {
        (self.joint_prob_prunes() as f64) / (self.actions_explored() as f64 + 1.0)
    }
}

////////////////////////
// Normalized Q-State //
////////////////////////

/// A normalized representation of a q-state / `(State, Action)` pair.
///
/// insight:  peek_cache(State {200, [1,1,5,X1,X2,X3]}, Roll[1,1,5])
///       and peek_cache(State {200, [1,1,5,Y1,Y2,Y3]}, Roll[1,1,5]) should hit
///       the same cache slot as long as all Xi and Yi aren't used by the action.
///
/// option 1: normalize the state representation only so _always_ non-scoring
///           dice get lumped into the 0-bucket and only affect the length.
///
/// option 2: normalize the (state, action) representation
///             (State {200, [1,1,5,2,2,3]}, Roll[1,1,5])
///           ≡ StateAction { round_total=450, ndice_left=3 }
///
/// `NormalizedStateAction` is option 2.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
struct NormalizedStateAction {
    /// The round total after applying the action.
    pub my_round_total: u16,
    /// The total score we're trying to hit. If we meet or exceed this value,
    /// then we win.
    pub target_score: u16,
    /// The number of dice left to reroll after applying the action.
    pub ndice_left: u8,
}

impl NormalizedStateAction {
    fn new(my_round_total: u16, target_score: u16, ndice_left: u8) -> Self {
        Self {
            my_round_total,
            target_score,
            ndice_left,
        }
    }

    /// An pseudo initial state, before rolling the first set of dice.
    fn init_state(target_score: u16) -> Self {
        Self {
            my_round_total: 0,
            target_score,
            ndice_left: 6,
        }
    }

    #[allow(unused)]
    fn from_state_action(state: State, action: Action) -> Self {
        match action {
            Action::Pass => Self {
                my_round_total: state.my_round_total + state.rolled_dice.score(),
                target_score: state.target_score,
                ndice_left: 0,
            },
            Action::Roll(held_dice) => Self::from_state_roll_action(state, held_dice),
        }
    }

    #[inline]
    fn from_state_roll_action(state: State, held_dice: Counts) -> Self {
        // we have this many dice left to roll
        let ndice_left = state.rolled_dice.len() - held_dice.len();

        // apparently you can actually roll again with a fresh hand if
        // you hold all the dice left and they're all scoring dice.
        let ndice_left = if ndice_left == 0 { 6 } else { ndice_left };

        Self {
            // fold the held dice score into the round total
            my_round_total: state.my_round_total + held_dice.exact_score(),
            target_score: state.target_score,
            ndice_left,
        }
    }

    #[inline]
    fn into_state(self, rolled_dice: Counts) -> State {
        State {
            my_round_total: self.my_round_total,
            target_score: self.target_score,
            rolled_dice,
        }
    }

    #[inline]
    fn is_pass(self) -> bool {
        self.ndice_left == 0
    }

    /// Return an `Iterator` over all possible states reachable after rolling the
    /// `ndice_left`, along with the conditional probability of reaching each
    /// state.
    #[inline]
    fn possible_roll_states(self) -> impl Iterator<Item = (State, f64)> {
        AllDiceMultisetsIter::new(self.ndice_left)
            .map(move |next_roll| (self.into_state(next_roll), next_roll.p_roll()))
    }

    /// Given a normalized `(State, Action)` pair, evaluate the expected value
    /// of applying the `Action` to the `State`, assuming we always choose
    /// subsequent actions by maximum expected value.
    fn expected_value(self, ctxt: &mut Context) -> f64 {
        ctxt.inc_actions_explored();

        if self.is_pass() {
            // expected value := P=1.0 * my_round_total
            return cmp::min(self.target_score, self.my_round_total) as f64;
        }

        // check if the cache already contains this normalized (State, Action)
        // pair.
        if let Some(action_value) = ctxt.peek_cache(&self) {
            return action_value;
        }

        // prune deep paths, low joint probability paths, and post-game finish
        // paths. if we pass any limit, just pretend this action always busts.
        if ctxt.should_prune(self.my_round_total, self.target_score) {
            return 0.0;
        }

        let mut expected_value = 0.0_f64;

        // for all possible dice rolls
        for (next_state, p_roll) in self.possible_roll_states() {
            // want to maximize expected value; choose action with
            // greatest expected value
            let best_action_value = ctxt.with_next_depth(p_roll, move |ctxt| {
                next_state
                    .actions()
                    .into_iter()
                    .map(|next_action| {
                        Self::from_state_action(next_state, next_action).expected_value(ctxt)
                    })
                    .max_by(total_cmp_f64)
                    .unwrap_or(0.0)
            });

            expected_value += p_roll * best_action_value;
        }

        ctxt.fill_cache(self, expected_value);
        expected_value
    }

    fn score_distribution(self, ctxt: &mut Context) -> ScorePMF {
        ctxt.inc_actions_explored();

        if self.is_pass() {
            let score = cmp::min(self.target_score, self.my_round_total);
            return ScorePMF::constant(score);
        }

        if let Some(pmf) = ctxt.peek_score_distr_cache(&self) {
            return pmf;
        }

        if ctxt.should_prune(self.my_round_total, self.target_score) {
            return ScorePMF::bust();
        }

        let mut score_pmf = ScorePMF::new();

        for (next_state, p_roll) in self.possible_roll_states() {
            let (_best_exp_value, best_score_distr) = ctxt.with_next_depth(p_roll, move |ctxt| {
                next_state
                    .actions()
                    .into_iter()
                    .map(|next_action| {
                        let distr = Self::from_state_action(next_state, next_action)
                            .score_distribution(ctxt);
                        (distr.expected_value(), distr)
                    })
                    .max_by(|(v1, _), (v2, _)| total_cmp_f64(v1, v2))
                    .unwrap_or_else(|| (0.0, ScorePMF::bust()))
            });

            score_pmf.add_conditional_distr(p_roll, &best_score_distr);
        }

        ctxt.fill_score_distr_cache(self, score_pmf.clone());
        score_pmf
    }
}

// TODO(philiphayes): probably fold the target score back into the state, so
// caches can be used across turns? wouldn't that add like n^2 # cache entries?
// what is wrong with adding points_needed_to_win vs round_total?

/////////////////
// Round State //
/////////////////

/// A representation of the player's round state.
#[derive(Copy, Clone, Debug)]
struct State {
    /// The set of dice the player just rolled.
    pub rolled_dice: Counts,
    /// The player's current round score (accumulated from previous rolls in the turn).
    pub my_round_total: u16,
    /// The total score we're trying to hit. If we meet or exceed this value,
    /// then we win.
    pub target_score: u16,
}

impl State {
    fn new(my_round_total: u16, target_score: u16, rolled_dice: Counts) -> Self {
        Self {
            rolled_dice,
            my_round_total,
            target_score,
        }
    }

    /// From the current round `State` return the complete set of possible
    /// `Action`s the player can take. Returns an empty set if the player has
    /// "busted".
    fn actions(&self) -> Vec<Action> {
        // if this dice roll has no scores whatsoever, then there are no actions
        // (our turn has ended).
        if self.rolled_dice.is_bust() {
            return Vec::new();
        }

        // we _can_ in fact hold all the dice, but they must all be scoring dice.
        let max_num_holds = self.rolled_dice.len();

        // insight: it's (almost always) not rational to choose an action with a
        // lower score but the same number of dice.
        //
        // for example, suppose we have [1,5] on the table. we then have 4 possible
        // actions: [pass, roll[1], roll[5], roll[1,5]].
        //
        // consider roll[1] vs roll[5]
        //
        // choosing roll[1] means scoring 100 pts then rolling 1 die while
        // roll[5] scores 50 pts and also rolls 1 die. (hypothesis) there's no
        // reason to lose out on 50 pts by only holding 5.

        // TODO(philiphayes): is this still necessary?
        let mut best_score_by_ndice = [0u16; 8];

        // the set of all possible dice we can hold from the board.
        // we must hold at least one die.
        let possible_holds = (1..=max_num_holds)
            .flat_map(|ndice| dice_multisets(self.rolled_dice, ndice))
            // only accept holds of scoring dice and the max score hold per hold size
            .filter_map(|held_dice| {
                let len = held_dice.len() as usize;
                let score = held_dice.exact_score();
                // this also handles rejecting zero score rolls (since strictly >)
                if score > best_score_by_ndice[len] {
                    best_score_by_ndice[len] = score;
                    Some(Action::Roll(held_dice))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        // let possible_holds = (1..=max_num_holds)
        //     .flat_map(|ndice| dice_multisets(self.rolled_dice, ndice))
        //     .filter(|held_dice| held_dice.exact_score() > 0)
        //     .map(Action::Roll)
        //     .collect::<Vec<_>>();

        let mut actions_vec = possible_holds;

        // can always pass if we have some scores
        actions_vec.push(Action::Pass);

        actions_vec
    }

    /// Evaluate the expected value of applying this `Action` to the `State`,
    /// assuming we always choose subsequent actions by maximum expected value.
    #[inline]
    fn action_expected_value(&self, ctxt: &mut Context, action: Action) -> f64 {
        NormalizedStateAction::from_state_action(*self, action).expected_value(ctxt)
    }

    /// Evaluate the probability of "busting" immediately after applying the
    /// given `Action` to the current turn `State`.
    ///
    /// Fun fact: 3% chance to bust on your first roll : )
    fn action_p_bust(&self, action: Action) -> f64 {
        match action {
            Action::Pass => 0.0,
            Action::Roll(held_dice) => {
                // we have this many dice left to roll
                let ndice_left = self.rolled_dice.len() - held_dice.len();
                let ndice_left = if ndice_left == 0 { 6 } else { ndice_left };

                let mut p_bust = 0.0_f64;
                for next_roll in AllDiceMultisetsIter::new(ndice_left) {
                    let p_roll = next_roll.p_roll();

                    if next_roll.is_bust() {
                        p_bust += p_roll
                    }
                }
                p_bust
            }
        }
    }

    /// For each possible `Action` from this `State`, conditioned on choosing that
    /// action, what is the expected turn score and bust probability?
    fn actions_by_expected_value(&self, ctxt: &mut Context) -> Vec<(Action, f64, f64)> {
        let mut actions_values = self
            .actions()
            .into_iter()
            .map(|action| {
                (
                    action,
                    self.action_expected_value(ctxt, action),
                    self.action_p_bust(action),
                )
            })
            .collect::<Vec<_>>();

        // sort by the expected turn score from highest to lowest.
        actions_values.sort_unstable_by(|(_, v1, _), (_, v2, _)| total_cmp_f64(v1, v2).reverse());
        actions_values
    }
}

/// Before even starting the game, what is our expected score?
/// => 564.35
#[allow(unused)]
fn expected_score_a_priori(ctxt: &mut Context, target_score: u16) -> f64 {
    NormalizedStateAction::init_state(target_score).expected_value(ctxt)
}

///////////////
// Score PMF //
///////////////

#[derive(Clone, Debug)]
struct ScorePMF(HashMap<u16, f64>);

impl ScorePMF {
    #[inline]
    fn new() -> Self {
        Self(HashMap::new())
    }

    #[inline]
    fn constant(score: u16) -> Self {
        let mut pmf = HashMap::new();
        pmf.insert(score, 1.0);
        Self(pmf)
    }

    #[inline]
    fn bust() -> Self {
        Self::constant(0)
    }

    fn into_dense(&self, target_score: u16) -> Array1<f64> {
        let num_states = MarkovMatrix::num_states(target_score);

        let mut dense_pmf = Array1::zeros(num_states);

        for state_idx in 0..num_states {
            dense_pmf[state_idx] = self
                .0
                .get(&MarkovMatrix::i2s(state_idx))
                .copied()
                .unwrap_or(0.0);
        }

        dense_pmf
    }

    fn expected_value(&self) -> f64 {
        self.0
            .iter()
            .map(|(&score, &p_score)| p_score * (score as f64))
            .sum()
    }

    fn add_conditional_distr(&mut self, p_cond: f64, cond_distr: &Self) {
        use std::ops::AddAssign;
        for (&score, &p_score) in &cond_distr.0 {
            self.0
                .entry(score)
                .or_insert(0.0)
                .add_assign(p_cond * p_score);
        }
    }

    fn total_mass(&self) -> f64 {
        self.0.values().sum()
    }
}

#[derive(Debug)]
struct MarkovMatrix(Array2<f64>);

impl MarkovMatrix {
    /// score to state index
    #[inline]
    const fn s2i(score: u16) -> usize {
        (score / 50) as usize
    }

    /// state index to score
    #[inline]
    const fn i2s(state_idx: usize) -> u16 {
        (state_idx * 50) as u16
    }

    #[inline]
    const fn num_states(target_score: u16) -> usize {
        Self::s2i(target_score) + 1
    }

    fn from_optimal_policy(target_score: u16) -> Self {
        let num_states = Self::num_states(target_score);

        let mut matrix = Array2::zeros((num_states, num_states));

        // compute the score distribution for each possible intermediate accumulated
        // turn score, then assemble into a matrix where each column x_i is the
        // score distribution pmf starting with turn score i*50.
        for turn_score_idx in Self::s2i(0)..Self::s2i(target_score) {
            let turn_score = Self::i2s(turn_score_idx);

            let mut ctxt = Context::new();
            let qstate = NormalizedStateAction::new(0, target_score - turn_score, 6);

            let score_pmf = qstate
                .score_distribution(&mut ctxt)
                .into_dense(target_score);

            matrix
                .slice_mut(s![turn_score_idx.., turn_score_idx])
                .assign(&score_pmf.slice(s![..num_states - turn_score_idx]));
        }

        // remember to set the absorber column [0 0 .. 1] last (if we've won the
        // game, we stay in the win state forever).
        matrix[[num_states - 1, num_states - 1]] = 1.0;

        Self(matrix)
    }

    fn turns_to_win_cdf(&self, max_num_turns: usize) -> Array1<f64> {
        let n = self.0.nrows();
        let mut accum = Array2::<f64>::eye(n);

        let mut turns_cdf = Array1::<f64>::zeros(max_num_turns);

        for turn in 0..max_num_turns {
            // A^i+1 = A^i * A
            accum = accum.dot(&self.0);
            // P[win on turn i] = (A^i · [1 0 .. 0]^T)_{n-1} = (A^i)_{n-1,0}
            turns_cdf[turn] = accum[[n - 1, 0]];
        }

        turns_cdf
    }
}

/// Given `X_1` and `X_2` random variables defined by the same CDF `cdf`, returns
/// the `Pr[X_1 <= X_2]`.
fn p_rv_lte_itself(cdf: ArrayView1<f64>) -> f64 {
    // p = ∑_{x_i} Pr[X_1 = x_i] * Pr[X_2 >= x_i]
    let p = 0.0;

    // c_i1 = Pr[X <= prev(x_i)] = cdf[i - 1]
    let c_i1 = 0.0;

    let (p, _c_i1) = cdf.into_iter().fold((p, c_i1), |(p, c_i1), &c_i| {
        // c_i = cdf[i]

        // p_1 = Pr[X_1 = x_i]
        //     = Pr[X_1 <= x_i] - Pr[X_1 <= prev(x_i)]
        //     = cdf[i] - cdf[i - 1]
        let p_1 = c_i - c_i1;

        // p_2 = Pr[X_2 >= x_i]
        //     = 1 - Pr[X_2 < x_i]
        //     = 1 - Pr[X_2 <= prev(x_i)]
        //     = 1 - cdf[i - 1]
        let p_2 = 1.0 - c_i1;

        (p + (p_1 * p_2), c_i)
    });

    p
}

/////////
// CLI //
/////////

trait Command: Sized {
    const USAGE: &'static str;

    fn parse_from_args(args: Arguments) -> Result<Self, String>;
    fn run(self);

    fn maybe_help(args: &mut Arguments) {
        if args.contains(["-h", "--help"]) {
            print!("{}", Self::USAGE);
            std::process::exit(0);
        }
    }
}

#[derive(Debug)]
struct BestActionCommand {
    round_total: u16,
    target_score: u16,
    rolled_dice: Counts,
}

impl Command for BestActionCommand {
    const USAGE: &'static str = "\
kcddice best-action - compute the best action to take in a round

USAGE:
    kcddice best-action [option ...] <round-total> <rolled-dice>

EXAMPLES:
    kcddice best-action 550 [1,1,2,3,5,6]

OPTIONS:
    · --target-score / -t score (default: 4000)
      The score needed to win the game
";

    fn parse_from_args(mut args: Arguments) -> Result<Self, String> {
        Self::maybe_help(&mut args);

        let cmd = Self {
            round_total: args.free_from_str().map_err(|err| err.to_string())?,
            rolled_dice: args.free_from_str().map_err(|err| err.to_string())?,
            target_score: args
                .opt_value_from_str(["--target-score", "-t"])
                .map_err(|err| err.to_string())?
                .unwrap_or(DEFAULT_TARGET_SCORE),
        };

        let remaining = args.finish();
        if !remaining.is_empty() {
            return Err(format!("unexpected arguments left: '{:?}'", remaining));
        }

        Ok(cmd)
    }

    fn run(self) {
        let state = State::new(self.round_total, self.target_score, self.rolled_dice);
        let mut ctxt = Context::new();

        let start_time = Instant::now();
        let actions_values = state.actions_by_expected_value(&mut ctxt);
        let search_duration = start_time.elapsed();

        let mut table = Table::new("{:>}  {:<}  {:>} {:>}").with_row(row!(
            "action",
            "held dice",
            "exp v",
            "pbust"
        ));

        let len = actions_values.len();
        for (action, value, p_bust) in actions_values.into_iter().take(10) {
            let value_str = format!("{:0.1}", value);
            let p_bust_str = format!("{:0.2}", p_bust);
            let (action_str, dice_str) = match action {
                Action::Pass => ("pass", String::new()),
                Action::Roll(held_dice) => ("hold dice", format!("{:?}", held_dice)),
            };
            table.add_row(row!(action_str, dice_str, value_str, p_bust_str));
        }

        // we only show the top 10 results, but display '...' to show that
        // there were more.
        if len > 10 {
            table.add_row(row!("...", format!("(x {})", len - 10), "", ""));
        }

        // display evaluation statistics
        table.add_heading("");
        table.add_row(row!(
            "search duration",
            format!("{:.2?}", search_duration),
            "",
            ""
        ));
        table.add_row(row!(
            "actions explored",
            ctxt.actions_explored().to_string(),
            "",
            ""
        ));
        table.add_row(row!("cache size", ctxt.cache_size().to_string(), "", ""));
        table.add_row(row!(
            "cache hit rate",
            format!("{:0.3}", ctxt.cache_hit_rate()),
            "",
            ""
        ));
        table.add_row(row!(
            "depth prune rate",
            format!("{:0.3}", ctxt.depth_prune_rate()),
            "",
            ""
        ));
        table.add_row(row!(
            "game finished rate",
            format!("{:0.3}", ctxt.game_finished_rate()),
            "",
            ""
        ));
        table.add_row(row!(
            "joint prob. prune rate",
            format!("{:0.3}", ctxt.joint_prob_prune_rate()),
            "",
            ""
        ));

        print!("\n{}", table);
    }
}

#[derive(Debug)]
struct ScoreDistrCommand {
    round_total: u16,
    target_score: u16,
    ndice_left: u8,
}

impl Command for ScoreDistrCommand {
    const USAGE: &'static str = "\
kcddice score-distr - TODO

USAGE:
    kcddice score-distr [option ...] <round-total> <target-score> <ndice-left>
";

    fn parse_from_args(mut args: Arguments) -> Result<Self, String> {
        Self::maybe_help(&mut args);

        let cmd = Self {
            round_total: args.free_from_str().map_err(|err| err.to_string())?,
            target_score: args.free_from_str().map_err(|err| err.to_string())?,
            ndice_left: args.free_from_str().map_err(|err| err.to_string())?,
        };

        let remaining = args.finish();
        if !remaining.is_empty() {
            return Err(format!("unexpected arguments left: '{:?}'", remaining));
        }

        Ok(cmd)
    }

    fn run(self) {
        let qstate =
            NormalizedStateAction::new(self.round_total, self.target_score, self.ndice_left);

        let mut ctxt = Context::new();

        let start_time = Instant::now();
        let score_pmf = qstate.score_distribution(&mut ctxt);
        let search_duration = start_time.elapsed();

        let mut dense_score_pmf = score_pmf.0.clone().into_iter().collect::<Vec<_>>();
        dense_score_pmf.sort_unstable_by(|(s1, _), (s2, _)| s1.cmp(s2));

        for (score, p_score) in dense_score_pmf {
            println!("{}\t{}", score, p_score);
        }

        let mut table = Table::new("{:>}  {:<}");
        table.add_row(row!(
            "pmf expected value",
            format!("{:.3}", score_pmf.expected_value())
        ));
        table.add_row(row!(
            "pmf total mass",
            format!("{}", score_pmf.total_mass())
        ));
        table.add_row(row!("search duration", format!("{:.2?}", search_duration)));
        table.add_row(row!(
            "actions explored",
            ctxt.actions_explored().to_string()
        ));
        eprint!("\n{}", table);
    }
}

#[derive(Debug)]
struct MarkovMatrixCommand {
    target_score: u16,
}

impl Command for MarkovMatrixCommand {
    // TODO(philiphayes): fill out
    const USAGE: &'static str = "";

    fn parse_from_args(mut args: Arguments) -> Result<Self, String> {
        Self::maybe_help(&mut args);

        let cmd = Self {
            target_score: args
                .opt_free_from_str()
                .map_err(|err| err.to_string())?
                .unwrap_or(DEFAULT_TARGET_SCORE),
        };

        let remaining = args.finish();
        if !remaining.is_empty() {
            return Err(format!("unexpected arguments left: '{:?}'", remaining));
        }

        Ok(cmd)
    }

    fn run(self) {
        let start_time = Instant::now();
        let matrix = MarkovMatrix::from_optimal_policy(self.target_score);
        let search_duration = start_time.elapsed();

        println!("{:?}", matrix);
        println!("\nsearch duration  {:.2?}", search_duration);
    }
}

#[derive(Debug)]
struct TurnsCdfCommand {
    target_score: u16,
    max_num_turns: usize,
}

impl Command for TurnsCdfCommand {
    // TODO(philiphayes): fill out
    const USAGE: &'static str = "";

    fn parse_from_args(mut args: Arguments) -> Result<Self, String> {
        Self::maybe_help(&mut args);

        let cmd = Self {
            target_score: args.free_from_str().map_err(|err| err.to_string())?,
            max_num_turns: args.free_from_str().map_err(|err| err.to_string())?,
        };

        let remaining = args.finish();
        if !remaining.is_empty() {
            return Err(format!("unexpected arguments left: '{:?}'", remaining));
        }

        Ok(cmd)
    }

    fn run(self) {
        let start_time = Instant::now();
        let matrix = MarkovMatrix::from_optimal_policy(self.target_score);
        let turns_cdf = matrix.turns_to_win_cdf(self.max_num_turns);
        let search_duration = start_time.elapsed();

        println!("turn\tcumulative probability");

        for turn in 1..=self.max_num_turns {
            println!("{}\t{}", turn, turns_cdf[turn - 1]);
        }

        // Assuming we play against ourselves and always go first, what is our a
        // priori win probability (before we even roll the first dice)? This is
        // useful for deciding how much to bet (e.g., following the
        // Kelley-Criterion).
        //
        // For the typical setup (target=4000, 6 normal dice), this is ≈0.562.
        // The Kelley-Criterion optimal bet is then 12.5% of your total wealth.
        let p_win = p_rv_lte_itself(turns_cdf.view());

        let mut table = Table::new("{:>}  {:<}");
        table.add_row(row!("search_duration", format!("{:.2?}", search_duration)));
        table.add_row(row!("Pr[win]", format!("{}", p_win)));
        eprintln!("\n{}", table);
    }
}

#[derive(Debug)]
enum BaseCommand {
    BestAction(BestActionCommand),
    ScoreDistr(ScoreDistrCommand),
    MarkovMatrix(MarkovMatrixCommand),
    TurnsCdf(TurnsCdfCommand),
}

impl Command for BaseCommand {
    const USAGE: &'static str = "\
kcddice - A utility for optimally playing the Kingdom Come: Deliverance dice game!

USAGE:
    kcddice [option ...] <subcommand>

SUBCOMMANDS:
    · kcddice best-action - compute the best action to take in a round
    · kcddice score-distr - compute full score PMF given the score and number of dice left to roll
    · kcddice markov-matrix - TODO
    · kcddice turns-cdf - TODO
";

    fn parse_from_args(mut args: Arguments) -> Result<Self, String> {
        let maybe_subcommand = args.subcommand().map_err(|err| err.to_string())?;

        match maybe_subcommand.as_deref() {
            Some("best-action") => Ok(Self::BestAction(BestActionCommand::parse_from_args(args)?)),
            Some("score-distr") => Ok(Self::ScoreDistr(ScoreDistrCommand::parse_from_args(args)?)),
            Some("markov-matrix") => Ok(Self::MarkovMatrix(MarkovMatrixCommand::parse_from_args(
                args,
            )?)),
            Some("turns-cdf") => Ok(Self::TurnsCdf(TurnsCdfCommand::parse_from_args(args)?)),
            Some(command) => Err(format!("'{}' is not a recognized command", command)),
            None => {
                Self::maybe_help(&mut args);
                Err("no subcommand specified".to_string())
            }
        }
    }

    fn run(self) {
        match self {
            Self::BestAction(cmd) => cmd.run(),
            Self::ScoreDistr(cmd) => cmd.run(),
            Self::MarkovMatrix(cmd) => cmd.run(),
            Self::TurnsCdf(cmd) => cmd.run(),
        }
    }
}

fn main() {
    let args = pico_args::Arguments::from_env();

    match BaseCommand::parse_from_args(args) {
        Ok(cmd) => cmd.run(),
        Err(err) => {
            eprintln!("error: {}", err);
            eprintln!("Try 'kcddice --help' for more information.");
            std::process::exit(1);
        }
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod test {
    use super::*;
    use approx::assert_relative_eq;
    use std::collections::HashSet;

    fn factorial_ref(n: u32) -> u32 {
        (1..=n).product()
    }

    #[test]
    fn test_factorial_lt() {
        for n in 0..NUM_FACTORIALS as u32 {
            assert_eq!(factorial_ref(n), factorial(n));
        }
    }

    // simple recursive implementation
    fn all_dice_multisets_ref(ndice: u8) -> Vec<Counts> {
        fn rec(cb: &mut impl FnMut(Counts), counts: Counts, current_dice: u8, ndice: u8) {
            // time to return the accumulator
            if ndice == 0 {
                cb(counts);
                return;
            // passed the end; nothing to return
            } else if current_dice > 6 {
                return;
            }

            // can choose 0..=ndice of the current element
            for choose_counts in 0..=ndice {
                let mut new_counts = counts;
                new_counts.set_count(current_dice, choose_counts);
                rec(cb, new_counts, current_dice + 1, ndice - choose_counts);
            }
        }

        let mut out = Vec::new();
        rec(&mut |counts| out.push(counts), Counts::new(), 1, ndice);
        out.sort_unstable();
        out
    }

    #[test]
    fn test_counts_score() {
        assert_eq!(0, Counts::from_rolls(&[]).score());
        assert_eq!(100, Counts::from_rolls(&[1]).score());
        assert_eq!(150, Counts::from_rolls(&[5, 1]).score());
        assert_eq!(0, Counts::from_rolls(&[2, 2, 3]).score());
        assert_eq!(1500, Counts::from_rolls(&[6, 5, 4, 3, 2, 1]).score());
        assert_eq!(250, Counts::from_rolls(&[1, 1, 5, 3, 2, 2]).score());
        assert_eq!(1050, Counts::from_rolls(&[1, 1, 5, 3, 1, 2]).score());
        assert_eq!(400, Counts::from_rolls(&[4, 4, 4]).score());
        assert_eq!(800, Counts::from_rolls(&[4, 4, 4, 4]).score());
        assert_eq!(1600, Counts::from_rolls(&[4, 4, 4, 4, 4]).score());
        assert_eq!(3200, Counts::from_rolls(&[4, 4, 4, 4, 4, 4]).score());
    }

    #[test]
    fn test_counts_from_rolls() {
        assert!(Counts::from_rolls(&[]).is_empty());

        assert_eq!(0, Counts::from_rolls(&[]).len());
        assert_eq!(1, Counts::from_rolls(&[1]).len());
        assert_eq!(2, Counts::from_rolls(&[4, 2]).len());
        assert_eq!(6, Counts::from_rolls(&[6, 5, 4, 3, 2, 1]).len());

        assert_eq!(1, Counts::from_rolls(&[6, 5, 4, 3, 2, 1]).get_count(6));
        assert_eq!(3, Counts::from_rolls(&[6, 5, 3, 3, 3]).get_count(3));
        assert_eq!(0, Counts::from_rolls(&[6, 5, 3, 3, 3]).get_count(1));
        assert_eq!(6, Counts::from_rolls(&[3, 3, 3, 3, 3, 3]).get_count(3));
        assert_eq!(6, Counts::from_rolls(&[6, 6, 6, 6, 6, 6]).get_count(6));
        assert_eq!(0, Counts::from_rolls(&[6, 6, 6, 6, 6, 6]).get_count(3));
    }

    #[test]
    fn test_all_dice_multisets_iter() {
        for k in 0..=6 {
            let mut combs = AllDiceMultisetsIter::new(k as u8).collect::<Vec<_>>();
            combs.sort_unstable();

            // outputs expected number of elements
            assert_eq!(num_multisets(6, k) as usize, combs.len(),);

            // no duplicates
            let combs_set = HashSet::<Counts>::from_iter(combs.clone().into_iter());
            assert_eq!(num_multisets(6, k) as usize, combs_set.len());

            // matches recursive implementation
            let combs_ref = all_dice_multisets_ref(k as u8);
            assert_eq!(combs, combs_ref);
        }
    }

    #[test]
    fn test_dice_multisets() {
        // [1 1], [1 3], [1 5], [3 5]
        let expected = vec![
            Counts::from_rolls(&[1, 1]),
            Counts::from_rolls(&[1, 3]),
            Counts::from_rolls(&[1, 5]),
            Counts::from_rolls(&[3, 5]),
        ];
        assert_eq!(
            &expected,
            &dice_multisets(Counts::from_rolls(&[1, 1, 3, 5]), 2)
        );

        // when all the elements are unique, this is just the normal set
        // combinations
        for k in 0..=5 {
            assert_eq!(
                num_combinations(5, k) as usize,
                dice_multisets(Counts::from_rolls(&[1, 2, 3, 4, 5]), k as u8).len(),
            );
        }
    }

    #[test]
    fn test_gen_actions() {
        fn actions(rolls: &[u8]) -> Vec<Action> {
            let state = State::new(0, DEFAULT_TARGET_SCORE, Counts::from_rolls(rolls));
            let mut actions = state.actions();
            actions.sort_unstable();
            actions
        }

        macro_rules! roll {
            () => (
                Action::Roll(Counts::new())
            );
            ($($x:expr),+ $(,)?) => (
                Action::Roll(Counts::from_rolls(&[ $($x),+ ]))
            );
        }

        // no scoring dice
        assert_eq!(Vec::<Action>::new(), actions(&[]));
        assert_eq!(Vec::<Action>::new(), actions(&[3]));
        assert_eq!(Vec::<Action>::new(), actions(&[3, 4]));
        assert_eq!(Vec::<Action>::new(), actions(&[3, 3, 6, 6]));

        // with scoring dice
        use super::Action::Pass;
        assert_eq!(vec![Pass, roll![1]], actions(&[1, 3]));
        assert_eq!(vec![Pass, roll![5]], actions(&[2, 3, 5, 6]));
        assert_eq!(vec![Pass, roll![1], roll![1, 5]], actions(&[1, 2, 3, 5, 6]));
        assert_eq!(vec![Pass, roll![1], roll![1, 1]], actions(&[1, 1, 3]));
        assert_eq!(vec![Pass, roll![1], roll![1, 1]], actions(&[1, 1, 3, 3]));
        assert_eq!(
            vec![
                Pass,
                roll![1],
                roll![1, 1],
                roll![1, 1, 3, 3, 3],
                roll![1, 3, 3, 3],
                roll![3, 3, 3]
            ],
            actions(&[1, 1, 3, 3, 3])
        );

        // should include hold (straight ++ 5) action
        assert_eq!(
            vec![
                Pass,
                roll![2, 3, 4, 5, 5, 6],
                roll![2, 3, 4, 5, 6],
                roll![5],
                roll![5, 5],
            ],
            actions(&[2, 3, 4, 5, 5, 6]),
        );

        assert_eq!(
            vec![
                Pass,
                roll![1],
                roll![1, 1],
                roll![1, 1, 2, 3, 4, 5],
                roll![1, 1, 5],
                roll![1, 2, 3, 4, 5],
            ],
            actions(&[1, 1, 2, 3, 4, 5]),
        );
    }

    #[test]
    fn test_counts_probability() {
        assert_relative_eq!(
            3.0 / ((6.0_f64).powf(3.0)),
            Counts::from_rolls(&[1, 1, 3]).p_roll()
        );

        for n in 1..=6 {
            assert_relative_eq!(
                1.0,
                AllDiceMultisetsIter::new(n)
                    .map(|counts| counts.p_roll())
                    .sum::<f64>(),
                epsilon = 1e-10,
            );
        }
    }

    #[test]
    fn test_p_rv_lte_itself() {
        let cdf = Array1::from_vec(vec![0.1, 0.6, 1.0]);
        assert_relative_eq!(0.71, p_rv_lte_itself(cdf.view()));
    }

    // #[test]
    // #[ignore]
    // fn test_actions_by_expected_value() {
    //     let state = State::new(0, Counts::from_rolls(&[5, 5, 3, 3, 4]));
    //
    //     for tuple in state.actions_by_expected_value().1 {
    //         println!("{:?}", tuple);
    //     }
    // }
}
