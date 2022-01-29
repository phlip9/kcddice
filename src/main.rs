use approx::relative_eq;
use std::{
    cell::Cell,
    cmp::{self, min},
    collections::HashMap,
    env, fmt,
};

const NUM_FACTORIALS: usize = 13;

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

const FACTORIAL_LT: [u32; NUM_FACTORIALS] = precompute_factorials();

const fn factorial(n: u32) -> u32 {
    FACTORIAL_LT[n as usize]
}

#[cfg(test)]
const fn num_combinations(n: u32, k: u32) -> u32 {
    factorial(n) / (factorial(k) * factorial(n - k))
}

#[cfg(test)]
#[inline]
const fn num_multisets(n: u32, k: u32) -> u32 {
    num_combinations(n + k - 1, k)
}

#[inline]
pub fn total_cmp_f64(a: &f64, b: &f64) -> cmp::Ordering {
    let mut left = a.to_bits() as i64;
    let mut right = b.to_bits() as i64;

    left ^= (((left >> 63) as u64) >> 1) as i64;
    right ^= (((right >> 63) as u64) >> 1) as i64;

    left.cmp(&right)
}

#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
struct Counts(pub u32);

impl Counts {
    #[inline]
    fn new() -> Self {
        Self(0)
    }

    #[cfg(test)]
    #[inline]
    fn from_roll(roll: u8, count: u8) -> Self {
        let mut counts = Self::new();
        counts.add_count(roll, count);
        counts
    }

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
    fn is_empty(&self) -> bool {
        self.0 == 0
    }

    // The number of dice in this set
    #[inline]
    fn len(&self) -> u8 {
        let mut len = 0;
        for roll in 1..=6 {
            len += self.get_count(roll);
        }
        len
    }

    #[inline]
    fn get_count(&self, roll: u8) -> u8 {
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

    // #[inline]
    // fn sub_count(&mut self, roll: u8, count: u8) {
    //     self.0 -= ((count as u32) & 0x7) << (4 * (roll as u32));
    // }

    fn add(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }

    // compute the score of a set of dice. will still work with non-scoring dice
    // mixed in.
    fn score(&self) -> u16 {
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
            return 750;
        } else if c1 > 0 && c2 > 0 && c3 > 0 && c4 > 0 && c5 > 0 {
            return 500;
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

    fn exact_score(&self) -> u16 {
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
        // [1 1 1 1 1 1]
        if c1 == 1 && c2 == 1 && c3 == 1 && c4 == 1 && c5 == 1 && c6 == 1 {
            return 1500;
        // [0 1 1 1 1 1]
        } else if c1 == 0 && c2 == 1 && c3 == 1 && c4 == 1 && c5 == 1 && c6 == 1 {
            return 750;
        // [1 1 1 1 1 0]
        } else if c1 == 1 && c2 == 1 && c3 == 1 && c4 == 1 && c5 == 1 && c6 == 0 {
            return 500;
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

    #[inline]
    fn is_bust(&self) -> bool {
        relative_eq!(self.score() as f64, 0.0)
    }

    // Return the probability of rolling this set of dice
    // n = number of dice in the set
    // P = n! / (6^n * ∏_{i∈[1,6]} c_i!)
    //     where c_i is the count of the i'th dice in the set
    fn p_roll(&self) -> f64 {
        let n = self.len();

        let prod: u32 = (1..=6)
            .map(|roll| factorial(self.get_count(roll) as u32))
            .product();

        let m = 6_u32.pow(n as u32);

        (factorial(n as u32) as f64) / ((prod * m) as f64)
    }

    fn to_rolls(&self) -> Vec<u8> {
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

// An `Iterator` over combinations (with replacement) of _all_ dice roll outcomes
struct AllDiceMultisetsIter {
    // the _next_ combination we'll output (unless we're done).
    counts: Counts,
    // total number of dice rolls per combination.
    total_dice: u8,
    // set to `true` when we're done generating.
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

// Return all possible sub-multisets of size `ndice` of a given set of dice rolls.
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

        // special case: |S| = n means just one combination: S
        if elts_left == ndice {
            cb(acc.add(left));
            return;
        }

        // will never have enough; return empty
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

// game state {
//   target_total
//   other_total
//   my_total
//   my_round_total
//   dice = {i | 1 <= i <= 6} x n
// }

// plan:
// 1. probability of given rolls combination
// 2. game state
// 3. given a current game state, want to compute action with maximum win
//    probability assuming minimax etc...
// 4. for now, probably just try to compute action with greatest expected reward
//    for a full round?

// dice = [1, 1, 3, 5, 6]
//
// [1] [3] [5] [6]
// [1, 1] [1, 3] [1, 5] [1, 6] [3, 5] [3, 6] [5, 6]
// [1, 1, 3] [1, 1, 5] [1, 1, 6] [1, 3, 5] [1, 3, 6] [1, 5, 6] [3, 5, 6]

// suppose we can roll 3 dice
// what is the probability we roll the exact sequence [1, 1, 3]?
// if we care about the exact sequence (i.e., order matters) then it's (1/6)^3,
// i.e., P(roll 1, roll 1, roll 3) = P(roll 1) * P(roll 1) * P(roll 3) = (1/6)^3
//
// what is the probability we roll [1, 1, 3] (in any order)?
// p1 p1 p3
// p1 p3 p1
// p3 p1 p1

// game is not guaranteed to terminate, need state cache?

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum Action {
    // Player passes their turn, adding their current round points to their total.
    Pass,
    // Player can choose to hold some non-empty subset of scoring dice and roll
    // the rest. The held dice score is added to their round total.
    Roll(Counts),
}

// fn states_expected_value(states: &[(State, f64)]) -> f64 {
//     states
//         .into_iter()
//         .map(|(state, p_state)| p_state * (state.my_total as f64))
//         .sum()
// }

struct Context {
    action_value_cache: HashMap<(State, Action), f64>,
    cache_hits: Cell<u64>,
    cache_misses: Cell<u64>,
}

impl Context {
    fn new() -> Self {
        Context {
            cache_hits: Cell::new(0),
            cache_misses: Cell::new(0),
            action_value_cache: HashMap::new(),
        }
    }

    #[inline]
    fn peek_cache(&self, key: &(State, Action)) -> Option<f64> {
        let out = self.action_value_cache.get(key).copied();
        if out.is_some() {
            self.cache_hits.set(self.cache_hits() + 1);
        } else {
            self.cache_misses.set(self.cache_misses() + 1);
        }
        out
    }

    #[inline]
    fn fill_cache(&mut self, key: (State, Action), value: f64) {
        self.action_value_cache.insert(key, value);
    }

    #[inline]
    fn cache_hits(&self) -> u64 {
        self.cache_hits.get()
    }

    #[inline]
    fn cache_misses(&self) -> u64 {
        self.cache_misses.get()
    }

    #[inline]
    fn cache_queries(&self) -> u64 {
        self.cache_hits() + self.cache_misses()
    }

    #[inline]
    fn cache_hit_rate(&self) -> f64 {
        (self.cache_hits() as f64) / (self.cache_queries() as f64)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
struct State {
    // the set of dice we just rolled
    pub rolled_dice: Counts,
    // pub target_total: u16,
    // pub other_total: u16,
    // pub my_total: u16,
    pub my_round_total: u16,
}

impl State {
    fn new(/* target_total: u16, */ rolled_dice: Counts) -> Self {
        Self {
            // target_total,
            // other_total: 0,
            // my_total: 0,
            my_round_total: 0,
            rolled_dice,
        }
    }

    // fn new_with_round_total(my_round_total: u16, rolled_dice: Counts) -> Self {
    //     Self {
    //         // my_total: 0,
    //         my_round_total,
    //         rolled_dice,
    //     }
    // }

    fn actions(&self) -> Vec<Action> {
        // if this dice roll has no scores whatsoever, then there are no actions
        // (our turn has ended).
        if self.rolled_dice.score() == 0 {
            return Vec::new();
        }

        // can hold at most 6 - k - 1 dice (can't hold all the dice left)
        let max_num_holds = self.rolled_dice.len() - 1;

        let possible_holds = (1..=max_num_holds)
            .flat_map(|ndice| dice_multisets(self.rolled_dice, ndice))
            // can only hold
            .filter(|held_dice| held_dice.exact_score() > 0)
            .map(|held_dice| Action::Roll(held_dice))
            .collect::<Vec<_>>();

        let mut actions_vec = possible_holds;

        // can always pass if we have some scores
        actions_vec.push(Action::Pass);

        actions_vec
    }

    // apply an action and return the set of possible end turn states along with
    // the probability of reaching each result state
    // fn apply_action(&self, action: Action) -> Vec<(State, f64)> {
    //     println!("({:?}).apply_action({:?})", self, action);
    //
    //     match action {
    //         Action::Pass => {
    //             let end_state = State {
    //                 my_total: self.my_total + self.my_round_total + self.rolled_dice.score(),
    //                 my_round_total: 0,
    //                 rolled_dice: Counts::new(),
    //             };
    //             println!("\t => pass end state: {:?}", end_state);
    //             vec![(end_state, 1.0)]
    //         }
    //         Action::Roll(held_dice) => {
    //             // we have this many dice left to roll
    //             let ndice_left = self.rolled_dice.len() - held_dice.len();
    //
    //             // fold the held dice score into the round total
    //             // the rolled_dice is just an empty placeholder at this point
    //             let partial_state = State {
    //                 my_total: self.my_total,
    //                 my_round_total: self.my_round_total + held_dice.exact_score(),
    //                 rolled_dice: Counts::new(),
    //             };
    //
    //             println!(
    //                 "\theld_dice: {:?}\n\t=> partial_state: {:?}",
    //                 held_dice, partial_state
    //             );
    //
    //             let mut out = Vec::new();
    //
    //             // for all possible dice rolls
    //             for next_roll in AllDiceMultisetsIter::new(ndice_left) {
    //                 let p_roll = next_roll.p_roll();
    //
    //                 println!("\t\tnext_roll: {:?}, p_roll: {}", next_roll, p_roll);
    //
    //                 let mut next_state = partial_state;
    //                 next_state.rolled_dice = next_roll;
    //
    //                 let next_actions = next_state.actions();
    //                 for next_action in next_actions {
    //                     // recurse; this will return the final turn end states
    //                     // with unconditioned probabilities
    //                     let mut end_states = next_state.apply_action(next_action);
    //
    //                     // condition on the roll probability
    //                     for (_, p_end_state) in &mut end_states {
    //                         *p_end_state *= p_roll;
    //                     }
    //
    //                     println!("end_states: {:?}", end_states);
    //
    //                     out.extend(end_states);
    //                 }
    //             }
    //
    //             out
    //         }
    //     }
    // }

    // assuming we take a particular action on this game state, what is the
    // expected value (i.e., P * my_total)?
    fn action_expected_value(
        &self,
        ctxt: &mut Context,
        action: Action, /*, depth: usize */
    ) -> f64 {
        // ACTIONS_EXPLORED.fetch_add(1, Ordering::Relaxed);
        if let Some(action_value) = ctxt.peek_cache(&(*self, action)) {
            return action_value;
        }

        // let pad = "    ".repeat(depth);
        // println!("{}({:?}).action_expected_value({:?})", pad, self, action);

        let expected_value = match action {
            Action::Pass => {
                // let end_state = State {
                //     // my_total: self.my_total + self.my_round_total + self.rolled_dice.score(),
                //     my_round_total: self.my_round_total + self.rolled_dice.score(),
                //     rolled_dice: Counts::new(),
                // };
                (self.my_round_total + self.rolled_dice.score()) as f64
                // println!("{}pass end state: {:?}", pad, end_state);
                // end_state.my_total as f64
            }
            Action::Roll(held_dice) => {
                // we have this many dice left to roll
                let ndice_left = self.rolled_dice.len() - held_dice.len();

                // fold the held dice score into the round total
                // the rolled_dice is just an empty placeholder at this point
                let partial_state = State {
                    // my_total: self.my_total,
                    my_round_total: self.my_round_total + held_dice.exact_score(),
                    rolled_dice: Counts::new(),
                };

                // println!(
                //     "{}held_dice: {:?} => partial_state: {:?}",
                //     pad, held_dice, partial_state
                // );

                let mut expected_value = 0.0_f64;

                // for all possible dice rolls
                for next_roll in AllDiceMultisetsIter::new(ndice_left) {
                    let p_roll = next_roll.p_roll();

                    // println!("{}next_roll: {:?} w.p. {}", pad, next_roll, p_roll);

                    let mut next_state = partial_state;
                    next_state.rolled_dice = next_roll;

                    // want to maximize expected value; choose action with
                    // greatest expected value

                    // let next_actions_by_value = next_state
                    //     .actions()
                    //     .into_iter()
                    //     .map(|next_action| {
                    //         (
                    //             next_state.action_expected_value(next_action, depth + 1),
                    //             next_action,
                    //         )
                    //     })
                    //     .collect::<Vec<_>>();
                    //
                    // for (next_action_value, next_action) in &next_actions_by_value {
                    //     println!(
                    //         "{}{}possible_action: {:?}, cond value: {}",
                    //         pad, pad, next_action, next_action_value
                    //     );
                    // }
                    //
                    // let best_action_value = next_actions_by_value
                    //     .into_iter()
                    //     .map(|(value, _)| value)
                    //     .max_by(total_cmp_f64)
                    //     .unwrap_or(0.0);
                    //
                    // println!(
                    //     "{}best cond value: {}, best value: {}",
                    //     pad,
                    //     best_action_value,
                    //     p_roll * &best_action_value
                    // );

                    let best_action_value = next_state
                        .actions()
                        .into_iter()
                        .map(|next_action| {
                            next_state.action_expected_value(
                                ctxt,
                                next_action,
                                /* depth + 1 */
                            )
                        })
                        .max_by(total_cmp_f64)
                        .unwrap_or(0.0);

                    expected_value += p_roll * best_action_value;
                }

                expected_value
            }
        };

        // println!(
        //     "{}({:?}).action_expected_value({:?}) ====> {}",
        //     pad, self, action, expected_value
        // );

        ctxt.fill_cache((*self, action), expected_value);

        expected_value
    }

    fn action_p_bust(&self, action: Action) -> f64 {
        match action {
            Action::Pass => 0.0,
            Action::Roll(held_dice) => {
                // we have this many dice left to roll
                let ndice_left = self.rolled_dice.len() - held_dice.len();

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

    // for each possible action, given that I chose that action, what is the
    // expected turn score?
    fn actions_by_expected_value(&self) -> (Context, Vec<(Action, f64, f64)>) {
        // ACTIONS_EXPLORED.store(0, Ordering::Relaxed);
        let mut ctxt = Context::new();

        let mut actions_values = self
            .actions()
            .into_iter()
            .map(|action| {
                (
                    action,
                    self.action_expected_value(&mut ctxt, action),
                    self.action_p_bust(action),
                )
            })
            .collect::<Vec<_>>();

        actions_values.sort_unstable_by(|(_, v1, _), (_, v2, _)| total_cmp_f64(v1, v2).reverse());
        (ctxt, actions_values)
    }

    // fn actions_by_p_bust(&self) -> Vec<(Action, f64)> {
    //     let mut actions_p_busts = self
    //         .actions()
    //         .into_iter()
    //         .map(|action| (action, self.action_p_bust(action)))
    //         .collect::<Vec<_>>();
    //
    //     actions_p_busts.sort_unstable_by(|(_, v1), (_, v2)| total_cmp_f64(v1, v2).reverse());
    //     actions_p_busts
    // }
}

fn usage() -> &'static str {
    "kcddice rolled-dice \
    \
    example: kcddice values [1,1,3,4,6]
    "
}

fn parse_dice(s: &str) -> Result<Counts, String> {
    let mut counts = Counts::new();

    let s = s.trim_start_matches('[');
    let s = s.trim_end_matches(']');

    for roll_str in s.split(',') {
        let roll = roll_str
            .parse::<u8>()
            .map_err(|err| format!("dice roll is not a valid integer: {}", err))?;
        if 1 <= roll && roll <= 6 {
            counts.add_count(roll, 1);
        } else {
            return Err(format!("roll is out of range [1, 6]: {}", roll));
        }
    }

    Ok(counts)
}

// enum Command {
//     Values,
//     PBust,
// }

fn parse_args(args: &[String]) -> Result<State, String> {
    match args {
        [rolled_dice] => {
            // let command = if command == "values" {
            //     Command::Values
            // } else if command == "pbust" {
            //     Command::PBust
            // } else {
            //     return Err(format!("Unrecognized command: {}", command));
            // };

            // let round_total: u16 = round_total
            //     .parse()
            //     .map_err(|err| format!("Parse error: round-total is not a valid u16: {}", err))?;

            let rolled_dice = parse_dice(rolled_dice)?;
            // let state = State::new_with_round_total(round_total, rolled_dice);
            let state = State::new(rolled_dice);

            Ok(state)
        }
        _ => Err(format!("Unexpected number of arguments: {}", args.len())),
    }
}

fn main() {
    let args = env::args().skip(1).collect::<Vec<_>>();

    match parse_args(&args) {
        Ok(state) => {
            println!("");
            // let actions_values = match command {
            //     Command::Values => state.actions_by_expected_value(),
            //     Command::PBust => state.actions_by_p_bust(),
            // };

            let (ctxt, actions_values) = state.actions_by_expected_value();

            println!(
                "{:>10}  {:<20} {:>5} {:>6}",
                "action", "held dice", "expv", "pbust"
            );

            let len = actions_values.len();
            for (action, value, p_bust) in actions_values.into_iter().take(10) {
                match action {
                    Action::Pass => println!(
                        "{:>10}  {} {:>5.1} {:>6.2}",
                        "pass",
                        " ".repeat(20),
                        value,
                        p_bust
                    ),
                    Action::Roll(held_dice) => {
                        let held_dice_str = format!("{:?}", held_dice);
                        println!(
                            "{:>10}  {:<20} {:>5.1} {:>6.2}",
                            "hold dice", held_dice_str, value, p_bust
                        )
                    }
                }
            }
            if len > 10 {
                println!("{:>10}  (x {})", "...", len - 10);
            }

            println!("\n{:>16}: {}", "cache queries", ctxt.cache_queries());
            println!("{:>16}: {:.3}", "cache hit rate", ctxt.cache_hit_rate());
        }
        Err(err) => {
            println!("Invalid arguments: {}", err);
            println!("\nusage: {}", usage());
        }
    }
}

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
        fn rec(cb: &mut impl FnMut(Counts), mut counts: Counts, current_dice: u8, ndice: u8) {
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
            let state = State::new(Counts::from_rolls(rolls));
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
        assert_eq!(
            vec![Pass, roll![1], roll![1, 5], roll![5]],
            actions(&[1, 2, 3, 5, 6])
        );
        assert_eq!(vec![Pass, roll![1], roll![1, 1]], actions(&[1, 1, 3]));
        assert_eq!(vec![Pass, roll![1], roll![1, 1]], actions(&[1, 1, 3, 3]));
        assert_eq!(
            vec![
                Pass,
                roll![1],
                roll![1, 1],
                roll![1, 3, 3, 3],
                roll![3, 3, 3]
            ],
            actions(&[1, 1, 3, 3, 3])
        );
    }

    #[test]
    fn test_counts_probability() {
        assert_relative_eq!(
            (3 as f64) / ((6.0_f64).powf(3.0)),
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
    fn test_actions_by_expected_value() {
        let state = State::new(Counts::from_rolls(&[5, 5, 3, 3, 4]));

        for tuple in state.actions_by_expected_value() {
            println!("{:?}", tuple);
        }
    }
}
