use crate::{dice::DiceCounts, total_cmp_f64};
use ndarray::{s, Array1, Array2, ArrayView1};
use std::{cell::Cell, cmp, collections::HashMap};

////////////
// Action //
////////////

/// An `Action` represents any of the possible actions the player can take from a
/// given round [`State`]. Note that a "bust" is simply the absence of any possible
/// actions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Action {
    /// If there is at least one scoring die, the player can pass their turn,
    /// ending their turn and scoring their current round total + the currently
    /// rolled dice score.
    Pass,
    /// If there is at least one scoring die and at least 2 dice left, the player
    /// can choose to hold some non-empty subset of scoring dice and re-roll the
    /// rest. The held dice score is added to their current round total.
    Roll(DiceCounts),
}

////////////////////////
// Evaluation Context //
////////////////////////

/// The `Context` is some additional evaluation state passed along while evaluating
/// `(state, action) -> expected value`. The evaluation should still be correct
/// even without the `Context`; rather, the `Context` should be used only for
/// optimizations or evaluation statistics.
pub struct Context {
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
    pub fn new() -> Self {
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
    pub fn actions_explored(&self) -> u64 {
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
    pub fn cache_size(&self) -> usize {
        self.action_value_cache.len()
    }

    #[inline]
    fn cache_hits(&self) -> u64 {
        self.cache_hits.get()
    }

    #[inline]
    pub fn cache_hit_rate(&self) -> f64 {
        (self.cache_hits() as f64) / (self.actions_explored() as f64 + 1.0)
    }

    #[inline]
    pub fn depth_prune_rate(&self) -> f64 {
        (self.depth_prunes() as f64) / (self.actions_explored() as f64 + 1.0)
    }

    #[inline]
    pub fn game_finished_rate(&self) -> f64 {
        (self.game_finished_prunes() as f64) / (self.actions_explored() as f64 + 1.0)
    }

    #[inline]
    pub fn joint_prob_prune_rate(&self) -> f64 {
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
pub struct NormalizedStateAction {
    /// The round total after applying the action.
    pub my_round_total: u16,
    /// The total score we're trying to hit. If we meet or exceed this value,
    /// then we win.
    pub target_score: u16,
    /// The number of dice left to reroll after applying the action.
    pub ndice_left: u8,
}

impl NormalizedStateAction {
    pub fn new(my_round_total: u16, target_score: u16, ndice_left: u8) -> Self {
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
    fn from_state_roll_action(state: State, held_dice: DiceCounts) -> Self {
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
    fn into_state(self, rolled_dice: DiceCounts) -> State {
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
        DiceCounts::all_multisets(self.ndice_left)
            .map(move |next_roll| (self.into_state(next_roll), next_roll.p_roll()))
    }

    /// Given a normalized `(State, Action)` pair, evaluate the expected value
    /// of applying the `Action` to the `State`, assuming we always choose
    /// subsequent actions by maximum expected value.
    pub fn expected_value(self, ctxt: &mut Context) -> f64 {
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

    pub fn score_distribution(self, ctxt: &mut Context) -> ScorePMF {
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
pub struct State {
    /// The set of dice the player just rolled.
    pub rolled_dice: DiceCounts,
    /// The player's current round score (accumulated from previous rolls in the turn).
    pub my_round_total: u16,
    /// The total score we're trying to hit. If we meet or exceed this value,
    /// then we win.
    pub target_score: u16,
}

impl State {
    pub fn new(my_round_total: u16, target_score: u16, rolled_dice: DiceCounts) -> Self {
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
            .flat_map(|ndice| self.rolled_dice.multisets(ndice))
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
                for next_roll in DiceCounts::all_multisets(ndice_left) {
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
    pub fn actions_by_expected_value(&self, ctxt: &mut Context) -> Vec<(Action, f64, f64)> {
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

///////////////
// Score PMF //
///////////////

#[derive(Clone, Debug)]
pub struct ScorePMF(HashMap<u16, f64>);

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

    pub fn into_vec(self) -> Vec<(u16, f64)> {
        self.0.into_iter().collect()
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

    pub fn expected_value(&self) -> f64 {
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

    pub fn total_mass(&self) -> f64 {
        self.0.values().sum()
    }
}

#[derive(Debug)]
pub struct MarkovMatrix(Array2<f64>);

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

    pub fn from_optimal_policy(target_score: u16) -> Self {
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

    pub fn turns_to_win_cdf(&self, max_num_turns: usize) -> Array1<f64> {
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
pub fn p_rv_lte_itself(cdf: ArrayView1<f64>) -> f64 {
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

///////////
// Tests //
///////////

#[cfg(test)]
mod test {
    use super::*;
    use crate::DEFAULT_TARGET_SCORE;
    use approx::assert_relative_eq;

    #[test]
    fn test_gen_actions() {
        fn actions(rolls: &[u8]) -> Vec<Action> {
            let state = State::new(0, DEFAULT_TARGET_SCORE, DiceCounts::from_rolls(rolls));
            let mut actions = state.actions();
            actions.sort_unstable();
            actions
        }

        macro_rules! roll {
            () => (
                Action::Roll(DiceCounts::new())
            );
            ($($x:expr),+ $(,)?) => (
                Action::Roll(DiceCounts::from_rolls(&[ $($x),+ ]))
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
    fn test_p_rv_lte_itself() {
        let cdf = Array1::from_vec(vec![0.1, 0.6, 1.0]);
        assert_relative_eq!(0.71, p_rv_lte_itself(cdf.view()));
    }

    // #[test]
    // #[ignore]
    // fn test_actions_by_expected_value() {
    //     let state = State::new(0, DiceCounts::from_rolls(&[5, 5, 3, 3, 4]));
    //
    //     for tuple in state.actions_by_expected_value().1 {
    //         println!("{:?}", tuple);
    //     }
    // }
}
