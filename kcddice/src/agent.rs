use crate::{
    dice::{DiceVec, DieKindCounts, DieKindTable},
    search::{Action, Cache, Context, NormalizedStateAction, State},
};
use rand::{seq::SliceRandom, Rng};
use std::cmp;

pub struct GameState {
    max_score: u16,

    opponent_starting_dice_table: DieKindTable,
    opponent_starting_dice: DieKindCounts,

    our_starting_dice_table: DieKindTable,
    our_starting_dice: DieKindCounts,
}

#[derive(Copy, Clone)]
pub struct TotalState {
    opponent_total_score: u16,
    our_total_score: u16,
}

pub struct RoundState<'a> {
    game_state: &'a GameState,
    total_state: TotalState,
    our_qstate: NormalizedStateAction,
}

impl<'a> RoundState<'a> {
    fn init_round_state(game_state: &'a GameState, total_state: TotalState) -> Self {
        let our_qstate = NormalizedStateAction::init_state(
            game_state.max_score - total_state.our_total_score,
            game_state.our_starting_dice,
        );
        Self {
            game_state,
            total_state,
            our_qstate,
        }
    }

    #[inline]
    fn our_dice_table(&self) -> DieKindTable {
        self.game_state.our_starting_dice_table
    }

    #[inline]
    fn our_starting_dice(&self) -> DieKindCounts {
        self.game_state.our_starting_dice
    }

    #[inline]
    fn our_state(&self, rolled_dice: DiceVec) -> State {
        self.our_qstate.into_state(rolled_dice)
    }

    fn roll_dice<R: Rng>(&self, rng: &mut R) -> DiceVec {
        self.our_qstate
            .dice_left
            .roll_dice(rng, self.our_dice_table())
    }

    /// Applys an Agent's action to the current state. Returns `Some(round_total)`
    /// if the agent passed.
    fn apply_action(&mut self, state: State, action: Action) -> Option<u16> {
        match action {
            Action::Pass => Some(state.my_round_total + state.rolled_dice.score()),
            Action::Roll(held_dice) => {
                self.our_qstate = NormalizedStateAction::from_state_roll_action(
                    self.our_starting_dice(),
                    state,
                    held_dice,
                );
                None
            }
        }
    }
}

// agent

// top-level context/parameters

pub trait Agent {
    fn action(&mut self, round_state: &RoundState, state: State) -> Action;
}

pub struct AlwaysPassAgent;

impl Agent for AlwaysPassAgent {
    fn action(&mut self, _round_state: &RoundState, _state: State) -> Action {
        Action::Pass
    }
}

pub struct RandomActionAgent<R> {
    rng: R,
}

impl<R: Rng> Agent for RandomActionAgent<R> {
    fn action(&mut self, _round_state: &RoundState, state: State) -> Action {
        let actions = state.actions();
        *actions
            .choose(&mut self.rng)
            .expect("no actions available!")
    }
}

pub struct ExpectedValueMaxAgent {
    context: Context,
}

impl Agent for ExpectedValueMaxAgent {
    fn action(&mut self, _round_state: &RoundState, state: State) -> Action {
        state
            .actions_by_expected_value(&mut self.context)
            .first()
            .expect("no actions available!")
            .action
    }
}

//

enum RoundOutcome {}

struct RoundRunner<R> {
    rng: R,
}

impl<R: Rng> RoundRunner<R> {
    // return the current agent's round total after sampling a round.
    fn run_round<A: Agent>(
        &mut self,
        agent: &mut A,
        game_state: &GameState,
        total_state: TotalState,
    ) -> u16 {
        let mut round_state = RoundState::init_round_state(game_state, total_state);
        let mut niters = 0;

        loop {
            debug_assert!(niters <= 50);

            let rolled_dice = round_state.roll_dice(&mut self.rng);
            if rolled_dice.is_bust() {
                // agent went bust; agent earned no points this round.
                return 0;
            }

            // ask the agent what they want to do
            let new_state = round_state.our_qstate.into_state(rolled_dice);
            let agent_action = agent.action(&round_state, new_state);
            let maybe_passed = round_state.apply_action(new_state, agent_action);

            if let Some(round_total) = maybe_passed {
                // agent passed; add their round total to their total score
                // and return for the next agent to play.
                return cmp::min(round_total, game_state.max_score);
            }

            niters += 1;
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{
        parse::{prop::arb_dice_set, DiceSet},
        prop::{arb_rng, niters, small_rng},
        search::{prop::arb_score, MarkovMatrix, ScorePMF},
        stats::{multinomial_test, multinomial_test2},
    };
    use claim::{assert_gt, assert_le};
    use ndarray::Array1;
    use proptest::prelude::*;
    use rand::SeedableRng;
    use rand_xoshiro::Xoroshiro64Star;
    use std::{cmp, collections::HashMap, ops::AddAssign, str::FromStr};

    // `stderr_mean` is the standard error of the mean (i.e., standard deviation
    // of the sample mean, possibly an estimate).
    //
    // `std_confidence` is in terms of standard sigmas from the standard mean 0.
    // it is a function of our tolerated failure probability, where we accidentally
    // sample a mean that is outside our confidence interval, despite the samples
    // actually coming from our population. as our tolerated failure probability
    // decreases, our required confidence bound increases.
    //
    // this function returns the maximum tolerated absolute deviation of the
    // observed sample mean from the true population mean, E = |μ - μ_x|.
    fn max_tolerated_abs_deviation(stderr_mean: f64, std_confidence: f64) -> f64 {
        stderr_mean * std_confidence
    }

    fn sample_emp_score_distr<A: Agent>(
        num_trials: usize,
        round_runner: &mut RoundRunner<impl Rng>,
        agent: &mut A,
        game_state: &GameState,
        total_state: TotalState,
    ) -> (ScorePMF, Array1<f64>) {
        let target_score = game_state.max_score - total_state.our_total_score;
        let mut emp_score_counts = Array1::<f64>::zeros(MarkovMatrix::num_states(target_score));
        for _ in 0..num_trials {
            let round_total = time!(
                "run_round",
                round_runner.run_round(agent, game_state, total_state)
            );
            let idx = MarkovMatrix::s2i(round_total);
            emp_score_counts[idx] += 1.0;
        }
        let iter = emp_score_counts
            .indexed_iter()
            .map(|(idx, &count)| (MarkovMatrix::i2s(idx), count as usize));
        let pmf = ScorePMF::from_counts_iter(num_trials, iter);
        (pmf, emp_score_counts)
    }

    #[test]
    fn test_run_round_mc_vs_score_distr() {
        let num_samples: usize = 1_000;
        let dice_set = DiceSet::from_str("l:3 o:2 e:1").unwrap();
        let target_score = 4000;
        let rng = small_rng(0xf00ba4);
        // println!("dice_set: {dice_set:?}, target_score: {target_score}");

        let (dice_table, starting_dice) = dice_set.to_compact_form();

        let game_state = GameState {
            max_score: target_score,
            our_starting_dice: starting_dice,
            our_starting_dice_table: dice_table,
            // not relevant
            opponent_starting_dice: DieKindCounts::new(),
            opponent_starting_dice_table: DieKindTable::all_standard(),
        };
        let total_state = TotalState {
            our_total_score: 0,
            // not relevant
            opponent_total_score: 0,
        };

        // the true population score distribution
        let mut context = Context::new(dice_table, starting_dice);
        let round_total = 0;
        let qstate = NormalizedStateAction::new(round_total, target_score, starting_dice);
        let score_distr = time!(qstate.score_distribution(&mut context));

        let mut round_runner = RoundRunner { rng };

        // empirical score distribution for various agents
        let (expmax_score_distr, _expmax_score_counts) = time!(
            "sample_emp_score_distr",
            sample_emp_score_distr(
                num_samples,
                &mut round_runner,
                &mut ExpectedValueMaxAgent {
                    context: Context::new(dice_table, starting_dice),
                },
                &game_state,
                total_state,
            )
        );

        // expmax_score_distr.compare_pmfs(&score_distr, target_score);

        let p = score_distr.to_dense(target_score);
        let p_hat = expmax_score_distr.to_dense(target_score);

        let score_counts = &p * (num_samples as f64);

        // our null hypothesis H_0: expmax_score_distr ~ score_distr
        let p_value = multinomial_test(num_samples, p.view(), p_hat.view());
        // println!("g-test: p-value: {p_value_1}");

        assert_gt!(p_value, 0.01);
    }
}
