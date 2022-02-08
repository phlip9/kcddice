use crate::{
    dice::{DiceCounts, DiceVec, DieKindCounts},
    search::{p_rv_lte_itself, Action, Context, MarkovMatrix, NormalizedStateAction, State},
    search2, DEFAULT_TARGET_SCORE,
};
use pico_args::{self, Arguments};
use std::time::Instant;
use tabular::{row, Table};

pub trait Command: Sized {
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
pub struct BestActionCommand {
    round_total: u16,
    target_score: u16,
    rolled_dice: DiceVec,
    all_dice: DieKindCounts,
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

    · --die-kinds / -k [s:n,hk:m,o:n,..] (default: [s:6])
      The kinds of dice we're using this game.
      For example, using 3 Standard, 2 HeavenlyKingdom, 1 Odd would be formatted
      as -k [s:3,hk:2,o:1]
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
            all_dice: args
                .opt_value_from_str(["--die-kinds", "-k"])
                .map_err(|err| err.to_string())?
                .unwrap_or(DieKindCounts::all_standard(6)),
        };

        cmd.all_dice
            .validate_init_set(&DieKindCounts::from_dice_vec(cmd.rolled_dice))?;

        let remaining = args.finish();
        if !remaining.is_empty() {
            return Err(format!("unexpected arguments left: '{:?}'", remaining));
        }

        Ok(cmd)
    }

    fn run(self) {
        let state = search2::State::new(self.round_total, self.target_score, self.rolled_dice);
        let mut ctxt = search2::Context::new();
        ctxt.set_all_dice(self.all_dice);

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
                search2::Action::Pass => ("pass", String::new()),
                search2::Action::Roll(held_dice) => ("hold dice", format!("{:?}", held_dice)),
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
pub struct ScoreDistrCommand {
    round_total: u16,
    target_score: u16,
    dice_left: DieKindCounts,
    all_dice: DieKindCounts,
}

impl Command for ScoreDistrCommand {
    const USAGE: &'static str = "\
kcddice score-distr - TODO

USAGE:
    kcddice score-distr [option ...] <round-total> <target-score> <dice-left>
";

    fn parse_from_args(mut args: Arguments) -> Result<Self, String> {
        Self::maybe_help(&mut args);

        let cmd = Self {
            round_total: args.free_from_str().map_err(|err| err.to_string())?,
            target_score: args.free_from_str().map_err(|err| err.to_string())?,
            dice_left: args.free_from_str().map_err(|err| err.to_string())?,
            all_dice: args
                .opt_value_from_str(["--die-kinds", "-k"])
                .map_err(|err| err.to_string())?
                .unwrap_or(DieKindCounts::all_standard(6)),
        };

        cmd.all_dice.validate_init_set(&cmd.dice_left)?;

        let remaining = args.finish();
        if !remaining.is_empty() {
            return Err(format!("unexpected arguments left: '{:?}'", remaining));
        }

        Ok(cmd)
    }

    fn run(self) {
        let qstate = search2::NormalizedStateAction::new(
            self.round_total,
            self.target_score,
            self.dice_left,
        );

        let mut ctxt = search2::Context::new();
        ctxt.set_all_dice(self.all_dice);

        let start_time = Instant::now();
        let score_pmf = qstate.score_distribution(&mut ctxt);
        let search_duration = start_time.elapsed();

        let mut dense_score_pmf = score_pmf.clone().into_vec();
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
pub struct MarkovMatrixCommand {
    target_score: u16,
    all_dice: DieKindCounts,
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
            all_dice: args
                .opt_value_from_str(["--die-kinds", "-k"])
                .map_err(|err| err.to_string())?
                .unwrap_or(DieKindCounts::all_standard(6)),
        };

        cmd.all_dice.validate_init_set(&cmd.all_dice)?;

        let remaining = args.finish();
        if !remaining.is_empty() {
            return Err(format!("unexpected arguments left: '{:?}'", remaining));
        }

        Ok(cmd)
    }

    fn run(self) {
        let start_time = Instant::now();
        let matrix = search2::MarkovMatrix::from_optimal_policy(self.all_dice, self.target_score);
        let search_duration = start_time.elapsed();

        println!("{:?}", matrix);
        println!("\nsearch duration  {:.2?}", search_duration);
    }
}

#[derive(Debug)]
pub struct TurnsCdfCommand {
    target_score: u16,
    max_num_turns: usize,
    our_dice: DieKindCounts,
    their_dice: DieKindCounts,
}

impl Command for TurnsCdfCommand {
    // TODO(philiphayes): fill out
    const USAGE: &'static str = "";

    fn parse_from_args(mut args: Arguments) -> Result<Self, String> {
        Self::maybe_help(&mut args);

        let cmd = Self {
            target_score: args.free_from_str().map_err(|err| err.to_string())?,
            max_num_turns: args.free_from_str().map_err(|err| err.to_string())?,
            our_dice: args
                .opt_value_from_str(["--our-die-kinds", "-o"])
                .map_err(|err| err.to_string())?
                .unwrap_or(DieKindCounts::all_standard(6)),
            their_dice: args
                .opt_value_from_str(["--their-die-kinds", "-t"])
                .map_err(|err| err.to_string())?
                .unwrap_or(DieKindCounts::all_standard(6)),
        };

        cmd.our_dice.validate_init_set(&cmd.our_dice)?;
        cmd.their_dice.validate_init_set(&cmd.their_dice)?;

        let remaining = args.finish();
        if !remaining.is_empty() {
            return Err(format!("unexpected arguments left: '{:?}'", remaining));
        }

        Ok(cmd)
    }

    fn run(self) {
        let start_time = Instant::now();

        let our_matrix =
            search2::MarkovMatrix::from_optimal_policy(self.our_dice, self.target_score);
        let our_turns_cdf = our_matrix.turns_to_win_cdf(self.max_num_turns);

        let their_turns_cdf = if self.our_dice == self.their_dice {
            our_turns_cdf.clone()
        } else {
            let their_matrix =
                search2::MarkovMatrix::from_optimal_policy(self.their_dice, self.target_score);
            their_matrix.turns_to_win_cdf(self.max_num_turns)
        };

        let search_duration = start_time.elapsed();

        println!("their turns CDF");
        println!("turn\tcumulative probability");

        for turn in 1..=self.max_num_turns {
            println!("{}\t{}", turn, their_turns_cdf[turn - 1]);
        }

        println!("\nour turns CDF");
        println!("turn\tcumulative probability");

        for turn in 1..=self.max_num_turns {
            println!("{}\t{}", turn, our_turns_cdf[turn - 1]);
        }

        // Assuming we always go first and play optimally, what is our a priori
        // win probability against an opponent that also plays optimally, but
        // may have different dice?
        //
        // Assuming the our # turns to win is modeled as a r.v. X_1 ~ our_turns_cdf
        // and the opponent's is modeled by independent r.v. X_2 ~ their_turns_cdf,
        // the win probability is (mostly) the same as `Pr[X_1 <= X_2]`, i.e., the
        // probability we (independently) finish the game in less turns than the
        // opponent.
        //
        // This is useful for deciding how much to bet (e.g., following the
        // Kelley-Criterion).
        //
        // For the typical setup (target=4000, our dice=6 normal, their dice=6
        // normal), this is ≈0.562. The Kelley-Criterion optimal bet is then 12.5%
        // of your total wealth.
        let p_win = search2::p_rv1_lte_rv2(our_turns_cdf.view(), their_turns_cdf.view());

        let mut table = Table::new("{:>}  {:<}");
        table.add_row(row!("search_duration", format!("{:.2?}", search_duration)));
        table.add_row(row!("Pr[win]", format!("{}", p_win)));
        eprintln!("\n{}", table);
    }
}

#[derive(Debug)]
pub enum BaseCommand {
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
