use crate::{
    parse,
    search::{p_rv1_lte_rv2, Action, Context, MarkovMatrix, NormalizedStateAction, State},
    DEFAULT_TARGET_SCORE,
};
use pico_args;
use std::{fmt, str::FromStr, time::Instant};
use tabular::{row, Table};

///////////////////////////
// String parser helpers //
///////////////////////////

fn parse_req<T>(s: &str) -> Result<T, String>
where
    T: FromStr,
    T::Err: fmt::Display,
{
    T::from_str(s).map_err(|err| err.to_string())
}

fn parse_opt<T>(opt_s: Option<&str>) -> Result<Option<T>, String>
where
    T: FromStr,
    T::Err: fmt::Display,
{
    opt_s
        .map(T::from_str)
        .transpose()
        .map_err(|err| err.to_string())
}

//////////////////////
// CLI Args Wrapper //
//////////////////////

pub struct Args(pico_args::Arguments);

impl Args {
    pub fn new(inner: pico_args::Arguments) -> Self {
        Self(inner)
    }

    fn subcommand(&mut self) -> Result<Option<String>, String> {
        self.0.subcommand().map_err(|err| err.to_string())
    }

    fn opt_value(&mut self, keys: impl Into<pico_args::Keys>) -> Result<Option<String>, String> {
        self.0
            .opt_value_from_fn(keys, |s| Result::<_, pico_args::Error>::Ok(s.to_owned()))
            .map_err(|err| err.to_string())
    }

    fn free_value(&mut self) -> Result<String, String> {
        self.0
            .free_from_fn(|s| Result::<_, pico_args::Error>::Ok(s.to_owned()))
            .map_err(|err| err.to_string())
    }

    fn expect_finished(self) -> Result<(), String> {
        let remaining = self.0.finish();
        if !remaining.is_empty() {
            Err(format!("unexpected arguments left: '{:?}'", remaining))
        } else {
            Ok(())
        }
    }

    fn maybe_help(&mut self, usage: &str) {
        if self.0.contains(["-h", "--help"]) {
            print!("{}", usage);
            std::process::exit(0);
        }
    }
}

///////////////////
// Command trait //
///////////////////

pub trait Command: Sized {
    const USAGE: &'static str;

    fn try_from_cli_args(args: Args) -> Result<Self, String>;
    fn run(self);
}

///////////////////////
// BestActionCommand //
///////////////////////

#[derive(Debug)]
pub struct BestActionCommand {
    round_total: u16,
    target_score: u16,
    rolled_dice: parse::DiceVec,
    all_dice: parse::DiceSet,
}

impl BestActionCommand {
    pub fn try_from_str_args(
        round_total: &str,
        target_score: Option<&str>,
        rolled_dice: &str,
        all_dice: Option<&str>,
    ) -> Result<Self, String> {
        let cmd = Self {
            round_total: parse_req(round_total)?,
            target_score: parse_opt(target_score)?.unwrap_or(DEFAULT_TARGET_SCORE),
            rolled_dice: parse_req(rolled_dice)?,
            all_dice: parse_opt(all_dice)?.unwrap_or_else(|| parse::DiceSet::all_standard(6)),
        };

        cmd.all_dice
            .validate_init_set(&cmd.rolled_dice.to_die_set())?;

        Ok(cmd)
    }
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

    fn try_from_cli_args(mut args: Args) -> Result<Self, String> {
        args.maybe_help(Self::USAGE);

        let target_score = args.opt_value(["-t", "--target-score"])?;
        let all_dice = args.opt_value(["-k", "--die-kinds"])?;
        let round_total = args.free_value()?;
        let rolled_dice = args.free_value()?;
        args.expect_finished()?;

        Self::try_from_str_args(
            &round_total,
            target_score.as_deref(),
            &rolled_dice,
            all_dice.as_deref(),
        )
    }

    fn run(self) {
        let (dice_table, all_dice) = self.all_dice.to_compact_form();
        let mut ctxt = Context::new(dice_table, all_dice);

        let rolled_dice = self.rolled_dice.to_compact_form(&dice_table);
        let state = State::new(self.round_total, self.target_score, rolled_dice);

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
                Action::Roll(held_dice) => (
                    "hold dice",
                    format!(
                        "{}",
                        parse::DiceVec::from_compact_form(&dice_table, held_dice)
                    ),
                ),
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
        table.add_row(row!(
            "game finished rate",
            format!("{:0.3}", ctxt.game_finished_rate()),
            "",
            ""
        ));
        table.add_heading("");
        table.add_row(row!(
            "action value cache size",
            format!(
                "{} ({})",
                ctxt.action_value_cache().cache_size(),
                ctxt.action_value_cache().cache_size_bytes()
            ),
            "",
            ""
        ));
        table.add_row(row!(
            "action value cache hit rate",
            format!(
                "{:0.3} (h: {}, m: {})",
                ctxt.action_value_cache().cache_hit_rate(),
                ctxt.action_value_cache().cache_hits(),
                ctxt.action_value_cache().cache_misses()
            ),
            "",
            ""
        ));
        table.add_heading("");
        table.add_row(row!(
            "actions cache size",
            format!(
                "{} ({})",
                ctxt.actions_cache().cache_size(),
                ctxt.actions_cache().cache_size_bytes()
            ),
            "",
            ""
        ));
        table.add_row(row!(
            "actions cache hit rate",
            format!(
                "{:0.3} (h: {}, m: {})",
                ctxt.actions_cache().cache_hit_rate(),
                ctxt.actions_cache().cache_hits(),
                ctxt.actions_cache().cache_misses()
            ),
            "",
            ""
        ));

        print!("\n{}", table);
    }
}

///////////////////////
// ScoreDistrCommand //
///////////////////////

#[derive(Debug)]
pub struct ScoreDistrCommand {
    round_total: u16,
    target_score: u16,
    dice_left: parse::DiceSet,
    all_dice: parse::DiceSet,
}

impl ScoreDistrCommand {
    fn try_from_str_args(
        round_total: &str,
        target_score: &str,
        dice_left: &str,
        all_dice: Option<&str>,
    ) -> Result<Self, String> {
        let cmd = Self {
            round_total: parse_req(round_total)?,
            target_score: parse_req(target_score)?,
            dice_left: parse_req(dice_left)?,
            all_dice: parse_opt(all_dice)?.unwrap_or_else(|| parse::DiceSet::all_standard(6)),
        };

        cmd.all_dice.validate_init_set(&cmd.dice_left)?;

        Ok(cmd)
    }
}

impl Command for ScoreDistrCommand {
    const USAGE: &'static str = "\
kcddice score-distr - TODO

USAGE:
    kcddice score-distr [option ...] <round-total> <target-score> <dice-left>
";

    fn try_from_cli_args(mut args: Args) -> Result<Self, String> {
        args.maybe_help(Self::USAGE);

        let all_dice = args.opt_value(["-k", "--die-kinds"])?;
        let round_total = args.free_value()?;
        let target_score = args.free_value()?;
        let dice_left = args.free_value()?;
        args.expect_finished()?;

        Self::try_from_str_args(&round_total, &target_score, &dice_left, all_dice.as_deref())
    }

    fn run(self) {
        let (dice_table, all_dice_counts) = self.all_dice.to_compact_form();
        let dice_left = self.dice_left.to_counts(&dice_table);
        let mut ctxt = Context::new(dice_table, all_dice_counts);

        let qstate = NormalizedStateAction::new(self.round_total, self.target_score, dice_left);

        let start_time = Instant::now();
        let score_pmf = qstate.score_distribution(&mut ctxt);
        let search_duration = start_time.elapsed();

        let mut dense_score_pmf = score_pmf.clone().into_vec();
        dense_score_pmf.sort_unstable_by(|(s1, _), (s2, _)| s1.cmp(s2));

        for (score, p_score) in dense_score_pmf {
            println!("{}\t{}", score, p_score);
        }

        let mut table = Table::new("{:>}  {:<}");
        table.add_row(row!("search duration", format!("{:.2?}", search_duration)));
        table.add_row(row!(
            "pmf expected value",
            format!("{:.3}", score_pmf.expected_value())
        ));
        table.add_row(row!(
            "pmf total mass",
            format!("{}", score_pmf.total_mass())
        ));
        table.add_row(row!(
            "actions explored",
            ctxt.actions_explored().to_string()
        ));
        table.add_row(row!(
            "game finished rate",
            format!("{:0.3}", ctxt.game_finished_rate())
        ));
        table.add_heading("");
        table.add_row(row!(
            "score distr cache size",
            format!(
                "{} ({})",
                ctxt.score_distr_cache().cache_size(),
                ctxt.score_distr_cache().cache_size_bytes()
            )
        ));
        table.add_row(row!(
            "score distr cache hit rate",
            format!(
                "{:0.3} (h: {}, m: {})",
                ctxt.score_distr_cache().cache_hit_rate(),
                ctxt.score_distr_cache().cache_hits(),
                ctxt.score_distr_cache().cache_misses()
            )
        ));
        table.add_heading("");
        table.add_row(row!(
            "actions cache size",
            format!(
                "{} ({})",
                ctxt.actions_cache().cache_size(),
                ctxt.actions_cache().cache_size_bytes()
            ),
        ));
        table.add_row(row!(
            "actions cache hit rate",
            format!(
                "{:0.3} (h: {}, m: {})",
                ctxt.actions_cache().cache_hit_rate(),
                ctxt.actions_cache().cache_hits(),
                ctxt.actions_cache().cache_misses()
            )
        ));
        eprint!("\n{}", table);
    }
}

/////////////////////////
// MarkovMatrixCommand //
/////////////////////////

#[derive(Debug)]
pub struct MarkovMatrixCommand {
    target_score: u16,
    all_dice: parse::DiceSet,
}

impl MarkovMatrixCommand {
    fn try_from_str_args(target_score: &str, all_dice: Option<&str>) -> Result<Self, String> {
        let cmd = Self {
            target_score: parse_req(target_score)?,
            all_dice: parse_opt(all_dice)?.unwrap_or_else(|| parse::DiceSet::all_standard(6)),
        };

        cmd.all_dice.validate_init_set(&cmd.all_dice)?;

        Ok(cmd)
    }
}

impl Command for MarkovMatrixCommand {
    // TODO(philiphayes): fill out
    const USAGE: &'static str = "";

    fn try_from_cli_args(mut args: Args) -> Result<Self, String> {
        args.maybe_help(Self::USAGE);

        let all_dice = args.opt_value(["-k", "--die-kinds"])?;
        let target_score = args.free_value()?;
        args.expect_finished()?;

        Self::try_from_str_args(&target_score, all_dice.as_deref())
    }

    fn run(self) {
        let start_time = Instant::now();
        let (die_table, die_counts) = self.all_dice.to_compact_form();
        let matrix = MarkovMatrix::from_optimal_policy(die_table, die_counts, self.target_score);
        let search_duration = start_time.elapsed();

        println!("{:?}", matrix);
        println!("\nsearch duration  {:.2?}", search_duration);
    }
}

/////////////////////
// TurnsCdfCommand //
/////////////////////

#[derive(Debug)]
pub struct TurnsCdfCommand {
    target_score: u16,
    max_num_turns: usize,
    our_dice: parse::DiceSet,
    their_dice: parse::DiceSet,
}

impl TurnsCdfCommand {
    fn try_from_str_args(
        target_score: &str,
        max_num_turns: &str,
        our_dice: Option<&str>,
        their_dice: Option<&str>,
    ) -> Result<Self, String> {
        let cmd = Self {
            target_score: parse_req(target_score)?,
            max_num_turns: parse_req(max_num_turns)?,
            our_dice: parse_opt(our_dice)?.unwrap_or_else(|| parse::DiceSet::all_standard(6)),
            their_dice: parse_opt(their_dice)?.unwrap_or_else(|| parse::DiceSet::all_standard(6)),
        };

        cmd.our_dice.validate_init_set(&cmd.our_dice)?;
        cmd.their_dice.validate_init_set(&cmd.their_dice)?;

        Ok(cmd)
    }
}

impl Command for TurnsCdfCommand {
    // TODO(philiphayes): fill out
    const USAGE: &'static str = "";

    fn try_from_cli_args(mut args: Args) -> Result<Self, String> {
        args.maybe_help(Self::USAGE);

        let our_dice = args.opt_value(["-o", "--our-die-kinds"])?;
        let their_dice = args.opt_value(["-t", "--their-die-kinds"])?;
        let target_score = args.free_value()?;
        let max_num_turns = args.free_value()?;
        args.expect_finished()?;

        Self::try_from_str_args(
            &target_score,
            &max_num_turns,
            our_dice.as_deref(),
            their_dice.as_deref(),
        )
    }

    fn run(self) {
        let start_time = Instant::now();

        let (our_die_table, our_die_counts) = self.our_dice.to_compact_form();
        let our_matrix =
            MarkovMatrix::from_optimal_policy(our_die_table, our_die_counts, self.target_score);
        let our_turns_cdf = our_matrix.turns_to_win_cdf(self.max_num_turns);

        let their_turns_cdf = if self.our_dice == self.their_dice {
            our_turns_cdf.clone()
        } else {
            let (their_die_table, their_die_counts) = self.their_dice.to_compact_form();
            let their_matrix = MarkovMatrix::from_optimal_policy(
                their_die_table,
                their_die_counts,
                self.target_score,
            );
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
        let p_win = p_rv1_lte_rv2(our_turns_cdf.view(), their_turns_cdf.view());

        let mut table = Table::new("{:>}  {:<}");
        table.add_row(row!("search_duration", format!("{:.2?}", search_duration)));
        table.add_row(row!("Pr[win]", format!("{}", p_win)));
        eprintln!("\n{}", table);
    }
}

/////////////////
// BaseCommand //
/////////////////

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

    fn try_from_cli_args(mut args: Args) -> Result<Self, String> {
        let maybe_subcommand = args.subcommand()?;

        match maybe_subcommand.as_deref() {
            Some("best-action") => Ok(Self::BestAction(BestActionCommand::try_from_cli_args(
                args,
            )?)),
            Some("score-distr") => Ok(Self::ScoreDistr(ScoreDistrCommand::try_from_cli_args(
                args,
            )?)),
            Some("markov-matrix") => Ok(Self::MarkovMatrix(
                MarkovMatrixCommand::try_from_cli_args(args)?,
            )),
            Some("turns-cdf") => Ok(Self::TurnsCdf(TurnsCdfCommand::try_from_cli_args(args)?)),
            Some(command) => Err(format!("'{}' is not a recognized command", command)),
            None => {
                args.maybe_help(Self::USAGE);
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
