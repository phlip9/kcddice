use kcddice::cli::{Args, BaseCommand, Command};

fn main() {
    let args = pico_args::Arguments::from_env();
    let args = Args::new(args);

    match BaseCommand::try_from_cli_args(args) {
        Ok(cmd) => cmd.run(),
        Err(err) => {
            eprintln!("error: {}", err);
            eprintln!("Try 'kcddice --help' for more information.");
            std::process::exit(1);
        }
    }
}
