use kcddice::cli::{BaseCommand, Command};

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
