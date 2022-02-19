use kcddice::cli::{Args, BaseCommand, Command};

fn main() {
    let args = pico_args::Arguments::from_env();
    let args = Args::new(args);

    let cmd = match BaseCommand::try_from_cli_args(args) {
        Ok(cmd) => cmd,
        Err(err) => {
            eprintln!("error: {}", err);
            eprintln!("Try 'kcddice --help' for more information.");
            std::process::exit(1);
        }
    };

    match cmd.run() {
        Ok(out_str) => println!("{}", out_str),
        Err(err_str) => {
            eprintln!("error: {}", err_str);
            std::process::exit(1);
        }
    }
}
