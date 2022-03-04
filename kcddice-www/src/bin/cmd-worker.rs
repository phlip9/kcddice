use gloo_worker::PrivateWorker;
use kcddice_www::cmd_worker::CmdWorker;
use log::trace;

fn main() {
    console_error_panic_hook::set_once();

    // Note: console.log's inside the web worker seem to print twice (on firefox
    // at least).
    console_log::init_with_level(log::Level::Info).expect("Failed to init logger");

    trace!("cmd-worker: init");

    CmdWorker::register();

    trace!("cmd-worker: registered");
}
