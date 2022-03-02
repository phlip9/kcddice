use gloo_worker::PrivateWorker;
use kcddice_www::cmd_worker::CmdWorker;
use log::trace;

fn main() {
    console_error_panic_hook::set_once();
    console_log::init_with_level(log::Level::Trace).expect("Failed to init logger");

    trace!("cmd_worker: init");

    CmdWorker::register();

    trace!("cmd_worker: registered");
}
