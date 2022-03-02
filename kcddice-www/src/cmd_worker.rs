use futures_channel::oneshot;
use gloo_worker::{Bridged, Private, PrivateWorker, Worker, WorkerLink};
use kcddice::{
    cli::{BestActionCommand, Command},
    parse,
};
use log::{debug, trace, warn};
use std::{cell::RefCell, rc::Rc};

pub struct CmdWorker {
    link: WorkerLink<Self>,
}

impl CmdWorker {
    fn handle_cmd(cmd: BestActionCommand) -> Result<parse::Action, String> {
        let out = cmd.run()?;

        debug!("CmdWorker::handle_cmd: out: {out}");

        let best_action = out
            .action_values
            .0
            .first()
            .ok_or_else(|| "no actions returned!".to_owned())?
            .action;
        let dice_table = out.dice_table;

        let best_action = parse::Action::from_compact_form(&dice_table, best_action);

        debug!("CmdWorker::handle_cmd: best_action: {best_action}");

        Ok(best_action)
    }

    /// Used by the main UI thread to request this worker to run a command.
    /// Effectively converts this weird callback pattern into a normal Future.
    pub async fn request(cmd: BestActionCommand) -> Result<parse::Action, String> {
        // create a new oneshot channel which we'll wait on to get the result.
        let (tx, rx) = oneshot::channel();

        // callback can theoretically be called multiple times... we'll take the
        // tx out of the slot once, then panic if we ever get called again.
        let tx_slot = RefCell::new(Some(tx));

        // create a callback which just takes the output and fulfills the oneshot channel.
        let cb = Rc::new(move |out| {
            let tx = tx_slot
                .take()
                .expect("command worker request callback called multiple times somehow");
            if let Err(_err) = tx.send(out) {
                trace!("CmdWorker::request: request cancelled!");
            }
        });

        // create an ephemeral connection to the worker and send one message.
        let mut bridge = Self::bridge(cb);
        bridge.send(cmd);

        // wait for the result.
        let out_res_res = rx.await;

        match out_res_res {
            Ok(out_res) => out_res,
            Err(_) => Err("command worker appears to be dead or non-responsive".to_owned()),
        }
    }
}

impl Worker for CmdWorker {
    type Message = ();
    type Input = BestActionCommand;
    type Output = Result<parse::Action, String>;
    type Reach = Private<Self>;

    fn create(link: WorkerLink<Self>) -> Self {
        trace!("CmdWorker::create");
        Self { link }
    }

    fn connected(&mut self, id: gloo_worker::HandlerId) {
        trace!("CmdWorker::connected: id: {id:?}");
    }

    fn disconnected(&mut self, id: gloo_worker::HandlerId) {
        trace!("CmdWorker::disconnected: id: {id:?}");
    }

    fn destroy(&mut self) {
        trace!("CmdWorker::destroy");
    }

    fn update(&mut self, _msg: Self::Message) {
        trace!("CmdWorker::update");
    }

    fn handle_input(&mut self, cmd: Self::Input, id: gloo_worker::HandlerId) {
        trace!("CmdWorker::handle_input: cmd: {:?}, id: {:?}", cmd, id);

        let maybe_out = Self::handle_cmd(cmd);

        if let Err(err) = &maybe_out {
            warn!("CmdWorker::handle_input: error: {err}");
        }

        self.link.respond(id, maybe_out);
    }

    fn name_of_resource() -> &'static str {
        "cmd_worker.js"
    }
}
