//! Command-line parsing is an adapter to the shared processing engine.
mod args;
use clap::Parser;
pub(crate) fn run() -> Result<(), Box<dyn std::error::Error>> {
    let options = args::Args::parse().into();
    smude_geometry::application::run(&options).map(|_| ())
}
