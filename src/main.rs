// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

mod demo;
mod lm;
mod utils;

use anyhow::Result;

#[derive(Debug, clap::Parser)]
struct DemoArgs {
    #[arg(long)]
    audio: String,

    #[arg(long, default_value_t = 8000)]
    port: u16,

    #[clap(short = 'a', long = "addr", default_value = "0.0.0.0")]
    addr: String,
}

#[derive(Debug, clap::Parser)]
struct LmArgs {
    #[arg(long)]
    config: String,

    #[arg(long, default_value_t = 8000)]
    port: u16,

    #[clap(short = 'a', long = "addr", default_value = "0.0.0.0")]
    addr: String,

    #[arg(long)]
    cpu: bool,
}

#[derive(Debug, clap::Subcommand)]
enum Command {
    Demo(DemoArgs),
    Lm(LmArgs),
}

#[derive(clap::Parser, Debug)]
#[clap(name = "server", about = "kyutai backend")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = <Args as clap::Parser>::parse();
    tracing_subscriber::fmt()
        .event_format(tracing_subscriber::fmt::format().with_file(true).with_line_number(true))
        .init();

    match args.command {
        Command::Demo(args) => demo::run(args).await?,
        Command::Lm(args) => lm::run(args).await?,
    };
    Ok(())
}
