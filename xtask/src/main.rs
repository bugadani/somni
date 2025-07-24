fn main() {
    use clap::Parser;
    #[derive(Parser)]
    #[clap(name = "xtask", about = "A task runner for the project")]
    struct Cli {
        #[clap(subcommand)]
        command: Command,
    }

    #[derive(clap::Subcommand)]
    enum Command {
        /// Run tests
        Test {
            #[clap()]
            filter: Option<String>,

            /// Bless the test output
            #[clap(long)]
            bless: bool,
        },
    }
    let cli = Cli::parse();
    match cli.command {
        Command::Test { filter, bless } => {
            // invoke `cargo test`
            let mut envs = vec![];
            envs.extend(bless.then_some(("BLESS", "1")));
            envs.extend(filter.as_deref().map(|f| ("TEST_FILTER", f)));

            let mut args = vec!["test", "--all-features", "--all"];
            if filter.is_some() {
                args.push("--");
                args.push("--test-threads=1");
                args.push("--nocapture");
            }

            std::process::Command::new("cargo")
                .args(args)
                .envs(envs)
                .status()
                .expect("Failed to run cargo test");
        }
    }
}
