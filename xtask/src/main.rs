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
            /// Bless the test output
            #[clap(long)]
            bless: bool,
        },
    }
    let cli = Cli::parse();
    match cli.command {
        Command::Test { bless } => {
            // invoke `cargo test`
            std::process::Command::new("cargo")
                .arg("test")
                .arg("--all-features")
                .envs(bless.then(|| ("BLESS", "1")))
                .status()
                .expect("Failed to run cargo test");
        }
    }
}
