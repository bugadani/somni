use std::time::Duration;

use criterion::Criterion;
use somni::{
    vm::{EvalContext, EvalEvent},
    Compiler,
};
use somni_expr::{Context, ExprContext};

const SOURCE_CODE: &str = r#"
fn fib(n: int) -> int {
    if n <= 1 {
        return n;
    }
    return fib(n - 1) + fib(n - 2);
}

fn main() {
    fib(20);
}"#;

pub fn eval(c: &mut Criterion) {
    let mut context = Context::parse(SOURCE_CODE).unwrap();

    c.bench_function("fib 20 (expr)", |b| {
        b.iter(|| {
            context
                .call_function("main", &[])
                .unwrap_or_else(|_| panic!("Failed to evaluate benchmark"))
        })
    });
}

pub fn vm(c: &mut Criterion) {
    let program = match Compiler::new().compile(&SOURCE_CODE) {
        Ok(program) => program,
        Err(e) => {
            println!("Error compiling `{SOURCE_CODE}`");
            println!("{:?}", e);
            panic!("Compilation failed");
        }
    };

    let mut strings = program.debug_info.strings.clone();
    c.bench_function("fib 20 (vm)", |b| {
        b.iter(|| {
            let mut context = EvalContext::new(SOURCE_CODE, &mut strings, &program);

            loop {
                match context.run() {
                    EvalEvent::UnknownFunctionCall(_) => {}
                    EvalEvent::Error(e) => {
                        context.print_backtrace();

                        panic!("{}", e.mark(&context, "Runtime error"));
                    }
                    EvalEvent::Complete(result) => break result,
                }
            }
        })
    });
}

fn main() {
    let mut criterion: criterion::Criterion<_> = (criterion::Criterion::default())
        .configure_from_args()
        .warm_up_time(Duration::from_secs(10))
        .measurement_time(Duration::from_secs(30));

    eval(&mut criterion);
    vm(&mut criterion);
}
