use std::time::Duration;

use criterion::Criterion;
use somni::{
    codegen,
    error::CompileError,
    ir,
    transform_ir::transform_ir,
    vm::{EvalContext, EvalEvent},
};
use somni_expr::{Context, ExprContext};
use somni_parser::parser;

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
    let ast = match parser::parse(SOURCE_CODE) {
        Ok(ast) => ast,
        Err(e) => {
            println!("Error parsing `{SOURCE_CODE}`");
            println!("{:?}", CompileError::new(SOURCE_CODE, e));
            panic!("Parsing failed");
        }
    };
    let mut ir = match ir::Program::compile(&SOURCE_CODE, &ast) {
        Ok(program) => program,
        Err(e) => {
            println!("Error compiling `{SOURCE_CODE}`");
            println!("{:?}", e);
            panic!("Compilation failed");
        }
    };
    if let Err(e) = transform_ir(&SOURCE_CODE, &mut ir) {
        println!("Error transforming IR for `{SOURCE_CODE}`");
        println!("{:?}", e);
        panic!("Transformation failed");
    }
    let mut strings = ir.strings.clone().finalize();
    let program = match codegen::compile(&SOURCE_CODE, &ir) {
        Ok(program) => program,
        Err(e) => {
            println!("Error compiling `{SOURCE_CODE}`");
            println!("{:?}", e);
            panic!("Compilation failed");
        }
    };

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
