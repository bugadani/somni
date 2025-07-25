use std::time::Duration;

use criterion::Criterion;
use somni::{
    codegen,
    error::CompileError,
    ir,
    transform_ir::transform_ir,
    vm::{EvalContext, EvalEvent},
};
use somni_parser::parser;

pub fn vm(c: &mut Criterion) {
    let source_code = r#"
fn fib(n: int) -> int {
    if n <= 1 {
        return n;
    }
    return fib(n - 1) + fib(n - 2);
}

fn main() {
    fib(20);
}"#;

    let ast = match parser::parse(source_code) {
        Ok(ast) => ast,
        Err(e) => {
            println!("Error parsing `{source_code}`");
            println!("{:?}", CompileError::new(source_code, e));
            panic!("Parsing failed");
        }
    };
    let mut ir = match ir::Program::compile(&source_code, &ast) {
        Ok(program) => program,
        Err(e) => {
            println!("Error compiling `{source_code}`");
            println!("{:?}", e);
            panic!("Compilation failed");
        }
    };
    if let Err(e) = transform_ir(&source_code, &mut ir) {
        println!("Error transforming IR for `{source_code}`");
        println!("{:?}", e);
        panic!("Transformation failed");
    }
    let mut strings = ir.strings.clone().finalize();
    let program = match codegen::compile(&source_code, &ir) {
        Ok(program) => program,
        Err(e) => {
            println!("Error compiling `{source_code}`");
            println!("{:?}", e);
            panic!("Compilation failed");
        }
    };

    c.bench_function("fib 20", |b| {
        b.iter(|| {
            let mut context = EvalContext::new(source_code, &mut strings, &program);

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

    vm(&mut criterion);
}
