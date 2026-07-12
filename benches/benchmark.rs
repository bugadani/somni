use std::time::Duration;

use criterion::Criterion;
use somni::{
    Compiler,
    vm::{EvalContext, EvalEvent},
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

// Exercises the struct/reference machinery: struct (and nested struct) literal
// construction, field reads and writes, taking references to a struct and to a
// nested field, mutation through references, and struct equality.
const STRUCTS_SOURCE_CODE: &str = r#"
struct Point { x: int, y: int }
struct Line { start: Point, end: Point }

fn translate(p: &Point, dx: int, dy: int) {
    p.x = p.x + dx;
    p.y = p.y + dy;
}

fn dot(a: &Point, b: &Point) -> int {
    return a.x * b.x + a.y * b.y;
}

fn bump(v: &int) {
    *v = *v + 1;
}

fn main() -> int {
    var acc = 0;
    var i = 0;
    loop {
        if i >= 1000 {
            break;
        }

        var line = Line { start: Point { x: i, y: i }, end: Point { x: 1, y: 2 } };

        translate(&line.start, 1, 2);
        bump(&line.end.x);
        acc = acc + dot(&line.start, &line.end);

        var mirror = Point { x: line.start.x, y: line.start.y };
        if mirror == line.start {
            acc = acc + 1;
        }

        i = i + 1;
    }
    return acc;
}"#;

pub fn eval(c: &mut Criterion) {
    eval_fib(c);
    eval_structs(c);
}

pub fn eval_fib(c: &mut Criterion) {
    let mut context = Context::parse(SOURCE_CODE).unwrap();
    c.bench_function("fib 20 (expr)", |b| {
        b.iter(|| {
            context
                .call_function("main", &[])
                .unwrap_or_else(|_| panic!("Failed to evaluate benchmark"))
        })
    });
}

pub fn eval_structs(c: &mut Criterion) {
    let mut context = Context::parse(STRUCTS_SOURCE_CODE).unwrap();
    c.bench_function("structs 1000 (expr)", |b| {
        b.iter(|| {
            context
                .call_function("main", &[])
                .unwrap_or_else(|_| panic!("Failed to evaluate benchmark"))
        })
    });
}

pub fn vm(c: &mut Criterion) {
    vm_fib(c);
    vm_structs(c);
}

pub fn vm_fib(c: &mut Criterion) {
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

pub fn vm_structs(c: &mut Criterion) {
    let program = match Compiler::new().compile(&STRUCTS_SOURCE_CODE) {
        Ok(program) => program,
        Err(e) => {
            println!("Error compiling `{STRUCTS_SOURCE_CODE}`");
            println!("{:?}", e);
            panic!("Compilation failed");
        }
    };

    let mut strings = program.debug_info.strings.clone();
    c.bench_function("structs 1000 (vm)", |b| {
        b.iter(|| {
            let mut context = EvalContext::new(STRUCTS_SOURCE_CODE, &mut strings, &program);

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
