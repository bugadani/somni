use std::collections::HashMap;

use crate::{
    codegen::{self, CodeAddress, Function, Instruction, MemoryAddress, Value},
    error::MarkInSource,
    lexer::Location,
    string_interner::{StringIndex, Strings},
};

#[derive(Clone, Debug)]
pub struct EvalError(Box<str>);

impl EvalError {
    pub fn mark<'a>(&'a self, context: &'a EvalContext, message: &'a str) -> MarkInSource<'a> {
        MarkInSource(context.source, context.current_location(), message, &self.0)
    }
}

#[derive(Clone, Debug)]
pub enum EvalEvent {
    UnknownFunctionCall,
    Error(EvalError),
    Complete(Value),
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum EvalState {
    Idle,
    Running,
    WaitingForFunctionResult(StringIndex, MemoryAddress, usize),
}

struct Frame {
    program_counter: CodeAddress,
    name: StringIndex,
    frame_pointer: usize, // Upon call, the stack pointer is saved here.
}

impl Frame {
    fn step(&mut self) {
        self.program_counter += 1;
    }

    fn pc(&self) -> usize {
        self.program_counter.0
    }
}

impl Function {
    fn stack_frame(&self) -> Frame {
        Frame {
            program_counter: self.entry_point,
            name: self.name,
            frame_pointer: 0, // Will be used to save callee stack pointer
        }
    }
}

type IntrinsicFn<'p> = fn(&EvalContext<'p>, &[Value]) -> Result<Value, EvalEvent>;

pub struct EvalContext<'p> {
    source: &'p str,
    strings: &'p Strings,
    program: &'p codegen::Program,
    intrinsics: HashMap<StringIndex, IntrinsicFn<'p>>,
    state: EvalState,

    memory: Memory,

    current_frame: Frame,
    call_stack: Vec<Frame>,
}

/// Virtual memory.
///
/// Memory layout:
/// - Global variables at the bottom, starting at index 0.
/// - Local variables for the current function, starting at the stack pointer (sp).
/// - Temporary values on the stack, above the local variables.
///
/// Function arguments are temporaries in the caller frame, and locals in the callee frame.
/// Calling a function will allocate space for the function's local variables, and adjust the stack
/// pointer accordingly. The old stack pointer is saved before reclassifying the arguments, so
/// restoring it on return will clear the function arguments.
struct Memory {
    data: Vec<Value>,
    sp: usize,
}

impl Memory {
    fn new() -> Self {
        Memory {
            data: Vec::new(),
            sp: 0,
        }
    }

    fn allocate(&mut self, size: usize) {
        self.data
            .resize(self.data.len().max(self.sp + size), Value::Void);
    }

    fn truncate(&mut self, frame_pointer: usize) {
        if frame_pointer > self.sp {
            panic!(
                "Trying to truncate memory to a frame pointer that is larger than the current stack pointer: {frame_pointer} > {}",
                self.sp
            );
        }
        let sp = self.sp;
        self.sp = frame_pointer;
        self.data.truncate(sp);
    }

    fn address(&self, var_id: MemoryAddress) -> usize {
        match var_id {
            MemoryAddress::Global(address) => address,
            MemoryAddress::Local(address) => self.sp + address,
        }
    }

    fn store(&mut self, var_id: MemoryAddress, value: Value) -> Result<(), String> {
        let address = self.address(var_id);

        let Some(variable) = self.data.get_mut(address) else {
            return Err(format!(
                "Trying to store value at address {address} which is out of bounds"
            ));
        };

        *variable = value;
        Ok(())
    }

    fn load(&self, var_id: MemoryAddress) -> Result<Value, String> {
        let address = self.address(var_id);

        let Some(variable) = self.data.get(address) else {
            return Err(format!(
                "Trying to load value from address {address} which is out of bounds"
            ));
        };

        Ok(*variable)
    }

    fn peek_from(&self, sp: MemoryAddress, count: usize) -> Result<&[Value], String> {
        let address = self.address(sp);
        if address >= self.data.len() {
            return Err(format!(
                "Trying to peek from address {address} which is out of bounds"
            ));
        }
        Ok(&self.data[address..][..count])
    }
}

macro_rules! arithmetic_operator {
    ($this:ident, $dst:ident, $lhs:ident, $rhs:ident, $op:tt, $check_error:literal, $type_error:literal $(, $float_op:tt)? ) => {{
        let lhs = match $this.load($lhs) {
            Ok(value) => value,
            Err(e) => return e,
        };
        let rhs = match $this.load($rhs) {
            Ok(value) => value,
            Err(e) => return e,
        };

        let result = match (lhs, rhs) {
            (Value::Int(a), Value::Int(b)) => {
                #[allow(irrefutable_let_patterns)]
                let Ok(b) = b.try_into() else {
                    return $this.runtime_error(format_args!($check_error, a, b));
                };
                if let Some(result) = <u64>::$op(a, b) {
                    Value::Int(result)
                } else {
                    return $this.runtime_error(format_args!($check_error, a, b));
                }
            }
            (Value::SignedInt(a), Value::SignedInt(b)) => {
                #[allow(irrefutable_let_patterns)]
                let Ok(b) = b.try_into() else {
                    return $this.runtime_error(format_args!($check_error, a, b));
                };
                if let Some(result) = <i64>::$op(a, b) {
                    Value::SignedInt(result)
                } else {
                    return $this.runtime_error(format_args!($check_error, a, b));
                }
            }
            $((Value::Float(a), Value::Float(b)) => Value::Float(a $float_op b),)?
            (a, b) => {
                return $this.runtime_error(format_args!($type_error, a.type_of(), b.type_of()));
            }
        };

        if let Err(e) = $this.store($dst, result) {
            return e;
        }
    }};
}

macro_rules! comparison_operator {
    ($this:ident, $dst:ident, $lhs:ident, $rhs:ident, $op:tt) => {{
        let lhs = match $this.load($lhs) {
            Ok(value) => value,
            Err(e) => return e,
        };
        let rhs = match $this.load($rhs) {
            Ok(value) => value,
            Err(e) => return e,
        };

        let result = match (lhs, rhs) {
            (Value::Int(a), Value::Int(b)) => Value::Bool(a $op b),
            (Value::SignedInt(a), Value::SignedInt(b)) => Value::Bool(a $op b),
            (Value::Float(a), Value::Float(b)) => Value::Bool(a $op b),
            (Value::Bool(a), Value::Bool(b)) => Value::Bool(a $op b),
            (Value::String(a), Value::String(b)) if stringify!($op) == "==" => Value::Bool(a == b),
            (a, b) => {
                return $this.runtime_error(format_args!("Cannot compare {} and {}", a.type_of(), b.type_of()));
            }
        };

        if let Err(e) = $this.store($dst, result) {
            return e;
        }
    }};
}

macro_rules! bitwise_operator {
    ($this:ident, $dst:ident, $lhs:ident, $rhs:ident, $op:tt) => {{
        let lhs = match $this.load($lhs) {
            Ok(value) => value,
            Err(e) => return e,
        };
        let rhs = match $this.load($rhs) {
            Ok(value) => value,
            Err(e) => return e,
        };

        let result = match (lhs, rhs) {
            (Value::Int(a), Value::Int(b)) => Value::Int(a $op b),
            (Value::SignedInt(a), Value::SignedInt(b)) => Value::SignedInt(a $op b),
            (Value::Bool(a), Value::Bool(b)) => Value::Bool(a $op b),
            (a, b) => {
                return $this.runtime_error(
                    format_args!("Cannot calculate {} {} {}", a.type_of(), stringify!($op), b.type_of()),
                );
            }
        };
        if let Err(e) = $this.store($dst, result) {
            return e;
        }
    }};
}

impl<'p> EvalContext<'p> {
    fn string(&self, index: StringIndex) -> &str {
        self.strings.lookup(index)
    }

    fn load_function_by_name(&self, name: &str) -> Option<&'p codegen::Function> {
        let name = self.strings.find(name)?;
        self.program.functions.get(&name)
    }

    pub fn new(source: &'p str, strings: &'p Strings, program: &'p codegen::Program) -> Self {
        let mut this = EvalContext {
            intrinsics: HashMap::new(),
            state: EvalState::Idle,
            program,
            memory: Memory::new(),
            call_stack: vec![],
            source,
            strings,
            current_frame: Frame {
                program_counter: CodeAddress(0),
                name: StringIndex::dummy(), // Will be set when the first function is called
                frame_pointer: 0,           // Will be set when the first function is called
            },
        };

        this.add_intrinsic("print", |program, args| {
            for value in args.iter() {
                match value {
                    Value::String(s) => print!("{}", program.string(*s)),
                    Value::Int(v) => print!("{v}"),
                    Value::SignedInt(v) => print!("{v}"),
                    Value::Bool(v) => print!("{v}"),
                    Value::Void => print!("(void)"),
                    Value::Float(v) => print!("{v}"),
                    Value::Address(..) => print!("Cannot print reference"),
                }
            }
            Ok(Value::Void)
        });

        this.memory.allocate(program.globals.len());
        for (address, (_, def)) in program.globals.iter().enumerate() {
            this.memory
                .store(MemoryAddress::Global(address), *def.value())
                .unwrap();
        }

        this
    }

    /// Calls the `main` function and starts the evaluation. If the program is already running,
    /// it will continue executing the current function.
    ///
    /// If the function returns with [`EvalEvent::UnknownFunctionCall`], it means that the script
    /// tried to call a function that is not defined in the program. You can use
    /// [`Self::unknown_function_info()`] to get the name and arguments of the function that
    /// was called. Set the return value with [`Self::set_return_value()`], then call [`Self::run`]
    /// to continue execution.
    pub fn run(&mut self) -> EvalEvent {
        if matches!(self.state, EvalState::Idle) {
            // Initialize the first frame with the main program
            self.call("main", &[])
        } else {
            self.execute()
        }
    }

    /// Calls a function by its name with the given arguments.
    ///
    /// Note that this resets the VM before calling the function, so it will not
    /// preserve the current state of the VM. If you want to continue executing the current
    /// function, use [`Self::run`] instead.
    ///
    /// If the function returns with [`EvalEvent::UnknownFunctionCall`], it means that the script
    /// tried to call a function that is not defined in the program. You can use
    /// [`Self::unknown_function_info()`] to get the name and arguments of the function that
    /// was called. Set the return value with [`Self::set_return_value()`], then call [`Self::run`]
    /// to continue execution.
    pub fn call(&mut self, func: &str, args: &[Value]) -> EvalEvent {
        let Some(function) = self.load_function_by_name(func) else {
            return self.runtime_error(format!("Unknown function: {func}"));
        };

        self.current_frame = function.stack_frame();

        self.memory.sp = self.program.globals.len();
        self.memory.allocate(function.stack_size);

        // Store the function arguments as temporaries in the caller's stack frame.
        for (i, arg) in args.iter().enumerate() {
            if let Err(e) = self.store(MemoryAddress::Local(i + 1), *arg) {
                return e;
            }
        }

        self.state = EvalState::Running;

        self.execute()
    }

    fn execute(&mut self) -> EvalEvent {
        if let EvalState::WaitingForFunctionResult(name, ..) = self.state {
            return self.runtime_error(format!(
                "Function '{}' is still waiting for a result",
                self.string(name)
            ));
        }

        loop {
            let instruction = self.current_instruction();

            // println!(
            //     "{}",
            //     crate::error::MarkInSource(
            //         self.source,
            //         self.current_location(),
            //         &format!("Executing instruction: {instruction:?}"),
            //         "",
            //     )
            // );

            match instruction {
                Instruction::Call(function_index, sp) => {
                    let (_, function) = self.program.functions.get_index(function_index).unwrap();
                    if let Err(e) = self.call_function(function, sp) {
                        return e;
                    }
                    continue; // Skip the step() call below, as we already stepped (into function)
                }
                Instruction::CallNamed(function_name, sp, args) => {
                    let Some(intrinsic) = self.intrinsics.get(&function_name) else {
                        self.state = EvalState::WaitingForFunctionResult(function_name, sp, args);
                        self.current_frame.step();
                        return EvalEvent::UnknownFunctionCall;
                    };

                    if let Err(e) = self
                        .memory
                        .peek_from(sp + 1, args)
                        .map_err(|e| {
                            self.runtime_error(format_args!("Failed to read arguments: {e}"))
                        })
                        .and_then(|args| intrinsic(self, args))
                        .and_then(|result| self.store(sp, result))
                    {
                        return e;
                    };
                }
                Instruction::Return => {
                    let Some(frame) = self.call_stack.pop() else {
                        // Outermost function has returned
                        let retval = self.memory.load(MemoryAddress::Local(0)).unwrap();
                        self.memory.truncate(self.program.globals.len());
                        self.state = EvalState::Running;
                        return EvalEvent::Complete(retval);
                    };

                    self.current_frame = frame;
                    self.memory.sp = self.current_frame.frame_pointer;
                }
                Instruction::Jump(n) => {
                    self.current_frame.program_counter = n;
                    continue; // Skip the step() call below, as we already stepped forward
                }
                Instruction::JumpIfFalse(condition, target) => {
                    let value = match self.load(condition) {
                        Ok(value) => value,
                        Err(e) => return e,
                    };

                    match value {
                        Value::Bool(false) => {
                            self.current_frame.program_counter = target;
                            continue; // Skip the step() call below, as we already stepped forward
                        }
                        Value::Bool(true) => {}
                        v => {
                            return self.runtime_error(format_args!(
                                "Type mismatch: expected bool, found {}",
                                v.type_of()
                            ));
                        }
                    }
                }
                Instruction::TestLessThan(dst, lhs, rhs) => {
                    comparison_operator!(self, dst, lhs, rhs, <)
                }
                Instruction::TestLessThanOrEqual(dst, lhs, rhs) => {
                    comparison_operator!(self, dst, lhs, rhs, <=)
                }
                Instruction::TestEquals(dst, lhs, rhs) => {
                    comparison_operator!(self, dst, lhs, rhs, ==)
                }
                Instruction::TestNotEquals(dst, lhs, rhs) => {
                    comparison_operator!(self, dst, lhs, rhs, !=)
                }
                Instruction::BitwiseOr(dst, lhs, rhs) => bitwise_operator!(self, dst, lhs, rhs, |),
                Instruction::BitwiseXor(dst, lhs, rhs) => bitwise_operator!(self, dst, lhs, rhs, ^),
                Instruction::BitwiseAnd(dst, lhs, rhs) => bitwise_operator!(self, dst, lhs, rhs, &),
                Instruction::ShiftLeft(dst, lhs, rhs) => {
                    arithmetic_operator!(
                        self,
                        dst,
                        lhs,
                        rhs,
                        checked_shl,
                        "{} << {} overflowed",
                        "Cannot shift {} by {}"
                    );
                }
                Instruction::ShiftRight(dst, lhs, rhs) => {
                    arithmetic_operator!(
                        self,
                        dst,
                        lhs,
                        rhs,
                        checked_shr,
                        "{} >> {} underflowed",
                        "Cannot shift {} by {}"
                    );
                }
                Instruction::Add(dst, lhs, rhs) => {
                    arithmetic_operator!(
                        self,
                        dst,
                        lhs,
                        rhs,
                        checked_add,
                        "{} + {} overflowed",
                        "Cannot add {} and {}",
                        +
                    );
                }
                Instruction::Subtract(dst, lhs, rhs) => {
                    arithmetic_operator!(
                        self,
                        dst,
                        lhs,
                        rhs,
                        checked_sub,
                        "{} - {} underflowed",
                        "Cannot subtract {} and {}",
                        -
                    );
                }
                Instruction::Multiply(dst, lhs, rhs) => {
                    arithmetic_operator!(
                        self,
                        dst,
                        lhs,
                        rhs,
                        checked_mul,
                        "{} * {} overflowed",
                        "Cannot multiply {} and {}",
                        *
                    );
                }
                Instruction::Divide(dst, lhs, rhs) => {
                    arithmetic_operator!(
                        self,
                        dst,
                        lhs,
                        rhs,
                        checked_div,
                        "Division by zero: {} / {}",
                        "Cannot divide {} and {}",
                        /
                    );
                }
                Instruction::Negate(dst, lhs) => {
                    let lhs = match self.load(lhs) {
                        Ok(value) => value,
                        Err(e) => return e,
                    };
                    let result = match lhs {
                        Value::SignedInt(a) => Value::SignedInt(-a),
                        Value::Float(a) => Value::Float(-a),

                        a => {
                            return self
                                .runtime_error(format_args!("Cannot negate {}", a.type_of()));
                        }
                    };
                    if let Err(e) = self.store(dst, result) {
                        return e;
                    }
                }
                Instruction::Not(dst, lhs) => {
                    let lhs = match self.load(lhs) {
                        Ok(value) => value,
                        Err(e) => return e,
                    };
                    let result = match lhs {
                        Value::Bool(a) => Value::Bool(!a),
                        Value::Int(a) => Value::Int(!a),
                        a => {
                            return self
                                .runtime_error(format_args!("Cannot invert {}", a.type_of()));
                        }
                    };
                    if let Err(e) = self.store(dst, result) {
                        return e;
                    }
                }
                Instruction::Address(dst, lhs) => {
                    let value = Value::Address(MemoryAddress::Global(self.memory.address(lhs)));
                    if let Err(e) = self.store(dst, value) {
                        return e;
                    }
                }
                Instruction::Dereference(dst, lhs) => {
                    let address = match self.load(lhs) {
                        Ok(addr) => addr,
                        Err(e) => return e,
                    };
                    let address = match address {
                        Value::Address(addr) => addr,
                        v => {
                            return self.runtime_error(format_args!(
                                "Expected address, found {}",
                                v.type_of()
                            ));
                        }
                    };

                    if let Err(e) = self.copy(address, dst) {
                        return e;
                    }
                }
                Instruction::Copy(dst, from) => {
                    if let Err(e) = self.copy(from, dst) {
                        return e;
                    }
                }
                Instruction::DerefCopy(addr, from) => {
                    let dst = match self.memory.load(addr) {
                        Ok(Value::Address(addr)) => addr,
                        Ok(v) => {
                            return self.runtime_error(format_args!(
                                "Expected address, found {}",
                                v.type_of()
                            ));
                        }
                        Err(e) => {
                            return self.runtime_error(format_args!("Failed to load address: {e}"));
                        }
                    };

                    if let Err(e) = self.copy(from, dst) {
                        return e;
                    }
                }
                Instruction::LoadValue(addr, value) => {
                    if let Err(e) = self.store(addr, value) {
                        return e;
                    }
                }
            }

            self.current_frame.step();
        }
    }

    fn store(&mut self, addr: MemoryAddress, value: Value) -> Result<(), EvalEvent> {
        self.memory
            .store(addr, value)
            .map_err(|e| self.runtime_error(format_args!("Failed to store value at address: {e}")))
    }

    fn load(&mut self, addr: MemoryAddress) -> Result<Value, EvalEvent> {
        self.memory
            .load(addr)
            .map_err(|e| self.runtime_error(format_args!("Failed to load value from address: {e}")))
    }

    fn copy(&mut self, addr: MemoryAddress, dst: MemoryAddress) -> Result<(), EvalEvent> {
        self.load(addr).and_then(|value| self.store(dst, value))
    }

    pub fn unknown_function_info(&self) -> Option<(&str, &[Value])> {
        if let EvalState::WaitingForFunctionResult(name, sp, args) = self.state {
            let function_name = self.string(name);
            let args = self.memory.peek_from(sp + 1, args).unwrap_or_default();
            Some((function_name, args))
        } else {
            None
        }
    }

    pub fn set_return_value(&mut self, value: Value) -> Result<(), EvalEvent> {
        let EvalState::WaitingForFunctionResult(_, sp, _) = self.state else {
            return Err(self.runtime_error("No function is currently waiting for a result"));
        };

        self.memory
            .store(sp, value)
            .map_err(|e| self.runtime_error(format_args!("Failed to store return value: {e}")))?;
        self.state = EvalState::Running;
        Ok(())
    }

    fn add_intrinsic(
        &mut self,
        name: &'static str,
        fun: fn(&Self, &[Value]) -> Result<Value, EvalEvent>,
    ) {
        // If the name is not interned yet, the intrinsic is not used and not needed.
        if let Some(name_index) = self.strings.find(name) {
            self.intrinsics.insert(name_index, fun);
        }
    }

    pub fn print_backtrace(&self) {
        let mut i = 1;
        println!("Frame {i}: {}", self.string(self.current_frame.name));
        for frame in self.call_stack.iter().rev() {
            i += 1;
            println!("Frame {i}: {}", self.string(frame.name));
        }
    }

    fn call_function(
        &mut self,
        function: &'p Function,
        sp: MemoryAddress,
    ) -> Result<(), EvalEvent> {
        self.current_frame.frame_pointer = self.memory.sp;
        self.memory.sp = self.memory.address(sp);
        // Allocate memory for the function's local variables.
        self.memory.allocate(function.stack_size);

        let old_frame = std::mem::replace(&mut self.current_frame, function.stack_frame());
        self.call_stack.push(old_frame);

        Ok(())
    }

    #[cold]
    fn runtime_error(&self, message: impl ToString) -> EvalEvent {
        EvalEvent::Error(EvalError(message.to_string().into_boxed_str()))
    }

    fn current_instruction(&self) -> Instruction {
        self.program.code[self.current_frame.pc()]
    }

    fn current_location(&self) -> Location {
        let pc = self.current_frame.program_counter;
        self.program.debug_info.instruction_locations[pc.0]
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn test_vm() {
        crate::test::run_eval_tests("tests/eval/");
    }
}
