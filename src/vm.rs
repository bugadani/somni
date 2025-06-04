use std::{collections::HashMap, mem::MaybeUninit};

use crate::{
    compiler::{
        Function, FunctionSignature, Instruction, Program, StringIndex, Type, Value, VarId,
    },
    error::MarkInSource,
    lexer::Location,
};

#[derive(Clone, Debug)]
pub struct EvalError(Box<str>);

impl EvalError {
    pub fn mark<'a>(&'a self, context: &'a EvalContext, message: &'a str) -> MarkInSource<'a> {
        let pc = context.current_frame.program_counter;
        let location = context.program.debug_info.instruction_locations[pc];
        let location = context.program.location_at(location);
        MarkInSource(&context.program.source, location, message, &self.0)
    }
}

#[derive(Clone, Debug)]
pub enum EvalEvent {
    UnknownFunctionCall,
    Error(EvalError),
    Complete(Value),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EvalState {
    Idle,
    Running,
    WaitingForFunctionResult(StringIndex, usize),
}

struct Frame {
    program_counter: usize,
    name: StringIndex,
    return_type: Type,
    // TODO: loops should be removed, codegen should generate jumps instead of loops.
    loops: Vec<(usize, usize)>, // (start, body length) of loops
    frame_pointer: usize,       // Upon call, the stack pointer is saved here.
}

impl Frame {
    fn step(&mut self) {
        self.step_forward(1);
    }

    fn step_forward(&mut self, n: usize) {
        self.program_counter += n;
    }

    fn step_back(&mut self, n: usize) -> bool {
        let Some(pc) = self.program_counter.checked_sub(n) else {
            return false;
        };

        self.program_counter = pc;
        true
    }
}

pub struct EvalContext<'p> {
    program: &'p Program,
    intrinsics: HashMap<StringIndex, fn(&Self, &[Value]) -> Result<Value, EvalEvent>>,
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
        self.sp = self.data.len();
        self.data.resize(self.sp + size, Value::Void);
    }

    fn truncate(&mut self, frame_pointer: usize) {
        if frame_pointer > self.sp {
            panic!(
                "Trying to truncate memory to a frame pointer that is larger than the current stack pointer: {} > {}",
                frame_pointer, self.sp
            );
        }
        let sp = self.sp;
        self.sp = frame_pointer;
        self.data.truncate(sp);
    }

    fn address(&self, var_id: VarId) -> usize {
        match var_id {
            VarId::Global(address, _) => address.index(),
            VarId::Local(address, _) => self.sp + address.index(),
        }
    }

    fn store(&mut self, var_id: VarId, value: Value) -> Result<(), String> {
        let address = self.address(var_id);

        let Some(variable) = self.data.get_mut(address) else {
            return Err(format!(
                "Trying to store value at address {} which is out of bounds",
                address
            ));
        };

        *variable = value;
        Ok(())
    }

    fn load(&self, var_id: VarId) -> Result<Value, String> {
        let address = self.address(var_id);

        let Some(variable) = self.data.get(address) else {
            return Err(format!(
                "Trying to load value from address {} which is out of bounds",
                address
            ));
        };

        Ok(*variable)
    }

    fn push(&mut self, value: Value) {
        self.data.push(value);
    }

    fn pop(&mut self) -> Option<Value> {
        self.data.pop()
    }

    fn peek_n(&self, n: usize) -> Option<&[Value]> {
        let len = self.data.len();
        len.checked_sub(n).and_then(|start| self.data.get(start..))
    }

    fn remove_n(&mut self, n: usize) -> bool {
        let Some(new_len) = self.data.len().checked_sub(n) else {
            return false;
        };
        self.data.truncate(new_len);
        true
    }

    fn pop_n<const N: usize>(&mut self) -> Option<[Value; N]> {
        let mut result = [const { MaybeUninit::uninit() }; N];
        for i in (0..N).rev() {
            result[i].write(self.data.pop()?);
        }
        Some(result.map(|v| unsafe { v.assume_init() }))
    }
}

macro_rules! arithmetic_operator {
    ($this:ident, $op:tt, $check_error:literal, $type_error:literal) => {{
        let Some([lhs, rhs]) = $this.memory.pop_n::<2>() else {
            return $this.runtime_error("Not enough arguments on the stack");
        };

        let result = match (lhs, rhs) {
            (Value::U8(a), Value::U8(b)) => {
                if let Some(result) = <u8>::$op(a, b) {
                    Value::U8(result)
                } else {
                    return $this.runtime_error(format_args!($check_error, a, b));
                }
            }
            (Value::U64(a), Value::U64(b)) => {
                if let Some(result) = <u64>::$op(a, b) {
                    Value::U64(result)
                } else {
                    return $this.runtime_error(format_args!($check_error, a, b));
                }
            }
            (a, b) => {
                return $this.runtime_error(format_args!($type_error, a.type_of(), b.type_of()));
            }
        };

        $this.memory.push(result);
    }};
}

macro_rules! comparison_operator {
    ($this:ident, $op:tt) => {{
        let Some([lhs, rhs]) = $this.memory.pop_n::<2>() else {
            return $this.runtime_error("Not enough arguments on the stack");
        };

        let result = match (lhs, rhs) {
            (Value::U8(a), Value::U8(b)) => Value::Bool(a $op b),
            (Value::U64(a), Value::U64(b)) => Value::Bool(a $op b),
            (a, b) => {
                return $this.runtime_error(format_args!("Cannot compare {} and {}", a.type_of(), b.type_of()));
            }
        };

        $this.memory.push(result);
    }};
}

macro_rules! bitwise_operator {
    ($this:ident, $op:tt) => {{
        let Some([lhs, rhs]) = $this.memory.pop_n::<2>() else {
            return $this.runtime_error("Not enough arguments on the stack");
        };

        let result = match (lhs, rhs) {
            (Value::U8(a), Value::U8(b)) => Value::U8(a $op b),
            (Value::U64(a), Value::U64(b)) => Value::U64(a $op b),
            (a, b) => {
                return $this.runtime_error(
                    format_args!("Cannot calculate {} {} {}", a.type_of(), stringify!($op), b.type_of()),
                );
            }
        };

        $this.memory.push(result);
    }};
}

impl Function {
    fn stack_frame<'p>(&'p self) -> Frame {
        Frame {
            program_counter: self.address,
            name: self.signature.name,
            return_type: self.return_type,
            loops: Vec::new(),
            frame_pointer: 0,
        }
    }
}

impl<'p> EvalContext<'p> {
    fn string(&self, index: StringIndex) -> &str {
        self.program
            .strings
            .lookup_value_by_index(index)
            .unwrap_or("<unknown>")
    }

    fn load_function_by_name(&self, name: &str, args: &[Value]) -> Option<&'p Function> {
        let name = self.program.strings.lookup_index_by_value(name)?;
        let function = self.program.functions.get_by_name(name)?;

        self.typecheck_args(function, args).then_some(function)
    }

    fn typecheck_args(&self, function: &'p Function, args: &[Value]) -> bool {
        let signature = &function.signature;
        signature.args.len() == args.len()
            && signature
                .args
                .iter()
                .zip(args.iter())
                .all(|((_, arg_type), arg)| *arg_type == arg.type_of())
    }

    pub fn new(program: &'p Program) -> Self {
        static EMPTY_FUNCTION: Function = Function {
            signature: FunctionSignature {
                name: StringIndex::default(),
                location: Location { start: 0, end: 0 },
                args: vec![],
            },
            return_type: Type::Void,
            address: 0,
            length: 0,
            scope_size: 0,
        };

        let mut this = EvalContext {
            intrinsics: HashMap::new(),
            state: EvalState::Idle,
            program,
            memory: Memory::new(),
            current_frame: EMPTY_FUNCTION.stack_frame(),
            call_stack: vec![],
        };

        this.add_intrinsic("print", |program, args| {
            for value in args.iter() {
                match value {
                    Value::String(s) => print!("{}", program.string(*s)),
                    Value::U64(v) => print!("{v}"),
                    Value::Bool(v) => print!("{v}"),
                    Value::Void => print!("(void)"),
                    Value::U8(v) => print!("{v}"),
                    Value::F32(v) => print!("{v}"),
                    Value::F64(v) => print!("{v}"),
                    Value::Reference(..) => print!("Cannot print reference"),
                }
            }
            Ok(Value::Void)
        });

        this.memory.allocate(program.globals.values.len());
        for (name, value) in program.globals.values.iter() {
            // TODO after compilation we shouldn't need to store mutability.
            this.memory
                .store(
                    VarId::Global(*name, value.is_mutable()),
                    value.value().clone(),
                )
                .unwrap();
        }

        this
    }

    fn store(&mut self, name: VarId) -> Result<(), EvalEvent> {
        let Some(value) = self.memory.data.pop() else {
            return Err(
                self.runtime_error("Cannot assign to variable without a value on the stack")
            );
        };

        match self.memory.store(name, value) {
            Ok(()) => Ok(()),
            Err(e) => Err(self.runtime_error(format_args!("Failed to store value: {e}"))),
        }
    }

    fn load(&mut self, name: VarId) -> Result<(), EvalEvent> {
        match self.memory.load(name) {
            Ok(value) => {
                self.memory.push(value);
                Ok(())
            }
            Err(e) => Err(self.runtime_error(format_args!("Failed to load value: {e}"))),
        }
    }

    pub fn run(&mut self) -> EvalEvent {
        if matches!(self.state, EvalState::Idle) {
            // Initialize the first frame with the main program
            let function = self.load_function_by_name("main", &[]).unwrap();
            self.current_frame = function.stack_frame();

            self.memory.allocate(function.scope_size);

            self.state = EvalState::Running;
        }

        self.execute()
    }

    fn execute(&mut self) -> EvalEvent {
        match self.state {
            EvalState::WaitingForFunctionResult(name, _) => {
                return self.runtime_error(format!(
                    "Function '{}' is still waiting for a result",
                    self.string(name)
                ));
            }
            _ => (),
        }

        loop {
            let instruction = self.current_instruction();

            //println!(
            //    "{}",
            //    crate::error::MarkInSource(
            //        &self.program.source,
            //        self.current_location(),
            //        &format!(
            //            "Executing instruction: {}",
            //            instruction.disasm(&self.program)
            //        ),
            //        "",
            //    )
            //);

            match instruction {
                Instruction::Push(value) => self.memory.push(value),
                Instruction::Pop => {
                    if self.memory.pop().is_none() {
                        return self.runtime_error("Cannot pop from an empty stack");
                    }
                }
                Instruction::Load(name) => {
                    if let Err(e) = self.load(name) {
                        return e;
                    }
                }
                Instruction::Store(name) => {
                    if let Err(e) = self.store(name) {
                        return e;
                    }
                }
                Instruction::Call(function_index, arg_count) => {
                    let function = self.program.functions.get(function_index).unwrap();
                    if let Err(e) = self.call_function(function, arg_count) {
                        return e;
                    }
                    continue; // Skip the step() call below, as we already stepped (into function)
                }
                Instruction::CallByName(name, arg_count) => {
                    let Some(args) = self.memory.peek_n(arg_count) else {
                        return self.runtime_error("Not enough arguments on the stack");
                    };
                    if let Some(intrinsic) = self.intrinsics.get(&name) {
                        match intrinsic(&self, &args) {
                            Ok(value) => {
                                self.memory.remove_n(arg_count);
                                self.memory.push(value)
                            }
                            Err(err) => return err,
                        }
                    } else {
                        match self.program.functions.get_by_name(name) {
                            Some(function) => {
                                if let Err(e) = self.call_function(function, arg_count) {
                                    return e;
                                }
                            }
                            None => {
                                // If the function is not found, return an error
                                self.state = EvalState::WaitingForFunctionResult(name, arg_count);
                                self.current_frame.step();
                                return EvalEvent::UnknownFunctionCall;
                            }
                        }
                    }
                }
                Instruction::Return => {
                    let retval = if self.current_frame.return_type == Type::Void {
                        // Push a Void as the return value
                        Value::Void
                    } else if let Some(value) = self.memory.pop() {
                        // There is a value on the stack to return
                        // Type check the return value
                        if value.type_of() != self.current_frame.return_type {
                            return self.runtime_error(format_args!(
                                "Type mismatch: expected {}, found {}",
                                self.current_frame.return_type,
                                value.type_of()
                            ));
                        }
                        value
                    } else {
                        return self.runtime_error(format_args!(
                            "Function '{}' should return a '{}' value.",
                            self.string(self.current_frame.name),
                            self.current_frame.return_type
                        ));
                    };

                    if let Some(frame) = self.call_stack.pop() {
                        self.current_frame = frame;
                        // Deallocate the current call frame
                        self.memory.truncate(self.current_frame.frame_pointer);
                        // Push the return value onto the eval stack
                        self.memory.push(retval);
                    } else {
                        // Outermost function has returned
                        self.memory.truncate(self.program.globals.values.len());
                        self.state = EvalState::Running;
                        return EvalEvent::Complete(retval);
                    }
                }
                Instruction::JumpForward(n) => {
                    self.current_frame.step_forward(n + 1);
                    continue; // Skip the step() call below, as we already stepped forward
                }
                Instruction::JumpForwardIf(n, bool) => match self.memory.pop() {
                    Some(Value::Bool(b)) => {
                        if b == bool {
                            self.current_frame.step_forward(n + 1);
                            continue; // Skip the step() call below, as we already stepped forward
                        }
                    }
                    Some(v) => {
                        return self.runtime_error(format_args!(
                            "Type mismatch: expected bool, found {}",
                            v.type_of()
                        ));
                    }
                    None => return self.runtime_error("Cannot pop value for comparison"),
                },
                Instruction::JumpBack(n) => {
                    if !self.current_frame.step_back(n + 1) {
                        return self
                            .runtime_error("Cannot step back beyond the start of the program");
                    }
                    continue; // Skip the step() call below, as we already stepped back
                }
                Instruction::LoopStart(length) => {
                    let start = self.current_frame.program_counter;
                    self.current_frame.loops.push((start, length));
                }
                Instruction::LoopEnd => {
                    self.current_frame.loops.pop().expect("No loop to end");
                }
                Instruction::Continue => {
                    match self.current_frame.loops.last() {
                        Some((start, _)) => {
                            // Jump back to the start of the loop
                            self.current_frame.program_counter = *start;
                        }
                        None => return self.runtime_error("Continue outside of a loop"),
                    }
                }
                Instruction::Break => match self.current_frame.loops.pop() {
                    Some((start, length)) => {
                        self.current_frame.program_counter = start + length - 1;
                    }
                    None => return self.runtime_error("Break outside of a loop"),
                },
                Instruction::TestLessThan => comparison_operator!(self, <),
                Instruction::TestLessThanOrEqual => comparison_operator!(self, <=),
                Instruction::TestEquals => comparison_operator!(self, ==),
                Instruction::TestNotEquals => comparison_operator!(self, !=),
                Instruction::BitwiseOr => bitwise_operator!(self, |),
                Instruction::BitwiseXor => bitwise_operator!(self, ^),
                Instruction::BitwiseAnd => bitwise_operator!(self, &),
                Instruction::ShiftLeft => bitwise_operator!(self, <<),
                Instruction::ShiftRight => bitwise_operator!(self, >>),
                Instruction::Add => {
                    arithmetic_operator!(
                        self,
                        checked_add,
                        "{} + {} overflowed",
                        "Cannot add {} and {}"
                    );
                }
                Instruction::Subtract => {
                    arithmetic_operator!(
                        self,
                        checked_sub,
                        "{} - {} underflowed",
                        "Cannot subtract {} and {}"
                    );
                }
                Instruction::Multiply => {
                    arithmetic_operator!(
                        self,
                        checked_mul,
                        "{} * {} overflowed",
                        "Cannot multiply {} and {}"
                    );
                }
                Instruction::Divide => {
                    arithmetic_operator!(
                        self,
                        checked_div,
                        "Division by zero: {} / {}",
                        "Cannot divide {} and {}"
                    );
                }
                Instruction::Negate => todo!(),
                Instruction::InvertBoolean => {
                    let Some(operand) = self.memory.pop() else {
                        return self.runtime_error("Not enough arguments on the stack");
                    };

                    let result = match operand {
                        Value::Bool(a) => Value::Bool(!a),
                        Value::U8(a) => Value::U8(!a),
                        Value::U64(a) => Value::U64(!a),
                        a => {
                            return self
                                .runtime_error(format_args!("Cannot invert {}", a.type_of()));
                        }
                    };

                    self.memory.push(result);
                }
            }

            self.current_frame.step();
        }
    }

    pub fn unknown_function_info(&self) -> Option<(&str, &[Value])> {
        if let EvalState::WaitingForFunctionResult(name, arg_count) = self.state {
            let name = self.string(name);
            let args = self.memory.peek_n(arg_count)?;
            Some((name, args))
        } else {
            None
        }
    }

    pub fn set_return_value(&mut self, value: Value) -> Result<(), EvalEvent> {
        match std::mem::replace(&mut self.state, EvalState::Running) {
            EvalState::WaitingForFunctionResult(..) => {
                self.state = EvalState::Running;
                self.memory.push(value);
                Ok(())
            }
            EvalState::Running => {
                Err(self.runtime_error("Cannot set return value outside of a function call"))
            }
            EvalState::Idle => Err(self.runtime_error("Cannot set return value when not running")),
        }
    }

    fn add_intrinsic(
        &mut self,
        name: &'static str,
        fun: fn(&Self, &[Value]) -> Result<Value, EvalEvent>,
    ) {
        // If the name is not interned yet, the intrinsic is not used and not needed.
        if let Some(name_index) = self.program.strings.lookup_index_by_value(name) {
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

    fn call_function(&mut self, function: &'p Function, arg_count: usize) -> Result<(), EvalEvent> {
        let Some(args) = self.memory.peek_n(arg_count) else {
            return Err(self.runtime_error("Not enough arguments on the stack"));
        };
        if !self.typecheck_args(function, args) {
            return Err(self.runtime_error(format_args!(
                "Function '{}' called with incorrect argument types. Expected: ('{}'), found: ('{}')",
                self.string(function.signature.name),
                function
                    .signature
                    .args
                    .iter()
                    .map(|(_, t)| t.to_string())
                    .collect::<Vec<_>>()
                    .join("', '"),
                args.iter()
                    .map(|v| v.type_of().to_string())
                    .collect::<Vec<_>>()
                    .join("', '")
            )));
        }
        self.current_frame.frame_pointer = self.memory.sp;
        // Allocate memory for the function's local variables.
        self.memory.allocate(function.scope_size - arg_count);
        // Massage the stack pointer so that arguments are in scope.
        self.memory.sp -= arg_count;

        let old_frame = std::mem::replace(&mut self.current_frame, function.stack_frame());
        self.call_stack.push(old_frame);

        Ok(())
    }

    #[cold]
    fn runtime_error(&self, message: impl ToString) -> EvalEvent {
        EvalEvent::Error(EvalError(message.to_string().into_boxed_str()))
    }

    fn current_instruction(&self) -> Instruction {
        self.program.code[self.current_frame.program_counter]
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn test_vm() {
        crate::test::run_eval_tests("tests/eval/");
    }
}
