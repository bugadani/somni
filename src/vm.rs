use std::{collections::HashMap, ops::Range};

use crate::{
    codegen::{self, CodeAddress, Function, Instruction, MemoryAddress, Type, ValueType},
    error::MarkInSource,
    eval::{ExprContext, ExpressionVisitor, OperatorError, TypedValue},
    string_interner::{StringIndex, Strings},
};
use somni_lexer::Location;

#[derive(Clone, Debug)]
pub struct EvalError(Box<str>);

impl EvalError {
    pub fn mark<'a>(&'a self, context: &'a EvalContext, message: &'a str) -> MarkInSource<'a> {
        MarkInSource(context.source, context.current_location(), message, &self.0)
    }
}

#[derive(Clone, Debug)]
pub enum EvalEvent {
    UnknownFunctionCall(StringIndex),
    Error(EvalError),
    Complete(TypedValue),
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum EvalState {
    Idle,
    Running,
    WaitingForFunctionResult(StringIndex, Type, MemoryAddress),
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

pub trait Arguments {
    fn read(ctx: &EvalContext<'_>, sp: MemoryAddress) -> Self;
}

impl Arguments for () {
    fn read(_: &EvalContext<'_>, _: MemoryAddress) -> Self {
        ()
    }
}

impl<V> Arguments for (V,)
where
    V: ValueType,
{
    fn read(ctx: &EvalContext<'_>, sp: MemoryAddress) -> Self {
        (ctx.load::<V>(sp).unwrap(),)
    }
}

pub struct SomniFn<'p> {
    func: Box<dyn Fn(&EvalContext<'p>, MemoryAddress) -> TypedValue>,
}

impl<'p> SomniFn<'p> {
    pub fn new<A, R, F>(func: F) -> Self
    where
        F: Fn(A) -> R,
        F: 'static,
        A: Arguments + 'static,
        R: ValueType + 'static,
    {
        Self {
            func: Box::new(move |ctx, sp| {
                let args = A::read(ctx, sp);
                func(args).into()
            }),
        }
    }

    fn call(&self, ctx: &EvalContext<'p>, sp: MemoryAddress) -> TypedValue {
        (self.func)(ctx, sp)
    }
}

pub struct EvalContext<'p> {
    source: &'p str,
    orig_strings: &'p Strings,
    program: &'p codegen::Program,
    strings: Strings,
    intrinsics: HashMap<StringIndex, SomniFn<'p>>,
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
    data: Vec<u8>,
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
        self.data.resize(self.data.len().max(self.sp + size), 0);
    }

    fn address(&self, var_id: MemoryAddress) -> usize {
        match var_id {
            MemoryAddress::Global(address) => address,
            MemoryAddress::Local(address) => self.sp + address,
        }
    }

    fn load(&self, addr: MemoryAddress, len: usize) -> Result<&[u8], String> {
        let address = self.address(addr);

        let Some(variable) = self.data.get(address..address + len) else {
            return Err(format!(
                "Trying to load value from address {address} which is out of bounds"
            ));
        };

        Ok(variable)
    }

    fn load_typed(
        &self,
        local: MemoryAddress,
        return_type: codegen::Type,
    ) -> Result<TypedValue, String> {
        let data = self.load(local, return_type.size_of())?;

        Ok(TypedValue::from_typed_bytes(return_type, data))
    }

    fn as_mut(&mut self, addr: Range<MemoryAddress>) -> Result<&mut [u8], String> {
        let from = self.address(addr.start);
        let to = self.address(addr.end);
        self.data.get_mut(from..to).ok_or_else(|| {
            format!("Trying to load value from address {from}..{to} which is out of bounds")
        })
    }
}

macro_rules! dispatch_binary {
    ($op:ident) => {
        fn $op(
            self,
            ctx: &mut EvalContext<'_>,
            dst: MemoryAddress,
            lhs: MemoryAddress,
            rhs: MemoryAddress,
        ) -> Result<(), EvalEvent> {
            match self {
                codegen::Type::Int => Self::binary_operator(ctx, dst, lhs, rhs, u64::$op),
                codegen::Type::SignedInt => Self::binary_operator(ctx, dst, lhs, rhs, i64::$op),
                codegen::Type::Float => Self::binary_operator(ctx, dst, lhs, rhs, f64::$op),
                codegen::Type::Bool => Self::binary_operator(ctx, dst, lhs, rhs, bool::$op),
                codegen::Type::String => {
                    Self::binary_operator(ctx, dst, lhs, rhs, StringIndex::$op)
                }
                codegen::Type::Address => Self::binary_operator(ctx, dst, lhs, rhs, u64::$op),
                codegen::Type::Void => Self::binary_operator(ctx, dst, lhs, rhs, <()>::$op),
            }
        }
    };
}

macro_rules! dispatch_unary {
    ($op:ident) => {
        fn $op(
            self,
            ctx: &mut EvalContext<'_>,
            dst: MemoryAddress,
            operand: MemoryAddress,
        ) -> Result<(), EvalEvent> {
            match self {
                codegen::Type::Int => {
                    Self::unary_operator(ctx, dst, operand, <u64 as ValueType>::$op)
                }
                codegen::Type::SignedInt => {
                    Self::unary_operator(ctx, dst, operand, <i64 as ValueType>::$op)
                }
                codegen::Type::Float => {
                    Self::unary_operator(ctx, dst, operand, <f64 as ValueType>::$op)
                }
                codegen::Type::Bool => {
                    Self::unary_operator(ctx, dst, operand, <bool as ValueType>::$op)
                }
                codegen::Type::String => {
                    Self::unary_operator(ctx, dst, operand, <StringIndex as ValueType>::$op)
                }
                codegen::Type::Address => {
                    Self::unary_operator(ctx, dst, operand, <u64 as ValueType>::$op)
                }
                codegen::Type::Void => {
                    Self::unary_operator(ctx, dst, operand, <() as ValueType>::$op)
                }
            }
        }
    };
}

impl codegen::Type {
    fn unary_operator<V: ValueType, R: ValueType>(
        ctx: &mut EvalContext<'_>,
        dst: MemoryAddress,
        operand: MemoryAddress,
        op: fn(V) -> Result<R, OperatorError>,
    ) -> Result<(), EvalEvent> {
        let operand = ctx.load::<V>(operand)?;

        match op(operand) {
            Ok(result) => ctx.store(dst, result),
            Err(e) => Err(ctx.runtime_error(format!("Failed to apply operator: {e:?}"))),
        }
    }

    fn binary_operator<V: ValueType, R: ValueType>(
        ctx: &mut EvalContext<'_>,
        dst: MemoryAddress,
        lhs: MemoryAddress,
        rhs: MemoryAddress,
        op: fn(V, V) -> Result<R, OperatorError>,
    ) -> Result<(), EvalEvent> {
        let lhs = ctx.load::<V>(lhs)?;
        let rhs = ctx.load::<V>(rhs)?;

        match op(lhs, rhs) {
            Ok(result) => ctx.store(dst, result),
            Err(e) => Err(ctx.runtime_error(format!("Failed to apply operator: {e:?}"))),
        }
    }

    dispatch_binary!(less_than);
    dispatch_binary!(less_than_or_equal);
    dispatch_binary!(equals);
    dispatch_binary!(not_equals);
    dispatch_binary!(bitwise_or);
    dispatch_binary!(bitwise_xor);
    dispatch_binary!(bitwise_and);
    dispatch_binary!(shift_left);
    dispatch_binary!(shift_right);
    dispatch_binary!(add);
    dispatch_binary!(subtract);
    dispatch_binary!(multiply);
    dispatch_binary!(divide);

    dispatch_unary!(not);
    dispatch_unary!(negate);
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
        EvalContext {
            intrinsics: HashMap::new(),
            state: EvalState::Idle,
            program,
            memory: Memory::new(),
            call_stack: vec![],
            source,
            strings: strings.clone(),
            orig_strings: strings,
            current_frame: Frame {
                program_counter: CodeAddress(0),
                name: StringIndex::dummy(), // Will be set when the first function is called
                frame_pointer: 0,           // Will be set when the first function is called
            },
        }
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
            // Restore VM state.
            self.reset();

            // Initialize the first frame with the main program
            self.call("main", &[])
        } else {
            self.execute()
        }
    }

    pub fn reset(&mut self) {
        // TODO: only if eval_expression is supported
        self.strings = self.orig_strings.clone();
        self.state = EvalState::Idle;
        self.memory = Memory::new();
        self.memory.allocate(
            self.program
                .globals
                .values()
                .map(|v| v.ty().size_of())
                .sum(),
        );
        let mut address = 0;
        for (_, def) in self.program.globals.iter() {
            self.store_typed(MemoryAddress::Global(address), def.value())
                .unwrap();
            address += def.ty().size_of();
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
    pub fn call(&mut self, func: &str, args: &[TypedValue]) -> EvalEvent {
        let Some(function) = self.load_function_by_name(func) else {
            return self.runtime_error(format!("Unknown function: {func}"));
        };

        self.current_frame = function.stack_frame();

        self.memory.sp = self.memory.data.len();
        self.memory.allocate(function.stack_size);

        if args.len() != function.arguments.len() {
            return self.runtime_error(format!(
                "Function '{func}' expects {} arguments, but got {}",
                function.arguments.len(),
                args.len()
            ));
        }

        // Store the function arguments as temporaries in the caller's stack frame.
        for (i, ((addr, ty), arg)) in function.arguments.iter().zip(args.iter()).enumerate() {
            if *ty != arg.type_of() {
                return self.runtime_error(format!(
                    "Function '{func}' expects argument {} to be of type {ty}, but got {}",
                    i + 1,
                    arg.type_of()
                ));
            }
            if let Err(e) = self.store_typed(*addr, *arg) {
                return e;
            }
        }

        self.state = EvalState::Running;

        self.execute()
    }

    /// Evaluates an expression and returns the result.
    ///
    /// An expression can use globals and functions defined in the program, but it cannot
    /// call functions that are not defined in the program.
    pub fn eval_expression(&mut self, expression: &str) -> TypedValue {
        if !matches!(self.state, EvalState::Idle) {
            panic!("Cannot evaluate expression while the VM is running");
        }

        // TODO: we can allow new globals to be defined in the expression, but that would require
        // storing a copy of the original globals, so that they can be reset?
        let tokens = somni_lexer::tokenize(expression)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        let ast = somni_parser::parse_expression(expression, &tokens).unwrap();

        let mut visitor = ExpressionVisitor {
            context: self,
            source: expression,
        };
        // TODO: handle errors
        visitor.visit_expression(&ast).unwrap()
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
                Instruction::CallNamed(function_name, ty, sp) => {
                    let Some(intrinsic) = self.intrinsics.get(&function_name) else {
                        self.state = EvalState::WaitingForFunctionResult(function_name, ty, sp);
                        self.current_frame.step();
                        return EvalEvent::UnknownFunctionCall(function_name);
                    };

                    let first_arg = sp + ty.size_of();
                    let retval = intrinsic.call(self, first_arg);
                    if let Err(e) = self.store_typed(sp, retval) {
                        return e;
                    }
                }
                Instruction::Return => {
                    let Some(frame) = self.call_stack.pop() else {
                        // Outermost function has returned

                        let function = self
                            .program
                            .functions
                            .get(&self.current_frame.name)
                            .unwrap();

                        let retval = self
                            .memory
                            .load_typed(MemoryAddress::Local(0), function.return_type)
                            .unwrap();
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
                    match self.load::<bool>(condition) {
                        Ok(false) => {
                            self.current_frame.program_counter = target;
                            continue; // Skip the step() call below, as we already stepped forward
                        }
                        Ok(true) => {}
                        Err(e) => return e,
                    }
                }
                Instruction::BinaryOperator {
                    ty,
                    operator,
                    dst,
                    lhs,
                    rhs,
                } => {
                    let result = match operator {
                        codegen::BinaryOperator::TestLessThan => ty.less_than(self, dst, lhs, rhs),
                        codegen::BinaryOperator::TestLessThanOrEqual => {
                            ty.less_than_or_equal(self, dst, lhs, rhs)
                        }
                        codegen::BinaryOperator::TestEquals => ty.equals(self, dst, lhs, rhs),
                        codegen::BinaryOperator::TestNotEquals => {
                            ty.not_equals(self, dst, lhs, rhs)
                        }
                        codegen::BinaryOperator::BitwiseOr => ty.bitwise_or(self, dst, lhs, rhs),
                        codegen::BinaryOperator::BitwiseXor => ty.bitwise_xor(self, dst, lhs, rhs),
                        codegen::BinaryOperator::BitwiseAnd => ty.bitwise_and(self, dst, lhs, rhs),
                        codegen::BinaryOperator::ShiftLeft => ty.shift_left(self, dst, lhs, rhs),
                        codegen::BinaryOperator::ShiftRight => ty.shift_right(self, dst, lhs, rhs),
                        codegen::BinaryOperator::Add => ty.add(self, dst, lhs, rhs),
                        codegen::BinaryOperator::Subtract => ty.subtract(self, dst, lhs, rhs),
                        codegen::BinaryOperator::Multiply => ty.multiply(self, dst, lhs, rhs),
                        codegen::BinaryOperator::Divide => ty.divide(self, dst, lhs, rhs),
                    };

                    if let Err(e) = result {
                        return e;
                    }
                }
                Instruction::UnaryOperator {
                    ty,
                    operator,
                    dst,
                    op,
                } => {
                    let result = match operator {
                        codegen::UnaryOperator::Negate => ty.negate(self, dst, op),
                        codegen::UnaryOperator::Not => ty.not(self, dst, op),
                    };

                    if let Err(e) = result {
                        return e;
                    }
                }
                Instruction::AddressOf(dst, lhs) => {
                    let value = self.memory.address(lhs) as u64;
                    if let Err(e) = self.store_typed(dst, TypedValue::Int(value)) {
                        return e;
                    }
                }
                Instruction::Dereference(ty, dst, lhs) => {
                    let address = match self.load::<u64>(lhs) {
                        Ok(addr) => MemoryAddress::Global(addr as usize),
                        Err(e) => return e,
                    };

                    if let Err(e) = self.copy(ty, address, dst) {
                        return e;
                    }
                }
                Instruction::Copy(ty, dst, from) => {
                    if let Err(e) = self.copy(ty, from, dst) {
                        return e;
                    }
                }
                Instruction::DerefCopy(ty, addr, from) => {
                    let dst = match self.load::<u64>(addr) {
                        Ok(addr) => MemoryAddress::Global(addr as usize),
                        Err(e) => return e,
                    };

                    if let Err(e) = self.copy(ty, from, dst) {
                        return e;
                    }
                }
                Instruction::LoadValue(addr, value) => {
                    if let Err(e) = self.store_typed(addr, value) {
                        return e;
                    }
                }
            }

            self.current_frame.step();
        }
    }

    fn store<V: ValueType>(&mut self, addr: MemoryAddress, value: V) -> Result<(), EvalEvent> {
        match self.memory.as_mut(addr..addr + V::BYTES) {
            Ok(memory) => {
                value.write(memory);
                Ok(())
            }
            Err(e) => {
                Err(self.runtime_error(format_args!("Failed to store value at address: {e}")))
            }
        }
    }

    fn store_typed(&mut self, addr: MemoryAddress, value: TypedValue) -> Result<(), EvalEvent> {
        match self.memory.as_mut(addr..addr + value.type_of().size_of()) {
            Ok(memory) => {
                value.write(memory);
                Ok(())
            }
            Err(e) => {
                Err(self.runtime_error(format_args!("Failed to store value at address: {e}")))
            }
        }
    }

    fn load<V: ValueType>(&self, addr: MemoryAddress) -> Result<V, EvalEvent> {
        let bytes = self.memory.load(addr, V::BYTES).map_err(|e| {
            self.runtime_error(format_args!("Failed to load value from address: {e}"))
        })?;
        Ok(V::from_bytes(&bytes))
    }

    fn load_typed(&self, addr: MemoryAddress, ty: Type) -> Result<TypedValue, EvalEvent> {
        self.memory
            .load_typed(addr, ty)
            .map_err(|e| self.runtime_error(format_args!("Failed to load value: {e}")))
    }

    fn copy(&mut self, ty: Type, addr: MemoryAddress, dst: MemoryAddress) -> Result<(), EvalEvent> {
        self.load_typed(addr, ty)
            .and_then(|value| self.store_typed(dst, value))
    }

    pub fn unknown_call_args<A: Arguments>(&self) -> Option<A> {
        if let EvalState::WaitingForFunctionResult(_name, ty, sp) = self.state {
            let first_arg = sp + ty.size_of();
            Some(A::read(self, first_arg))
        } else {
            None
        }
    }

    pub fn set_return_value(&mut self, value: TypedValue) -> Result<(), EvalEvent> {
        let EvalState::WaitingForFunctionResult(_, ty, sp) = self.state else {
            return Err(self.runtime_error("No function is currently waiting for a result"));
        };

        if value.type_of() != ty {
            return Err(self.runtime_error(format_args!("Expcted a {ty} as the return value")));
        }

        self.store_typed(sp, value)?;
        self.state = EvalState::Running;
        Ok(())
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

impl<'s> ExprContext for EvalContext<'s> {
    fn intern_string(&mut self, s: &str) -> StringIndex {
        if let Some(string_index) = self.strings.find(s) {
            string_index
        } else {
            let string_index = self.strings.intern(s);
            string_index
        }
    }

    fn try_load_variable(&self, name: &str) -> Option<TypedValue> {
        let name_idx = self
            .strings
            .find(name)
            .unwrap_or_else(|| panic!("Variable '{name}' not found"));
        let global = self
            .program
            .globals
            .get(&name_idx)
            .unwrap_or_else(|| panic!("Variable '{name}' not found in program"));

        let value = self
            .memory
            .load_typed(global.address, global.ty())
            .unwrap_or_else(|_| panic!("Variable '{name}' not found in memory"));

        Some(value)
    }

    fn call_function(&mut self, function_name: &str, args: &[TypedValue]) -> TypedValue {
        match self.call(function_name, &args) {
            EvalEvent::UnknownFunctionCall(fn_name) => {
                todo!("unknown function call {:?}", self.string(fn_name))
            }
            EvalEvent::Error(e) => todo!("{e:?}"),
            EvalEvent::Complete(value) => value,
        }
    }

    fn address_of(&self, name: &str) -> TypedValue {
        let name_idx = self.strings.find(name).unwrap_or_else(|| {
            panic!("Variable '{name}' not found in program");
        });
        let addr = self
            .program
            .globals
            .get_index_of(&name_idx)
            .unwrap_or_else(|| {
                panic!("Variable '{name}' not found in program");
            });
        TypedValue::from(addr as u64)
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn test_vm() {
        crate::test::run_eval_tests("tests/eval/");
    }
}
