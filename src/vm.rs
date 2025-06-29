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

    outer_function_name: StringIndex,
    program_counter: CodeAddress,
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

    fn copy(
        &mut self,
        addr: MemoryAddress,
        dst: MemoryAddress,
        amount: usize,
    ) -> Result<(), String> {
        let from = self.address(addr);
        let to = self.address(dst);

        self.data.copy_within(from..from + amount, to);
        Ok(())
    }

    fn as_mut(&mut self, addr: Range<MemoryAddress>) -> Result<&mut [u8], String> {
        let from = self.address(addr.start);
        let to = self.address(addr.end);
        self.data.get_mut(from..to).ok_or_else(|| {
            format!("Trying to load value from address {from}..{to} which is out of bounds")
        })
    }
}

macro_rules! dispatch_type {
    ($op:ident, $pat:tt => $code:tt) => {{
        macro_rules! inner { $pat => { $code }; }

        match $op {
            codegen::Type::Int => inner!(u64),
            codegen::Type::SignedInt => inner!(i64),
            codegen::Type::Float => inner!(f64),
            codegen::Type::Bool => inner!(bool),
            codegen::Type::String => inner!(StringIndex),
            codegen::Type::Void => inner!(()),
        }
    }};
}

macro_rules! for_each_binary_operator {
    ($pat:tt => $code:tt) => {
        macro_rules! inner { $pat => $code; }

        inner!(codegen::BinaryOperator::TestLessThan, less_than);
        inner!(
            codegen::BinaryOperator::TestLessThanOrEqual,
            less_than_or_equal
        );
        inner!(codegen::BinaryOperator::TestEquals, equals);
        inner!(codegen::BinaryOperator::TestNotEquals, not_equals);
        inner!(codegen::BinaryOperator::BitwiseOr, bitwise_or);
        inner!(codegen::BinaryOperator::BitwiseXor, bitwise_xor);
        inner!(codegen::BinaryOperator::BitwiseAnd, bitwise_and);
        inner!(codegen::BinaryOperator::ShiftLeft, shift_left);
        inner!(codegen::BinaryOperator::ShiftRight, shift_right);
        inner!(codegen::BinaryOperator::Add, add);
        inner!(codegen::BinaryOperator::Subtract, subtract);
        inner!(codegen::BinaryOperator::Multiply, multiply);
        inner!(codegen::BinaryOperator::Divide, divide);
    };
}

macro_rules! dispatch_binary_operator {
    ($op:ident, $pat:tt => $code:tt) => {{
        macro_rules! inner { $pat => $code; }

        match $op {
            codegen::BinaryOperator::TestLessThan => inner!(less_than),
            codegen::BinaryOperator::TestLessThanOrEqual => inner!(less_than_or_equal),
            codegen::BinaryOperator::TestEquals => inner!(equals),
            codegen::BinaryOperator::TestNotEquals => inner!(not_equals),
            codegen::BinaryOperator::BitwiseOr => inner!(bitwise_or),
            codegen::BinaryOperator::BitwiseXor => inner!(bitwise_xor),
            codegen::BinaryOperator::BitwiseAnd => inner!(bitwise_and),
            codegen::BinaryOperator::ShiftLeft => inner!(shift_left),
            codegen::BinaryOperator::ShiftRight => inner!(shift_right),
            codegen::BinaryOperator::Add => inner!(add),
            codegen::BinaryOperator::Subtract => inner!(subtract),
            codegen::BinaryOperator::Multiply => inner!(multiply),
            codegen::BinaryOperator::Divide => inner!(divide),
        }
    }};
}

macro_rules! for_each_unary_operator {
    ($pat:tt => $code:tt) => {
        macro_rules! inner { $pat => $code; }

        inner!(codegen::UnaryOperator::Negate, negate);
        inner!(codegen::UnaryOperator::Not, not);
    };
}

macro_rules! dispatch_unary_operator {
    ($op:ident, $pat:tt => $code:tt) => {{
        macro_rules! inner { $pat => { $code }; }

        match $op {
            codegen::UnaryOperator::Negate => inner!(negate),
            codegen::UnaryOperator::Not => inner!(not),
        }
    }};
}

for_each_binary_operator!(
    ($_name:path, $op:ident) => {
        impl codegen::Type {
            fn $op(
                self,
                ctx: &mut EvalContext<'_>,
                dst: MemoryAddress,
                lhs: MemoryAddress,
                rhs: MemoryAddress,
            ) -> Result<(), EvalEvent> {
                dispatch_type!(self, ($ty:ty) => {
                    let lhs = ctx.load::<$ty>(lhs)?;
                    let rhs = ctx.load::<$ty>(rhs)?;

                    match <$ty>::$op(lhs, rhs) {
                        Ok(result) => ctx.store(dst, result),
                        Err(e) => Err(operator_error(ctx, e)),
                    }
                })
            }
        }
    }
);
for_each_unary_operator!(
    ($_name:path, $op:ident) => {
        impl codegen::Type {
            fn $op(
                self,
                ctx: &mut EvalContext<'_>,
                dst: MemoryAddress,
                operand: MemoryAddress,
            ) -> Result<(), EvalEvent> {
                dispatch_type!(self, ($ty:ty) => {
                    let operand = ctx.load::<$ty>(operand)?;

                    match <$ty>::$op(operand) {
                        Ok(result) => ctx.store(dst, result),
                        Err(e) => Err(operator_error(ctx, e)),
                    }
                })
            }
        }
    }
);

#[cold]
fn operator_error(ctx: &EvalContext<'_>, error: OperatorError) -> EvalEvent {
    ctx.runtime_error(format_args!("Failed to apply operator: {error:?}"))
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
            source,
            strings: strings.clone(),
            orig_strings: strings,
            outer_function_name: StringIndex::dummy(), // Will be set when the first function is called
            program_counter: CodeAddress(0),
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
                .sum::<usize>()
                + 16, // SP + PC
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

        let sp = MemoryAddress::Global(self.memory.data.len());
        if let Err(e) = self.store(sp - 16, 0_u64) {
            return e;
        }
        if let Err(e) = self.store(sp - 8, 0_u64) {
            return e;
        }

        self.outer_function_name = function.name;
        self.program_counter = function.entry_point;
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
            if let Err(e) = self.step() {
                return e;
            }
        }
    }

    fn step(&mut self) -> Result<(), EvalEvent> {
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
                self.call_function(function, sp)?;
                return Ok(()); // Skip the step() call below, as we already stepped (into function)
            }
            Instruction::CallNamed(function_name, ty, sp) => {
                let Some(intrinsic) = self.intrinsics.get(&function_name) else {
                    self.state = EvalState::WaitingForFunctionResult(function_name, ty, sp);
                    self.program_counter += 1;
                    return Err(EvalEvent::UnknownFunctionCall(function_name));
                };

                let first_arg = sp + ty.size_of();
                let retval = intrinsic.call(self, first_arg);
                self.store_typed(sp, retval)?;
            }
            Instruction::Return => {
                let pc = self.load::<u64>(MemoryAddress::Global(self.memory.sp - 16))?;
                let sp = self.load::<u64>(MemoryAddress::Global(self.memory.sp - 8))?;

                if sp == 0 {
                    // Outermost function has returned

                    let function = self
                        .program
                        .functions
                        .get(&self.outer_function_name)
                        .unwrap();

                    let retval = self
                        .memory
                        .load_typed(MemoryAddress::Local(0), function.return_type)
                        .unwrap();
                    self.state = EvalState::Running;

                    return Err(EvalEvent::Complete(retval));
                } else {
                    self.memory.sp = sp as usize;
                    self.program_counter = CodeAddress(pc as usize);
                }
            }
            Instruction::Jump(n) => {
                self.program_counter = n;
                return Ok(()); // Skip the step() call below, as we already stepped forward
            }
            Instruction::JumpIfFalse(condition, target) => {
                if !self.load::<bool>(condition)? {
                    self.program_counter = target;
                    return Ok(()); // Skip the step() call below, as we already stepped forward
                }
            }
            Instruction::BinaryOperator {
                ty,
                operator,
                dst,
                lhs,
                rhs,
            } => {
                dispatch_binary_operator!(operator, ($function:tt) => { ty.$function(self, dst, lhs, rhs)? })
            }
            Instruction::UnaryOperator {
                ty,
                operator,
                dst,
                op,
            } => {
                dispatch_unary_operator!(operator, ($function:tt) => { ty.$function(self, dst, op)? })
            }

            Instruction::AddressOf(dst, lhs) => {
                let value = self.memory.address(lhs) as u64;
                self.store_typed(dst, TypedValue::Int(value))?;
            }
            Instruction::Dereference(ty, dst, lhs) => {
                let address = self.load::<u64>(lhs)?;
                let address = MemoryAddress::Global(address as usize);

                self.copy(address, dst, ty.size_of())?;
            }
            Instruction::Copy(dst, from, amount) => self.copy(from, dst, amount as usize)?,
            Instruction::DerefCopy(ty, addr, from) => {
                let dst = self.load::<u64>(addr)?;
                let dst = MemoryAddress::Global(dst as usize);

                self.copy(from, dst, ty.size_of())?;
            }
            Instruction::LoadValue(addr, value) => self.store_typed(addr, value)?,
        }

        self.program_counter += 1;

        Ok(())
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

    fn copy(
        &mut self,
        addr: MemoryAddress,
        dst: MemoryAddress,
        amount: usize,
    ) -> Result<(), EvalEvent> {
        self.memory
            .copy(addr, dst, amount)
            .map_err(|e| self.runtime_error(format_args!("Failed to copy value: {e}")))
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
        let mut fns = self
            .program
            .functions
            .values()
            .map(|f| (f.entry_point.0, f.name))
            .collect::<Vec<_>>();

        fns.sort_by_key(|(e, _f)| *e);

        let mut prev_sp = self.memory.sp;
        let mut i = 0;
        while prev_sp != 0 {
            i += 1;
            let pc = self
                .load::<u64>(MemoryAddress::Global(prev_sp - 16))
                .unwrap();
            let sp = self
                .load::<u64>(MemoryAddress::Global(prev_sp - 8))
                .unwrap();

            let name = if sp == 0 {
                self.string(self.outer_function_name)
            } else {
                let mut name = None;
                for (ep, func) in fns.iter().copied() {
                    if ep > pc as usize {
                        break;
                    }
                    name = Some(func);
                }

                name.map(|idx| self.string(idx)).unwrap_or("<unknown>")
            };

            println!("Frame {i}: {name}");
            prev_sp = sp as usize;
        }
    }

    fn call_function(
        &mut self,
        function: &'p Function,
        sp: MemoryAddress,
    ) -> Result<(), EvalEvent> {
        self.store(sp - 16, self.program_counter.0 as u64)?;
        self.store(sp - 8, self.memory.sp as u64)?;
        self.memory.sp = self.memory.address(sp);
        // Allocate memory for the function's local variables.
        self.memory.allocate(function.stack_size);
        self.program_counter = function.entry_point;

        Ok(())
    }

    #[cold]
    fn runtime_error(&self, message: impl ToString) -> EvalEvent {
        EvalEvent::Error(EvalError(message.to_string().into_boxed_str()))
    }

    fn current_instruction(&self) -> Instruction {
        self.program.code[self.program_counter.0]
    }

    fn current_location(&self) -> Location {
        let pc = self.program_counter;
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
                println!("unknown function call {:?}", self.string(fn_name));
                self.print_backtrace();
                todo!();
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
