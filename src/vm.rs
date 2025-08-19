use std::{collections::HashMap, marker::PhantomData, ops::Range};

use crate::{
    codegen::{self, CodeAddress, Function, Instruction, MemoryAddress},
    string_interner::{StringIndex, Strings},
    strip_ansi,
    types::{MemoryRepr, TypeExt, TypedValue as VmTypedValue, VmTypeSet},
};

use somni_expr::{
    error::MarkInSource,
    value::{LoadOwned, LoadStore, ValueType},
    DynFunction, ExprContext, ExpressionVisitor, FunctionCallError, OperatorError, Type,
};
use somni_parser::{parser, Location};

pub type TypedValue = somni_expr::value::TypedValue<VmTypeSet>;

#[derive(Clone, Debug)]
pub struct EvalError(Box<str>);

impl EvalError {
    pub fn mark<'a>(&'a self, context: &'a EvalContext, message: &'a str) -> MarkInSource<'a> {
        MarkInSource(context.source, context.current_location(), message, &self.0)
    }

    pub fn as_str(&self) -> &str {
        self.0.as_ref()
    }
}

#[derive(Clone, Debug)]
pub enum EvalEvent<V> {
    UnknownFunctionCall(StringIndex),
    Error(EvalError),
    Complete(V),
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

somni_expr::for_all_tuples! {
    ($($arg:ident),*) => {
        impl<$($arg),*> Arguments for ($($arg,)*)
        where
            $($arg: ValueType + MemoryRepr,)*
        {
            #[allow(non_snake_case)]
            fn read(ctx: &EvalContext<'_>, sp: MemoryAddress) -> Self {
                let offset = 0;
                $(
                let $arg = ctx.load::<$arg>(sp + offset).unwrap();
                let offset = offset + <$arg>::BYTES;
                )*

                #[allow(clippy::unused_unit)]
                ($($arg,)*)
            }
        }
    };
}

pub trait NativeFunction<A>: DynFunction<A, VmTypeSet> + Clone {
    fn call_from_vm(
        &self,
        ctx: &mut EvalContext<'_>,
        sp: MemoryAddress,
    ) -> Result<VmTypedValue, EvalEvent<TypedValue>>;
}

somni_expr::for_all_tuples! {
    ($($arg:ident),*) => {
        impl<$($arg,)* R, F> NativeFunction<($($arg,)*)> for F
        where
            $($arg: LoadStore<VmTypeSet> + ValueType,)*
            F: Fn($($arg,)*) -> R,
            F: for<'t> Fn($($arg::Output<'t>,)*) -> R,
            F: Clone,
            R: ValueType + LoadStore<VmTypeSet>,
        {
            #[allow(non_snake_case)]
            fn call_from_vm(&self, ctx: &mut EvalContext<'_>, sp: MemoryAddress) -> Result<VmTypedValue, EvalEvent<TypedValue>> {
                let offset = 0;
                $(
                    let $arg = match ctx.memory.load_typed(sp + offset, <$arg>::TYPE) {
                        Ok(typed) => TypedValue::from(typed),
                        Err(e) => return Err(ctx.runtime_error(e)),
                    };
                    let offset = offset + <$arg>::TYPE.vm_size_of();
                )*

                Ok(VmTypedValue::from(self($(
                    <$arg>::load(&ctx.type_ctx, &$arg).expect("Expect to be able to load the specified type"),
                )*).store(ctx.type_context())))
            }
        }
    };
}

pub struct SomniFn<'p> {
    #[allow(clippy::type_complexity)]
    func: Box<
        dyn Fn(&mut EvalContext<'p>, MemoryAddress) -> Result<VmTypedValue, EvalEvent<TypedValue>>
            + 'p,
    >,

    #[allow(clippy::type_complexity)]
    expr_func:
        Box<dyn Fn(&mut VmTypeSet, &[TypedValue]) -> Result<TypedValue, FunctionCallError> + 'p>,
}

impl<'p> SomniFn<'p> {
    pub fn new<A, F>(func: F) -> Self
    where
        F: NativeFunction<A> + 'p,
    {
        let expr_func = func.clone();
        Self {
            func: Box::new(move |ctx, sp| func.call_from_vm(ctx, sp)),
            expr_func: Box::new(move |ctx, args| expr_func.call(ctx, args)),
        }
    }

    fn call_from_vm(
        &self,
        ctx: &mut EvalContext<'p>,
        sp: MemoryAddress,
    ) -> Result<VmTypedValue, EvalEvent<TypedValue>> {
        (self.func)(ctx, sp)
    }

    fn call_from_expr(
        &self,
        ctx: &mut dyn ExprContext<VmTypeSet>,
        args: &[TypedValue],
    ) -> Result<TypedValue, FunctionCallError> {
        (self.expr_func)(&mut *ctx.type_context(), args)
    }
}

impl<A> DynFunction<A, VmTypeSet> for &SomniFn<'_> {
    fn call(
        &self,
        ctx: &mut VmTypeSet,
        args: &[TypedValue],
    ) -> Result<TypedValue, FunctionCallError> {
        (self.expr_func)(ctx, args)
    }
}

pub struct EvalContext<'p> {
    source: &'p str,
    program: &'p codegen::Program,
    strings: &'p mut Strings,
    intrinsics: HashMap<StringIndex, SomniFn<'p>>,
    state: EvalState,
    type_ctx: VmTypeSet,

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

    fn load_typed(&self, local: MemoryAddress, return_type: Type) -> Result<VmTypedValue, String> {
        let data = self.load(local, return_type.vm_size_of())?;

        Ok(VmTypedValue::from_typed_bytes(return_type, data))
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
            Type::Int => inner!(u64),
            Type::MaybeSignedInt => inner!(u64),
            Type::SignedInt => inner!(i64),
            Type::Float => inner!(f64),
            Type::Bool => inner!(bool),
            Type::String => inner!(StringIndex),
            Type::Void => inner!(()),
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
        fn $op(
            ty: Type,
            ctx: &mut EvalContext<'_>,
            dst: MemoryAddress,
            lhs: MemoryAddress,
            rhs: MemoryAddress,
        ) -> Result<(), EvalEvent<TypedValue>> {
            dispatch_type!(ty, ($ty:ty) => {
                let lhs = ctx.load::<$ty>(lhs)?;
                let rhs = ctx.load::<$ty>(rhs)?;

                match <$ty>::$op(lhs, rhs) {
                    Ok(result) => ctx.store(dst, result),
                    Err(e) => Err(operator_error(ctx, e)),
                }
            })
        }
    }
);
for_each_unary_operator!(
    ($_name:path, $op:ident) => {
        fn $op(
            ty: Type,
            ctx: &mut EvalContext<'_>,
            dst: MemoryAddress,
            operand: MemoryAddress,
        ) -> Result<(), EvalEvent<TypedValue>> {
            dispatch_type!(ty, ($ty:ty) => {
                let operand = ctx.load::<$ty>(operand)?;

                match <$ty>::$op(operand) {
                    Ok(result) => ctx.store(dst, result),
                    Err(e) => Err(operator_error(ctx, e)),
                }
            })
        }
    }
);

#[cold]
fn operator_error(ctx: &EvalContext<'_>, error: OperatorError) -> EvalEvent<TypedValue> {
    ctx.runtime_error(format_args!("Failed to apply operator: {error:?}"))
}

impl<'p> EvalContext<'p> {
    pub fn string(&self, index: StringIndex) -> &str {
        self.strings.lookup(index)
    }

    fn load_function_by_name(&self, name: &str) -> Option<&'p codegen::Function> {
        let name_index = self.strings.find(name)?;
        self.program.functions.get(&name_index)
    }

    pub fn new(source: &'p str, strings: &'p mut Strings, program: &'p codegen::Program) -> Self {
        EvalContext {
            intrinsics: HashMap::new(),
            state: EvalState::Idle,
            type_ctx: program.type_ctx.clone(),
            program,
            memory: Memory::new(),
            source,
            strings,
            outer_function_name: StringIndex::dummy(), // Will be set when the first function is called
            program_counter: CodeAddress(0),
        }
    }

    pub fn add_function<A>(&mut self, name: &str, f: impl NativeFunction<A> + 'p) {
        let name = self
            .strings
            .find(name)
            .unwrap_or_else(|| self.strings.intern(name));
        self.intrinsics.insert(name, SomniFn::new(f));
    }

    pub fn reset(&mut self) {
        // TODO: only if eval_expression is supported
        self.state = EvalState::Idle;
        self.memory = Memory::new();
        self.memory.allocate(
            self.program
                .globals
                .values()
                .map(|v| v.ty().vm_size_of())
                .sum::<usize>()
                + 16, // SP + PC
        );
        let mut address = 0;
        for (_, def) in self.program.globals.iter() {
            self.store_typed(MemoryAddress::Global(address), def.value())
                .unwrap();
            address += def.ty().vm_size_of();
        }
    }

    /// Calls the `main` function and starts the evaluation. If the program is already running,
    /// it will continue executing the current function.
    ///
    /// If the function returns with [`EvalEvent::UnknownFunctionCall`], it means that the script
    /// tried to call a function that is not defined in the program. You can use
    /// [`Self::string`] to read the function name, and [`Self::unknown_call_args()`]
    /// to get the arguments of the function that was called. Set the return value with
    /// [`Self::set_return_value()`], then call [`Self::run`] to continue execution.
    pub fn run(&mut self) -> EvalEvent<TypedValue> {
        if matches!(self.state, EvalState::Idle) {
            // Restore VM state.
            self.reset();

            // Initialize the first frame with the main program
            self.call("main", &[])
        } else {
            self.execute()
        }
    }

    /// Calls a function by its name with the given arguments.
    ///
    /// If the function returns with [`EvalEvent::UnknownFunctionCall`], it means that the script
    /// tried to call a function that is not defined in the program. You can use
    /// [`Self::string`] to read the function name, and [`Self::unknown_call_args()`]
    /// to get the arguments of the function that was called. Set the return value with
    /// [`Self::set_return_value()`], then call [`Self::run`] to continue execution.
    pub fn call(&mut self, func: &str, args: &[TypedValue]) -> EvalEvent<TypedValue> {
        let Some(function) = self.load_function_by_name(func) else {
            if let Some(fn_name) = self.strings.find(func) {
                if let Some((name, intrinsic)) = self.intrinsics.remove_entry(&fn_name) {
                    let retval = intrinsic.call_from_expr(self, args);
                    self.intrinsics.insert(name, intrinsic);
                    return match retval {
                        Ok(result) => EvalEvent::Complete(result),
                        Err(FunctionCallError::IncorrectArgumentCount { expected }) => self
                            .runtime_error(format_args!(
                                "{func} takes {expected} arguments, {} given",
                                args.len()
                            )),
                        Err(FunctionCallError::IncorrectArgumentType { idx, expected }) => self
                            .runtime_error(format_args!(
                                "{func} expects argument {idx} to be {expected}, got {}",
                                args[idx].type_of()
                            )),
                        Err(FunctionCallError::FunctionNotFound) => {
                            self.runtime_error(format_args!("Function {func} is not found"))
                        }
                        Err(FunctionCallError::Other(error)) => {
                            self.runtime_error(format_args!("Failed to call {func}: {error}"))
                        }
                    };
                }
            }

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
            let arg = if let TypedValue::MaybeSignedInt(int) = *arg {
                match ty {
                    Type::Int => TypedValue::Int(int),
                    Type::SignedInt => TypedValue::SignedInt(int as i64),
                    _ => arg.clone(),
                }
            } else {
                arg.clone()
            };
            if *ty != arg.type_of() {
                return self.runtime_error(format!(
                    "Function '{func}' expects argument {} to be of type {ty}, but got {}",
                    i + 1,
                    arg.type_of()
                ));
            }
            if let Err(e) = self.store_typed(*addr, arg.into()) {
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
    pub fn eval_expression<V>(&mut self, expression: &str) -> Result<V::Output, EvalError>
    where
        V: LoadOwned<VmTypeSet>,
    {
        if !matches!(self.state, EvalState::Idle) {
            panic!("Cannot evaluate expression while the VM is running");
        }

        // TODO: we can allow new globals to be defined in the expression, but that would require
        // storing a copy of the original globals, so that they can be reset?

        let ast = parser::parse_expression(expression).map_err(|e| {
            EvalError(
                MarkInSource(
                    expression,
                    e.location,
                    "Failed to parse expression",
                    &e.error,
                )
                .to_string()
                .into_boxed_str(),
            )
        })?;

        let mut visitor = ExpressionVisitor {
            context: self,
            source: expression,
            _marker: PhantomData,
        };
        // TODO: handle errors
        let result = match visitor.visit_expression(&ast) {
            Ok(result) => result,
            Err(e) => {
                // Bad way to extract the error message, but we lost structure.
                let mut error = strip_ansi(e.message.lines().last().unwrap())
                    .trim_start_matches([' ', '^', '|'])
                    .to_string();

                self.backtrace(|name, pc| {
                    println!("{name}");
                    error = MarkInSource(
                        self.source,
                        self.program.debug_info.instruction_locations[pc],
                        &format!("while evaluating {name}"),
                        &error,
                    )
                    .to_string();
                });
                return Err(EvalError(
                    MarkInSource(
                        expression,
                        ast.location(),
                        e.message.lines().next().unwrap(),
                        &error.to_string(),
                    )
                    .to_string()
                    .into_boxed_str(),
                ));
            }
        };

        let result_ty = result.type_of();
        let result = V::load_owned(self.type_context(), &result).ok_or_else(|| {
            EvalError(
                MarkInSource(
                    expression,
                    ast.location(),
                    "Eval error",
                    &format!(
                        "Expression evaluates to {result_ty}, which cannot be converted to {}",
                        std::any::type_name::<V>()
                    ),
                )
                .to_string()
                .into_boxed_str(),
            )
        })?;

        Ok(result)
    }

    fn execute(&mut self) -> EvalEvent<TypedValue> {
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

    fn step(&mut self) -> Result<(), EvalEvent<TypedValue>> {
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
            Instruction::CallNamed(function_name, return_ty, sp) => {
                let Some((name, intrinsic)) = self.intrinsics.remove_entry(&function_name) else {
                    self.state = EvalState::WaitingForFunctionResult(function_name, return_ty, sp);
                    self.program_counter += 1;
                    return Err(EvalEvent::UnknownFunctionCall(function_name));
                };

                let first_arg = sp + return_ty.vm_size_of();
                let retval = intrinsic.call_from_vm(self, first_arg)?;
                self.intrinsics.insert(name, intrinsic);
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
                        .unwrap()
                        .into();
                    self.state = EvalState::Idle;

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
                dispatch_binary_operator!(operator, ($function:tt) => { $function(ty, self, dst, lhs, rhs)? })
            }
            Instruction::UnaryOperator {
                ty,
                operator,
                dst,
                op,
            } => {
                dispatch_unary_operator!(operator, ($function:tt) => { $function(ty, self, dst, op)? })
            }

            Instruction::AddressOf(dst, lhs) => {
                let value = self.memory.address(lhs) as u64;
                self.store_typed(dst, VmTypedValue::Int(value))?;
            }
            Instruction::Dereference(ty, dst, lhs) => {
                let address = self.load::<u64>(lhs)?;
                let address = MemoryAddress::Global(address as usize);

                self.copy(address, dst, ty.vm_size_of())?;
            }
            Instruction::Copy(dst, from, amount) => self.copy(from, dst, amount as usize)?,
            Instruction::DerefCopy(ty, addr, from) => {
                let dst = self.load::<u64>(addr)?;
                let dst = MemoryAddress::Global(dst as usize);

                self.copy(from, dst, ty.vm_size_of())?;
            }
            Instruction::LoadValue(addr, value) => self.store_typed(addr, value)?,
        }

        self.program_counter += 1;

        Ok(())
    }

    fn store<V: MemoryRepr>(
        &mut self,
        addr: MemoryAddress,
        value: V,
    ) -> Result<(), EvalEvent<TypedValue>> {
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

    fn store_typed(
        &mut self,
        addr: MemoryAddress,
        value: VmTypedValue,
    ) -> Result<(), EvalEvent<TypedValue>> {
        match self
            .memory
            .as_mut(addr..addr + value.type_of().vm_size_of())
        {
            Ok(memory) => {
                value.write(memory);
                Ok(())
            }
            Err(e) => {
                Err(self.runtime_error(format_args!("Failed to store value at address: {e}")))
            }
        }
    }

    fn load<V: MemoryRepr>(&self, addr: MemoryAddress) -> Result<V, EvalEvent<TypedValue>> {
        let bytes = self.memory.load(addr, V::BYTES).map_err(|e| {
            self.runtime_error(format_args!("Failed to load value from address: {e}"))
        })?;
        Ok(V::from_bytes(bytes))
    }

    fn copy(
        &mut self,
        addr: MemoryAddress,
        dst: MemoryAddress,
        amount: usize,
    ) -> Result<(), EvalEvent<TypedValue>> {
        self.memory
            .copy(addr, dst, amount)
            .map_err(|e| self.runtime_error(format_args!("Failed to copy value: {e}")))
    }

    pub fn unknown_call_args<A: Arguments>(&self) -> Option<A> {
        if let EvalState::WaitingForFunctionResult(_name, ty, sp) = self.state {
            let first_arg = sp + ty.vm_size_of();
            Some(A::read(self, first_arg))
        } else {
            None
        }
    }

    pub fn set_return_value(&mut self, value: TypedValue) -> Result<(), EvalEvent<TypedValue>> {
        let EvalState::WaitingForFunctionResult(_, ty, sp) = self.state else {
            return Err(self.runtime_error("No function is currently waiting for a result"));
        };

        let value = VmTypedValue::from(value);

        if value.type_of() != ty {
            return Err(self.runtime_error(format_args!("Expcted a {ty} as the return value")));
        }

        self.store_typed(sp, value)?;
        self.state = EvalState::Running;
        Ok(())
    }

    fn backtrace(&self, mut f: impl FnMut(&str, usize)) {
        let mut fns = self
            .program
            .functions
            .values()
            .map(|f| (f.entry_point.0, f.name))
            .collect::<Vec<_>>();

        fns.sort_by_key(|(e, _f)| *e);

        let mut sp = self.memory.sp;
        let mut pc = self.program_counter.0;
        while sp != 0 {
            let mut name = None;
            for (entry_point, func) in fns.iter().copied() {
                if entry_point > pc {
                    break;
                }
                name = Some(func);
            }

            f(name.map(|idx| self.string(idx)).unwrap_or("<unknown>"), pc);

            let next_pc = self.load::<u64>(MemoryAddress::Global(sp - 16)).unwrap();
            let next_sp = self.load::<u64>(MemoryAddress::Global(sp - 8)).unwrap();
            sp = next_sp as usize;
            pc = next_pc as usize;
        }
    }

    pub fn print_backtrace(&self) {
        let mut n = 0;
        self.backtrace(move |function, _| {
            n += 1;
            println!("Frame {n}: {function}");
        });
    }

    fn call_function(
        &mut self,
        function: &'p Function,
        sp: MemoryAddress,
    ) -> Result<(), EvalEvent<TypedValue>> {
        self.store(sp - 16, self.program_counter.0 as u64)?;
        self.store(sp - 8, self.memory.sp as u64)?;
        self.memory.sp = self.memory.address(sp);
        // Allocate memory for the function's local variables.
        self.memory.allocate(function.stack_size);
        self.program_counter = function.entry_point;

        Ok(())
    }

    #[cold]
    fn runtime_error(&self, message: impl ToString) -> EvalEvent<TypedValue> {
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

impl<'s> ExprContext<VmTypeSet> for EvalContext<'s> {
    fn type_context(&mut self) -> &mut VmTypeSet {
        &mut self.type_ctx
    }

    fn declare(&mut self, _variable: &str, _value: TypedValue) {
        unimplemented!()
    }

    fn assign_variable(&mut self, _variable: &str, _value: &TypedValue) -> Result<(), Box<str>> {
        unimplemented!()
    }

    fn assign_address(
        &mut self,
        _address: TypedValue,
        _value: &TypedValue,
    ) -> Result<(), Box<str>> {
        unimplemented!()
    }

    fn at_address(&mut self, _address: TypedValue) -> Result<TypedValue, Box<str>> {
        unimplemented!()
    }

    fn open_scope(&mut self) {
        unimplemented!()
    }

    fn close_scope(&mut self) {
        unimplemented!()
    }

    fn try_load_variable(&mut self, name: &str) -> Option<TypedValue> {
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

        Some(value.into())
    }

    fn call_function(
        &mut self,
        function_name: &str,
        args: &[TypedValue],
    ) -> Result<TypedValue, FunctionCallError> {
        match self.call(function_name, args) {
            EvalEvent::UnknownFunctionCall(_fn_name) => Err(FunctionCallError::FunctionNotFound),
            EvalEvent::Error(e) => Err(FunctionCallError::Other(
                e.mark(self, "Runtime error").to_string().into_boxed_str(),
            )),
            EvalEvent::Complete(value) => Ok(value),
        }
    }

    fn address_of(&mut self, name: &str) -> TypedValue {
        let name_idx = self
            .strings
            .find(name)
            .unwrap_or_else(|| panic!("Variable '{name}' not found in program"));
        let addr = self
            .program
            .globals
            .get_index_of(&name_idx)
            .unwrap_or_else(|| panic!("Variable '{name}' not found in program"));
        TypedValue::Int(addr as u64)
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn test_vm() {
        crate::test::run_eval_tests("tests/eval/");
    }
}
