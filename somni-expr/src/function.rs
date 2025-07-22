use crate::{for_all_tuples, value::ValueType, ExprContext, Type, TypeSet, TypedValue};

/// An error that occurs when calling a function.
pub enum FunctionCallError {
    /// The function was not found in the context.
    FunctionNotFound,

    /// The number of arguments passed to the function does not match the expected count.
    IncorrectArgumentCount {
        /// The expected number of arguments.
        expected: usize,
    },

    /// The type of an argument does not match the expected type.
    IncorrectArgumentType {
        /// The 0-based index of the argument that has the incorrect type.
        idx: usize,
        /// The expected type of the argument.
        expected: Type,
    },

    /// An error occurred while calling the function.
    Other(&'static str),
}

#[doc(hidden)]
pub trait DynFunction<A, T>
where
    T: TypeSet,
    T::Integer: ValueType,
    T::Float: ValueType,
{
    fn call(
        &self,
        ctx: &mut dyn ExprContext<T>,
        args: &[TypedValue<T>],
    ) -> Result<TypedValue<T>, FunctionCallError>;
}

macro_rules! ignore {
    ($arg:tt) => {};
}

for_all_tuples! {
    ($($arg:ident),*) => {
        impl<$($arg,)* R, F, T> DynFunction<($($arg,)*), T> for F
        where
            $($arg: ValueType + TryFrom<TypedValue<T>>,)*
            F: Fn($($arg,)*) -> R,
            R: ValueType + Into<TypedValue<T>>,
            T: TypeSet,
            T::Integer: ValueType,
            T::Float: ValueType,
        {
            #[allow(non_snake_case, unused)]
            fn call(&self, _ctx: &mut dyn ExprContext<T>, args: &[TypedValue<T>]) -> Result<TypedValue<T>, FunctionCallError> {

                // TODO: while it's great that we can now allow access to the context, it's a bit of a pain to use.
                // The ideal API would load strings in this function, and store the string when returning from the user's function.

                let arg_count = 0;
                $(
                    ignore!($arg);
                    let arg_count = arg_count + 1;
                )*

                let idx = 0;
                let mut args = args.iter().copied();
                $(
                    let Some(arg) = args.next() else {
                        return Err(FunctionCallError::IncorrectArgumentCount { expected: arg_count });
                    };
                    let $arg = match <$arg>::try_from(arg) {
                        Ok(arg) => arg,
                        Err(_) => return Err(FunctionCallError::IncorrectArgumentType { idx, expected: $arg::TYPE }),
                    };
                    let idx = idx + 1;
                )*

                if args.next().is_some() {
                    return Err(FunctionCallError::IncorrectArgumentCount { expected: arg_count });
                }

                Ok(self($($arg),*).into())
            }
        }
    };
}

pub(crate) struct ExprFn<'ctx, T>
where
    T: TypeSet,
    T::Integer: ValueType,
    T::Float: ValueType,
{
    #[allow(clippy::type_complexity)]
    func: Box<
        dyn Fn(
                &mut dyn ExprContext<T>,
                &[TypedValue<T>],
            ) -> Result<TypedValue<T>, FunctionCallError>
            + 'ctx,
    >,
}

impl<'ctx, T> ExprFn<'ctx, T>
where
    T: TypeSet,
    T::Integer: ValueType,
    T::Float: ValueType,
{
    pub fn new<A, F>(func: F) -> Self
    where
        F: DynFunction<A, T> + 'ctx,
    {
        Self {
            func: Box::new(move |ctx, args| func.call(ctx, args)),
        }
    }

    pub fn call(
        &self,
        ctx: &mut dyn ExprContext<T>,
        args: &[TypedValue<T>],
    ) -> Result<TypedValue<T>, FunctionCallError> {
        (self.func)(ctx, args)
    }
}
