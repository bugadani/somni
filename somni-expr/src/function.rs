use crate::{
    for_all_tuples,
    value::{Load, Store},
    Type, TypeSet, TypedValue,
};

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
{
    const ARG_COUNT: usize;

    fn call(&self, ctx: &mut T, args: &[TypedValue<T>])
        -> Result<TypedValue<T>, FunctionCallError>;
}

macro_rules! substitute {
    ($arg:tt, $replacement:tt) => {
        $replacement
    };
}

for_all_tuples! {
    ($($arg:ident),*) => {
        impl<$($arg,)* R, F, T> DynFunction<($($arg,)*), T> for F
        where
            $($arg: Load<T>,)*
            // This double bound on F ensures that we can work with reference types (&str), too:
            // The first one ensures type-inference matches the Load implementation we want
            // The second one ensures we don't run into "... is not generic enough" errors.
            F: Fn($($arg,)*) -> R,
            F: for<'t> Fn($($arg::Output<'t>,)*) -> R,
            R: Store<T>,
            T: TypeSet,
        {
            const ARG_COUNT: usize = 0 $( + substitute!($arg, 1) )* ;

            #[allow(non_snake_case, unused)]
            fn call(&self, ctx: &mut T, args: &[TypedValue<T>]) -> Result<TypedValue<T>, FunctionCallError> {
                if args.len() != Self::ARG_COUNT {
                    return Err(FunctionCallError::IncorrectArgumentCount { expected: Self::ARG_COUNT });
                }

                let idx = 0;
                $(
                    let $arg = <$arg>::load(ctx, &args[idx]).ok_or_else(|| {
                        FunctionCallError::IncorrectArgumentType { idx, expected: $arg::TYPE }
                    })?;
                    let idx = idx + 1;
                )*


                Ok(self($($arg),*).store(ctx))
            }
        }
    };
}

pub(crate) struct ExprFn<'ctx, T>
where
    T: TypeSet,
{
    #[allow(clippy::type_complexity)]
    func: Box<dyn Fn(&mut T, &[TypedValue<T>]) -> Result<TypedValue<T>, FunctionCallError> + 'ctx>,
}

impl<'ctx, T> ExprFn<'ctx, T>
where
    T: TypeSet,
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
        ctx: &mut T,
        args: &[TypedValue<T>],
    ) -> Result<TypedValue<T>, FunctionCallError> {
        (self.func)(ctx, args)
    }
}
