use somni_parser::{
    ast::{
        Body, Expression, Function, If, LiteralValue, Loop, Statement, TypeHint, VariableDefinition,
    },
    lexer,
    parser::DefaultTypeSet,
    Location,
};

use crate::{
    value::LoadStore, EvalError, ExprContext, FunctionCallError, Type, TypeSet, TypedValue,
};

/// A visitor that can process an abstract syntax tree.
pub struct ExpressionVisitor<'a, C, T = DefaultTypeSet> {
    /// The context in which the expression is evaluated.
    pub context: &'a mut C,
    /// The source code from which the expression was parsed.
    pub source: &'a str,
    /// The types of the variables in the context.
    pub _marker: std::marker::PhantomData<T>,
}

impl<'a, C, T> ExpressionVisitor<'a, C, T>
where
    C: ExprContext<T>,
    T: TypeSet,
{
    fn visit_variable(&mut self, variable: &lexer::Token) -> Result<TypedValue<T>, EvalError> {
        let name = variable.source(self.source);
        self.context.try_load_variable(name).ok_or(EvalError {
            message: format!("Variable {name} was not found").into_boxed_str(),
            location: variable.location,
        })
    }

    /// Visits an expression and evaluates it, returning the result as a `TypedValue`.
    pub fn visit_expression(
        &mut self,
        expression: &Expression<T::Parser>,
    ) -> Result<TypedValue<T>, EvalError> {
        let result = match expression {
            Expression::Variable { variable } => self.visit_variable(variable)?,
            Expression::Literal { value } => match &value.value {
                LiteralValue::Integer(value) => TypedValue::<T>::MaybeSignedInt(*value),
                LiteralValue::Float(value) => TypedValue::<T>::Float(*value),
                LiteralValue::String(value) => value.store(self.context.type_context()),
                LiteralValue::Boolean(value) => TypedValue::<T>::Bool(*value),
            },
            Expression::UnaryOperator { name, operand } => match name.source(self.source) {
                "!" => {
                    let operand = self.visit_expression(operand)?;

                    match TypedValue::<T>::not(self.context.type_context(), operand) {
                        Ok(r) => r,
                        Err(error) => {
                            return Err(EvalError {
                                message: format!("Failed to evaluate expression: {error}")
                                    .into_boxed_str(),
                                location: expression.location(),
                            });
                        }
                    }
                }

                "-" => {
                    let value = self.visit_expression(operand)?;
                    let ty = value.type_of();
                    TypedValue::<T>::negate(self.context.type_context(), value).map_err(|e| {
                        EvalError {
                            message: format!("Cannot negate {ty}: {e}").into_boxed_str(),
                            location: operand.location(),
                        }
                    })?
                }
                "&" => match operand.as_variable() {
                    Some(variable) => {
                        let name = variable.source(self.source);
                        self.context.address_of(name)
                    }
                    None => {
                        return Err(EvalError {
                            message: String::from("Cannot take address of non-variable expression")
                                .into_boxed_str(),
                            location: operand.location(),
                        });
                    }
                },
                "*" => {
                    let address = self.visit_expression(operand)?;
                    self.context.at_address(address).map_err(|e| EvalError {
                        message: format!("Failed to load variable from address: {e}")
                            .into_boxed_str(),
                        location: operand.location(),
                    })?
                }
                _ => {
                    return Err(EvalError {
                        message: format!("Unknown unary operator: {}", name.source(self.source))
                            .into_boxed_str(),
                        location: expression.location(),
                    });
                }
            },
            Expression::BinaryOperator { name, operands } => {
                let short_circuiting = ["&&", "||"];
                let operator = name.source(self.source);

                // Special cases
                if operator == "=" {
                    return match &operands[0] {
                        Expression::UnaryOperator { name, operand }
                            if name.source(self.source) == "*" =>
                        {
                            let address = self.visit_expression(operand)?;
                            let rhs = self.visit_expression(&operands[1])?;
                            self.context
                                .assign_address(address, &rhs)
                                .map_err(|e| EvalError {
                                    message: e,
                                    location: expression.location(),
                                })?;
                            Ok(TypedValue::Void)
                        }
                        Expression::Variable { variable } => {
                            let name = variable.source(self.source);
                            let rhs = self.visit_expression(&operands[1])?;
                            self.context
                                .assign_variable(name, &rhs)
                                .map_err(|e| EvalError {
                                    message: e,
                                    location: expression.location(),
                                })?;
                            Ok(TypedValue::Void)
                        }
                        _ => Err(EvalError {
                            location: operands[0].location(),
                            message: "Not a valid left-hand side of an assignment".into(),
                        }),
                    };
                }
                if short_circuiting.contains(&operator) {
                    let lhs = self.visit_expression(&operands[0])?;
                    return match operator {
                        "&&" if lhs == TypedValue::<T>::Bool(false) => Ok(TypedValue::Bool(false)),
                        "||" if lhs == TypedValue::<T>::Bool(true) => Ok(TypedValue::Bool(true)),
                        _ => self.visit_expression(&operands[1]),
                    };
                }

                // "Normal" binary operators
                let lhs = self.visit_expression(&operands[0])?;
                let rhs = self.visit_expression(&operands[1])?;
                let type_context = self.context.type_context();
                let result = match operator {
                    "+" => TypedValue::<T>::add(type_context, lhs, rhs),
                    "-" => TypedValue::<T>::subtract(type_context, lhs, rhs),
                    "*" => TypedValue::<T>::multiply(type_context, lhs, rhs),
                    "/" => TypedValue::<T>::divide(type_context, lhs, rhs),
                    "<" => TypedValue::<T>::less_than(type_context, lhs, rhs),
                    ">" => TypedValue::<T>::less_than(type_context, rhs, lhs),
                    "<=" => TypedValue::<T>::less_than_or_equal(type_context, lhs, rhs),
                    ">=" => TypedValue::<T>::less_than_or_equal(type_context, rhs, lhs),
                    "==" => TypedValue::<T>::equals(type_context, lhs, rhs),
                    "!=" => TypedValue::<T>::not_equals(type_context, lhs, rhs),
                    "|" => TypedValue::<T>::bitwise_or(type_context, lhs, rhs),
                    "^" => TypedValue::<T>::bitwise_xor(type_context, lhs, rhs),
                    "&" => TypedValue::<T>::bitwise_and(type_context, lhs, rhs),
                    "<<" => TypedValue::<T>::shift_left(type_context, lhs, rhs),
                    ">>" => TypedValue::<T>::shift_right(type_context, lhs, rhs),

                    other => {
                        return Err(EvalError {
                            message: format!("Unknown binary operator: {other}").into_boxed_str(),
                            location: expression.location(),
                        });
                    }
                };

                match result {
                    Ok(r) => r,
                    Err(error) => {
                        return Err(EvalError {
                            message: format!("Failed to evaluate expression: {error}")
                                .into_boxed_str(),
                            location: expression.location(),
                        });
                    }
                }
            }
            Expression::FunctionCall { name, arguments } => {
                let function_name = name.source(self.source);
                let mut args = Vec::with_capacity(arguments.len());
                for arg in arguments {
                    args.push(self.visit_expression(arg)?);
                }

                match self.context.call_function(function_name, &args) {
                    Ok(result) => result,
                    Err(FunctionCallError::IncorrectArgumentCount { expected }) => {
                        return Err(EvalError {
                            message: format!(
                                "{function_name} takes {expected} arguments, {} given",
                                args.len()
                            )
                            .into_boxed_str(),
                            location: expression.location(),
                        });
                    }
                    Err(FunctionCallError::IncorrectArgumentType { idx, expected }) => {
                        return Err(EvalError {
                            message: format!(
                                "{function_name} expects argument {idx} to be {expected}, got {}",
                                args[idx].type_of()
                            )
                            .into_boxed_str(),
                            location: arguments[idx].location(),
                        });
                    }
                    Err(FunctionCallError::FunctionNotFound) => {
                        return Err(EvalError {
                            message: format!("Function {function_name} is not found")
                                .into_boxed_str(),
                            location: expression.location(),
                        });
                    }
                    Err(FunctionCallError::Other(error)) => {
                        return Err(EvalError {
                            message: format!("Failed to call {function_name}: {error}")
                                .into_boxed_str(),
                            location: expression.location(),
                        });
                    }
                }
            }
        };

        Ok(result)
    }

    fn typecheck_with_hint(
        &self,
        value: TypedValue<T>,
        hint: Option<TypeHint>,
    ) -> Result<TypedValue<T>, EvalError> {
        let Some(hint) = hint else {
            // No hint
            return Ok(value);
        };

        let ty =
            Type::from_name(hint.type_name.source(self.source)).map_err(|message| EvalError {
                message,
                location: hint.type_name.location,
            })?;

        self.typecheck(value, ty, hint.type_name.location)
    }

    fn typecheck(
        &self,
        value: TypedValue<T>,
        hint: Type,
        location: Location,
    ) -> Result<TypedValue<T>, EvalError> {
        match (value, hint) {
            (value, hint) if value.type_of() == hint => Ok(value),
            (TypedValue::MaybeSignedInt(val), Type::Int) => Ok(TypedValue::Int(val)),
            (TypedValue::MaybeSignedInt(val), Type::SignedInt) => Ok(TypedValue::<T>::SignedInt(
                T::to_signed(val).map_err(|_| EvalError {
                    message: format!("Failed to cast {val:?} to signed int").into_boxed_str(),
                    location,
                })?,
            )),
            (value, hint) => Err(EvalError {
                message: format!("Expected {hint}, got {}", value.type_of()).into_boxed_str(),
                location,
            }),
        }
    }

    /// Evaluates a function with the given arguments.
    pub fn visit_function(
        &mut self,
        function: &Function<T::Parser>,
        args: &[TypedValue<T>],
    ) -> Result<TypedValue<T>, EvalError> {
        for (arg, arg_value) in function.arguments.iter().zip(args.iter()) {
            let arg_name = arg.name.source(self.source);

            let arg_value = self.typecheck_with_hint(arg_value.clone(), Some(arg.arg_type))?;

            self.context.declare(arg_name, arg_value);
        }

        let retval = match self.visit_body(&function.body)? {
            StatementResult::Return(typed_value) => typed_value,
            StatementResult::EndOfBody => TypedValue::Void,
            StatementResult::LoopBreak | StatementResult::LoopContinue => todo!(),
        };

        let retval =
            self.typecheck_with_hint(retval, function.return_decl.as_ref().map(|d| d.return_type))?;

        Ok(retval)
    }

    fn visit_body(&mut self, body: &Body<T::Parser>) -> Result<StatementResult<T>, EvalError> {
        self.context.open_scope();
        for statement in body.statements.iter() {
            if let Some(retval) = self.visit_statement(statement)? {
                self.context.close_scope();
                return Ok(retval);
            }
        }

        self.context.close_scope();
        Ok(StatementResult::EndOfBody)
    }

    fn visit_statement(
        &mut self,
        statement: &Statement<T::Parser>,
    ) -> Result<Option<StatementResult<T>>, EvalError> {
        match statement {
            Statement::Return(return_with_value) => {
                return self
                    .visit_expression(&return_with_value.expression)
                    .map(StatementResult::Return)
                    .map(Some);
            }
            Statement::EmptyReturn(_) => {
                return Ok(Some(StatementResult::Return(TypedValue::Void)));
            }
            Statement::If(if_statement) => return self.visit_if(if_statement),
            Statement::Loop(loop_statement) => return self.visit_loop(loop_statement),
            Statement::Break(_) => return Ok(Some(StatementResult::LoopBreak)),
            Statement::Continue(_) => return Ok(Some(StatementResult::LoopContinue)),
            Statement::VariableDefinition(variable_definition) => {
                self.visit_declaration(variable_definition)?;
            }
            Statement::Expression { expression, .. } => {
                self.visit_expression(expression)?;
            }
        }

        Ok(None)
    }

    fn visit_declaration(&mut self, decl: &VariableDefinition<T::Parser>) -> Result<(), EvalError> {
        let name = decl.identifier.source(self.source);
        let value = self.visit_expression(&decl.initializer)?;

        let value = self.typecheck_with_hint(value, decl.type_token)?;

        self.context.declare(name, value);

        Ok(())
    }

    fn visit_if(
        &mut self,
        if_statement: &If<T::Parser>,
    ) -> Result<Option<StatementResult<T>>, EvalError> {
        let condition = self.visit_expression(&if_statement.condition)?;

        let condition = self.typecheck(condition, Type::Bool, if_statement.condition.location())?;

        let body = if condition == TypedValue::Bool(true) {
            &if_statement.body
        } else if let Some(ref else_branch) = if_statement.else_branch {
            &else_branch.else_body
        } else {
            // Condition is false, but there is no `else`
            return Ok(None);
        };

        let retval = match self.visit_body(body)? {
            StatementResult::EndOfBody => None,
            other => Some(other),
        };
        Ok(retval)
    }

    fn visit_loop(
        &mut self,
        loop_statement: &Loop<T::Parser>,
    ) -> Result<Option<StatementResult<T>>, EvalError> {
        loop {
            match self.visit_body(&loop_statement.body)? {
                ret @ StatementResult::Return(_) => return Ok(Some(ret)),
                StatementResult::LoopBreak => return Ok(None),
                StatementResult::LoopContinue | StatementResult::EndOfBody => {}
            }
        }
    }
}

enum StatementResult<T: TypeSet> {
    Return(TypedValue<T>),
    LoopBreak,
    LoopContinue,
    EndOfBody,
}
