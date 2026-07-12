use indexmap::IndexMap;
use somni_parser::{
    Location,
    ast::{
        Body, Expression, For, Function, If, LeftHandExpression, LiteralValue, Loop,
        RightHandExpression, Statement, TypeHint, VariableDefinition,
    },
    lexer,
    parser::DefaultTypeSet,
};

use crate::{
    EvalError, ExprContext, FunctionCallError, RefPointee, Type, TypeSet, TypedValue,
    value::{LoadStore, Place, Reference, SomniStruct},
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
            Expression::Expression { expression } => {
                self.visit_right_hand_expression(expression)?
            }
            Expression::Assignment {
                left_expr,
                operator: _,
                right_expr,
            } => {
                let rhs = self.visit_right_hand_expression(right_expr)?;
                let assign_result = match left_expr {
                    LeftHandExpression::Name { variable } => {
                        let name = variable.source(self.source);
                        self.context.assign_variable(name, &rhs)
                    }
                    LeftHandExpression::Deref { .. } | LeftHandExpression::Field { .. } => {
                        let place = self.resolve_place_lhs(left_expr)?;
                        self.context.store_place(&place, &rhs)
                    }
                };

                if let Err(error) = assign_result {
                    return Err(EvalError {
                        message: error,
                        location: expression.location(),
                    });
                }

                TypedValue::Void
            }
        };

        Ok(result)
    }

    /// Visits an expression and evaluates it, returning the result as a `TypedValue`.
    pub fn visit_right_hand_expression(
        &mut self,
        expression: &RightHandExpression<T::Parser>,
    ) -> Result<TypedValue<T>, EvalError> {
        let result = match expression {
            RightHandExpression::Variable { variable } => self.visit_variable(variable)?,
            RightHandExpression::Literal { value } => match &value.value {
                LiteralValue::Integer(value) => TypedValue::<T>::MaybeSignedInt(*value),
                LiteralValue::Float(value) => TypedValue::<T>::Float(*value),
                LiteralValue::String(value) => value.store(self.context.type_context()),
                LiteralValue::Boolean(value) => TypedValue::<T>::Bool(*value),
            },
            RightHandExpression::UnaryOperator { name, operand } => {
                match name.source(self.source) {
                    "!" => {
                        let operand = self.visit_right_hand_expression(operand)?;

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
                        let value = self.visit_right_hand_expression(operand)?;
                        let ty = value.type_of();
                        TypedValue::<T>::negate(self.context.type_context(), value).map_err(
                            |e| EvalError {
                                message: format!("Cannot negate {ty}: {e}").into_boxed_str(),
                                location: operand.location(),
                            },
                        )?
                    }

                    "&" => {
                        let place = self.resolve_place_rhs(operand)?;
                        let value = self.context.load_place(&place).map_err(|e| EvalError {
                            message: e,
                            location: operand.location(),
                        })?;
                        let pointee =
                            RefPointee::from_type(value.type_of()).ok_or_else(|| EvalError {
                                message: String::from("Cannot take a reference to a reference")
                                    .into_boxed_str(),
                                location: operand.location(),
                            })?;
                        TypedValue::Ref(Reference { pointee, place })
                    }
                    "*" => {
                        let value = self.visit_right_hand_expression(operand)?;
                        let TypedValue::Ref(reference) = value else {
                            return Err(EvalError {
                                message: format!("Cannot dereference {}", value.type_of())
                                    .into_boxed_str(),
                                location: operand.location(),
                            });
                        };
                        self.context
                            .load_place(&reference.place)
                            .map_err(|e| EvalError {
                                message: format!("Failed to load variable from address: {e}")
                                    .into_boxed_str(),
                                location: operand.location(),
                            })?
                    }
                    _ => {
                        return Err(EvalError {
                            message: format!(
                                "Unknown unary operator: {}",
                                name.source(self.source)
                            )
                            .into_boxed_str(),
                            location: expression.location(),
                        });
                    }
                }
            }
            RightHandExpression::BinaryOperator { name, operands } => {
                let lhs = self.visit_right_hand_expression(&operands[0])?;

                let short_circuiting = ["&&", "||"];
                let operator = name.source(self.source);

                // Special cases
                if short_circuiting.contains(&operator) {
                    return match operator {
                        "&&" if lhs == TypedValue::<T>::Bool(false) => Ok(TypedValue::Bool(false)),
                        "||" if lhs == TypedValue::<T>::Bool(true) => Ok(TypedValue::Bool(true)),
                        _ => self.visit_right_hand_expression(&operands[1]),
                    };
                }

                // "Normal" binary operators
                let rhs = self.visit_right_hand_expression(&operands[1])?;

                // Structs and references support only structural equality. Handle
                // it here since the scalar operator dispatch cannot.
                if matches!(operator, "==" | "!=") {
                    let is_aggregate =
                        |v: &TypedValue<T>| matches!(v, TypedValue::Struct(_) | TypedValue::Ref(_));
                    if is_aggregate(&lhs) || is_aggregate(&rhs) {
                        let equal = lhs == rhs;
                        return Ok(TypedValue::Bool(if operator == "==" {
                            equal
                        } else {
                            !equal
                        }));
                    }
                }

                let type_context = self.context.type_context();
                let result = match operator {
                    "+" => TypedValue::<T>::add(type_context, lhs, rhs),
                    "-" => TypedValue::<T>::subtract(type_context, lhs, rhs),
                    "*" => TypedValue::<T>::multiply(type_context, lhs, rhs),
                    "/" => TypedValue::<T>::divide(type_context, lhs, rhs),
                    "%" => TypedValue::<T>::modulo(type_context, lhs, rhs),
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
            RightHandExpression::FunctionCall { name, arguments } => {
                let function_name = name.source(self.source);
                let mut args = Vec::with_capacity(arguments.len());
                for arg in arguments {
                    args.push(self.visit_right_hand_expression(arg)?);
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
            RightHandExpression::FieldAccess { base, field, .. } => {
                // Reading a field: evaluate the base, auto-dereferencing through a
                // reference, then clone the named field out of the struct.
                let base_value = self.visit_right_hand_expression(base)?;
                let base_value = match base_value {
                    TypedValue::Ref(reference) => self
                        .context
                        .load_place(&reference.place)
                        .map_err(|e| EvalError {
                            message: e,
                            location: base.location(),
                        })?,
                    other => other,
                };

                let field_name = field.source(self.source);
                let TypedValue::Struct(structure) = base_value else {
                    return Err(EvalError {
                        message: format!(
                            "Cannot access field `{field_name}` of {}",
                            base_value.type_of()
                        )
                        .into_boxed_str(),
                        location: base.location(),
                    });
                };

                match structure.fields().get(field_name) {
                    Some(value) => value.clone(),
                    None => {
                        return Err(EvalError {
                            message: format!(
                                "Struct `{}` has no field `{field_name}`",
                                structure.name()
                            )
                            .into_boxed_str(),
                            location: field.location,
                        });
                    }
                }
            }
            RightHandExpression::StructLiteral { name, fields, .. } => {
                self.visit_struct_literal(name, fields, expression.location())?
            }
        };

        Ok(result)
    }

    /// Constructs a struct value from a struct literal, validating field names
    /// against the definition and coercing each field to its declared type.
    fn visit_struct_literal(
        &mut self,
        name: &lexer::Token,
        fields: &[somni_parser::ast::StructLiteralField<T::Parser>],
        location: Location,
    ) -> Result<TypedValue<T>, EvalError> {
        let struct_name = name.source(self.source);
        let schema = self
            .context
            .struct_fields(struct_name)
            .ok_or_else(|| EvalError {
                message: format!("Unknown struct `{struct_name}`").into_boxed_str(),
                location: name.location,
            })?;

        // Evaluate each provided field, rejecting unknown or duplicate fields.
        let mut provided: IndexMap<Box<str>, (TypedValue<T>, Location)> = IndexMap::new();
        for field in fields {
            let field_name = field.name.source(self.source);
            if !schema.iter().any(|(n, _)| &**n == field_name) {
                return Err(EvalError {
                    message: format!("Struct `{struct_name}` has no field `{field_name}`")
                        .into_boxed_str(),
                    location: field.name.location,
                });
            }
            let value = self.visit_right_hand_expression(&field.value)?;
            if provided
                .insert(Box::from(field_name), (value, field.name.location))
                .is_some()
            {
                return Err(EvalError {
                    message: format!("Duplicate field `{field_name}` in struct `{struct_name}`")
                        .into_boxed_str(),
                    location: field.name.location,
                });
            }
        }

        // Build the fields in declaration order, coercing to each declared type.
        let mut out = IndexMap::with_capacity(schema.len());
        for (field_name, field_type) in &schema {
            let (value, field_location) =
                provided.swap_remove(field_name).ok_or_else(|| EvalError {
                    message: format!("Missing field `{field_name}` in struct `{struct_name}`")
                        .into_boxed_str(),
                    location,
                })?;
            let coerced = self.typecheck_named(value, field_type, false, field_location)?;
            out.insert(field_name.clone(), coerced);
        }

        Ok(TypedValue::Struct(SomniStruct::new(
            Box::from(struct_name),
            out,
        )))
    }

    /// Resolves the [`Place`] denoted by a right-hand expression used as the
    /// operand of `&`. Supports variables, field accesses (auto-dereferencing
    /// through references), and explicit dereferences.
    fn resolve_place_rhs(
        &mut self,
        expr: &RightHandExpression<T::Parser>,
    ) -> Result<Place, EvalError> {
        match expr {
            RightHandExpression::Variable { variable } => {
                let name = variable.source(self.source);
                self.context
                    .place_of_variable(name)
                    .map_err(|message| EvalError {
                        message,
                        location: variable.location,
                    })
            }
            RightHandExpression::UnaryOperator { name, operand }
                if name.source(self.source) == "*" =>
            {
                let value = self.visit_right_hand_expression(operand)?;
                match value {
                    TypedValue::Ref(reference) => Ok(reference.place),
                    other => Err(EvalError {
                        message: format!("Cannot dereference {}", other.type_of()).into_boxed_str(),
                        location: operand.location(),
                    }),
                }
            }
            RightHandExpression::FieldAccess { base, field, .. } => {
                let base_place = self.resolve_place_rhs(base)?;
                let container = self.container_place(base_place, base.location())?;
                self.append_field(container, field)
            }
            other => Err(EvalError {
                message: String::from("Cannot take the address of this expression")
                    .into_boxed_str(),
                location: other.location(),
            }),
        }
    }

    /// Resolves the [`Place`] denoted by a left-hand (assignment) expression.
    fn resolve_place_lhs(&mut self, lhs: &LeftHandExpression) -> Result<Place, EvalError> {
        match lhs {
            LeftHandExpression::Name { variable } => {
                let name = variable.source(self.source);
                self.context
                    .place_of_variable(name)
                    .map_err(|message| EvalError {
                        message,
                        location: variable.location,
                    })
            }
            LeftHandExpression::Deref { name, .. } => {
                let value = self.visit_variable(name)?;
                match value {
                    TypedValue::Ref(reference) => Ok(reference.place),
                    other => Err(EvalError {
                        message: format!("Cannot dereference {}", other.type_of()).into_boxed_str(),
                        location: name.location,
                    }),
                }
            }
            LeftHandExpression::Field { base, field, .. } => {
                let base_place = self.resolve_place_lhs(base)?;
                let container = self.container_place(base_place, base.location())?;
                self.append_field(container, field)
            }
        }
    }

    /// Given the place of a field-access base, returns the place of the aggregate
    /// to index into: if the base holds a reference, its target (auto-deref);
    /// otherwise the base place itself.
    fn container_place(
        &mut self,
        base_place: Place,
        base_location: Location,
    ) -> Result<Place, EvalError> {
        let base_value = self
            .context
            .load_place(&base_place)
            .map_err(|message| EvalError {
                message,
                location: base_location,
            })?;
        Ok(match base_value {
            TypedValue::Ref(reference) => reference.place,
            _ => base_place,
        })
    }

    /// Appends a field to a container place, validating that the container is a
    /// struct that has the named field.
    fn append_field(&mut self, container: Place, field: &lexer::Token) -> Result<Place, EvalError> {
        let container_value = self
            .context
            .load_place(&container)
            .map_err(|message| EvalError {
                message,
                location: field.location,
            })?;
        let field_name = field.source(self.source);
        let TypedValue::Struct(structure) = &container_value else {
            return Err(EvalError {
                message: format!(
                    "Cannot access field `{field_name}` of {}",
                    container_value.type_of()
                )
                .into_boxed_str(),
                location: field.location,
            });
        };
        if !structure.fields().contains_key(field_name) {
            return Err(EvalError {
                message: format!("Struct `{}` has no field `{field_name}`", structure.name())
                    .into_boxed_str(),
                location: field.location,
            });
        }

        let mut path = container.path.into_vec();
        path.push(Box::from(field_name));
        Ok(Place {
            root: container.root,
            path: path.into_boxed_slice(),
        })
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

        self.typecheck_named(
            value,
            hint.type_name.source(self.source),
            false,
            hint.type_name.location,
        )
    }

    /// Type-checks a value against a named type, which may be a scalar, a struct
    /// name, or (when `is_ref`) a reference to one of those.
    fn typecheck_named(
        &self,
        value: TypedValue<T>,
        type_name: &str,
        is_ref: bool,
        location: Location,
    ) -> Result<TypedValue<T>, EvalError> {
        if is_ref {
            // Reference-typed position: accept any reference. The pointee kind is
            // checked loosely and struct identity is checked at runtime on use.
            return match &value {
                TypedValue::Ref(_) => Ok(value),
                other => Err(EvalError {
                    message: format!("Expected &{type_name}, got {}", other.type_of())
                        .into_boxed_str(),
                    location,
                }),
            };
        }

        match Type::from_name(type_name) {
            Ok(ty) => self.typecheck(value, ty, location),
            Err(_) => match &value {
                // A struct-typed position: the value must be a struct with a
                // matching name (loose, runtime-checked identity).
                TypedValue::Struct(structure) if structure.name() == type_name => Ok(value),
                TypedValue::Struct(structure) => Err(EvalError {
                    message: format!(
                        "Expected struct `{type_name}`, got struct `{}`",
                        structure.name()
                    )
                    .into_boxed_str(),
                    location,
                }),
                other => {
                    if self.context.struct_fields(type_name).is_some() {
                        Err(EvalError {
                            message: format!(
                                "Expected struct `{type_name}`, got {}",
                                other.type_of()
                            )
                            .into_boxed_str(),
                            location,
                        })
                    } else {
                        Err(EvalError {
                            message: format!("Unknown type `{type_name}`").into_boxed_str(),
                            location,
                        })
                    }
                }
            },
        }
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

            let arg_value = self.typecheck_named(
                arg_value.clone(),
                arg.arg_type.type_name.source(self.source),
                arg.reference_token.is_some(),
                arg.arg_type.type_name.location,
            )?;

            self.context.declare(arg_name, arg_value);
        }

        let retval = match self.visit_body(&function.body)? {
            StatementResult::Return(typed_value) | StatementResult::ImplicitReturn(typed_value) => {
                typed_value
            }
            StatementResult::EndOfBody => TypedValue::Void,
            StatementResult::LoopBreak | StatementResult::LoopContinue => todo!(),
        };

        let retval =
            self.typecheck_with_hint(retval, function.return_decl.as_ref().map(|d| d.return_type))?;

        Ok(retval)
    }

    fn visit_body(&mut self, body: &Body<T::Parser>) -> Result<StatementResult<T>, EvalError> {
        self.context.open_scope();

        let mut body_result = StatementResult::EndOfBody;
        for statement in body.statements.iter() {
            if let Some(retval) = self.visit_statement(statement)? {
                body_result = retval;
                match body_result {
                    StatementResult::ImplicitReturn(_) => {}
                    _ => break,
                }
            } else {
                // Reset result if we have statements after implicit returns.
                body_result = StatementResult::EndOfBody;
            }
        }

        self.context.close_scope();
        Ok(body_result)
    }

    fn visit_statement(
        &mut self,
        statement: &Statement<T::Parser>,
    ) -> Result<Option<StatementResult<T>>, EvalError> {
        match statement {
            Statement::Return(return_with_value) => {
                return self
                    .visit_right_hand_expression(&return_with_value.expression)
                    .map(|rv| Some(StatementResult::Return(rv)));
            }
            Statement::ImplicitReturn(expression) => {
                return self
                    .visit_right_hand_expression(expression)
                    .map(|rv| Some(StatementResult::ImplicitReturn(rv)));
            }
            Statement::EmptyReturn(_) => {
                return Ok(Some(StatementResult::Return(TypedValue::Void)));
            }
            Statement::If(if_statement) => return self.visit_if(if_statement),
            Statement::Loop(loop_statement) => return self.visit_loop(loop_statement),
            Statement::For(for_statement) => return self.visit_for(for_statement),
            Statement::Break(_) => return Ok(Some(StatementResult::LoopBreak)),
            Statement::Continue(_) => return Ok(Some(StatementResult::LoopContinue)),
            Statement::Scope(body) => {
                return self.visit_body(body).map(|r| match r {
                    StatementResult::EndOfBody => None,
                    r => Some(r),
                });
            }
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
        let value = self.visit_right_hand_expression(&decl.initializer)?;

        let value = self.typecheck_with_hint(value, decl.type_token)?;

        self.context.declare(name, value);

        Ok(())
    }

    fn visit_if(
        &mut self,
        if_statement: &If<T::Parser>,
    ) -> Result<Option<StatementResult<T>>, EvalError> {
        let condition = self.visit_right_hand_expression(&if_statement.condition)?;

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
                StatementResult::LoopContinue
                | StatementResult::EndOfBody
                | StatementResult::ImplicitReturn(_) => {}
            }
        }
    }

    fn visit_for(
        &mut self,
        for_statement: &For<T::Parser>,
    ) -> Result<Option<StatementResult<T>>, EvalError> {
        // Evaluate the iterable. It must produce an iterator.
        let iterable = self.visit_right_hand_expression(&for_statement.iterable)?;
        let TypedValue::Iter(iter) = iterable else {
            return Err(EvalError {
                message: format!("Expected an iterator, got {}", iterable.type_of())
                    .into_boxed_str(),
                location: for_statement.iterable.location(),
            });
        };

        // The loop variable's declared type, if annotated. Values produced by the
        // iterator are checked against it, matching the VM's runtime behavior. When
        // the annotation is omitted, values are bound as-is.
        let elem_ty = match for_statement.var_type.as_ref() {
            Some(var_type) => Some(
                Type::from_name(var_type.type_name.source(self.source)).map_err(|message| {
                    EvalError {
                        message,
                        location: var_type.type_name.location,
                    }
                })?,
            ),
            None => None,
        };
        let var_name = for_statement.variable.source(self.source);

        loop {
            let Some(value) = self.context.type_context().iter_next(&iter) else {
                return Ok(None);
            };
            // A mismatch is about the declared element type, so point the error at
            // the type annotation rather than the loop variable name. Without an
            // annotation the value is bound as-is.
            let value = match elem_ty {
                Some(elem_ty) => self.typecheck(value, elem_ty, for_statement.variable.location)?,
                None => value,
            };

            // Bind the loop variable in a fresh scope for this iteration.
            self.context.open_scope();
            self.context.declare(var_name, value);

            let body_result = self.visit_body(&for_statement.body);
            self.context.close_scope();

            match body_result? {
                ret @ StatementResult::Return(_) => return Ok(Some(ret)),
                StatementResult::LoopBreak => return Ok(None),
                StatementResult::LoopContinue
                | StatementResult::EndOfBody
                | StatementResult::ImplicitReturn(_) => {}
            }
        }
    }
}

enum StatementResult<T: TypeSet> {
    Return(TypedValue<T>),
    ImplicitReturn(TypedValue<T>),
    LoopBreak,
    LoopContinue,
    EndOfBody,
}
