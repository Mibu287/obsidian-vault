#Rust #ProgrammingLanguages


# 1. Declarative macros

## 1.1.  Definition

```Rust
macro_rules! foobar {
    ...
}
```


## 1.2. Usage

```Rust
foobar!(...);
let x = foobar!(...);
let y = foobar![...];
let z = foobar!{...};
```

Any of the invocations above is valid. In fact, the grouping tokens are not passed to the invocation.

==NOTE:==
- C/C++ macros which are glorified text replacement and the expansion happen before any work done by the compiler. 
- Rust macro expansion happens after the source codes are parsed into ASTs ==> The expanded code only replace the AST node.

Examples:

```C++
#include <iostream>
#define add(x, y) x + y
int main() {
  int result = 2 * add(3, 4);
  std::cout << result << '\n';
}
// 10
```

```Rust
macro_rules! add {
    ($x:tt, $y:tt) => {
        $x + $y
    };
}

fn main() {
    let result = 2 * add!(3, 4);
    println!("{}", result);
}
// 14
```

NOTE:
- Rust macros are more sane than C/C++ counterpart since its expansion must match the AST type expected at that position.
- Rust macros can not expand to incomplete / syntactically invalid constructs.
- C/C++ macros can be expanded to anything. Only requirement is all macro expansion and source code piece together to be syntactically valid ==> Easy to be abused and lead to unreadable codes.

## 1.3. Notes

- The macro parser work like a LL(1) parser, it requires 1 token ahead to decide which rule to match.
  E.g: The macro rule below is ambiguous because the parser can not decide which rule to match.
```Rust
macro_rules! ambiguous {
    ($($x:tt)* $y:tt) => {}
}
```

- Rust macro is hygienic in the sense that it does not allow tokens from in and out of the macros merged together like C/C++ macros.
  E.g: Consider the code and the result below:
```Rust
macro_rules! using_a {
    ($e:expr) => {
        {
            let a = 42;
            $e
        }
    }
}

let four = using_a!(a / 10);
```

```Text
error[E0425]: cannot find value `a` in this scope
  --> src/main.rs:13:21
   |
13 | let four = using_a!(a / 10);
   |                     ^ not found in this scope

```

The identifier `a` is defined inside the macro body is marked different form the macro passed into the macro ==> The compiler emit the error above.

- Rust allow internal rules inside macro. E.g: The 2 code snippets below are equivalent.

```Rust
macro_rules! foo {
    (@as_expr $e:expr) => {$e};

    ($($tts:tt)*) => {
        foo!(@as_expr $($tts)*)
    };
}
```

```Rust
macro_rules! as_expr { ($e:expr) => {$e} }

macro_rules! foo {
    ($($tts:tt)*) => {
        as_expr!($($tts)*)
    };
}

```

- Macros do not follow visibility rules like functions / structs. Consider the example below:

```Rust
mod foo {
    mod bar {
        #[macro_export]
        macro_rules! foo {
            ($a:ident, $expr:expr) => {{
                let $a: i32 = 100;
                $expr
            }};
        }
    }
}

mod spam {
    pub(crate) fn foo() {
        let a = crate::foo!(a, a / 10);
        println!("{}", a);
    }
}

fn main() {
    spam::foo();
}
```

`#[macro_export]` make macro visible at crate level. The function inside `spam` module can refer to that macro without the full path.