Somni language and VM
=====================

[![Crates.io](https://img.shields.io/crates/v/somni?labelColor=1C2C2E&color=C96329&logo=Rust&style=flat-square)](https://crates.io/crates/somni)
[![docs.rs](https://img.shields.io/docsrs/somni?labelColor=1C2C2E&color=C96329&logo=rust&style=flat-square)](https://docs.rs/somni/latest/)
![MSRV](https://img.shields.io/badge/MSRV-1.82.0-blue?labelColor=1C2C2E&style=flat-square)
![Crates.io](https://img.shields.io/crates/l/somni?labelColor=1C2C2E&style=flat-square)

Somni is a simple, embeddable scripting language. There are two separate runtimes to run Somni programs:

- [`somni-expr`](https://crates.io/crates/somni-expr): a simpler, slower syntax tree evaluator
- [`somni`](https://crates.io/crates/somni): an optimizing compiler and VM, which is much more experimental and much less refined

## License

All packages within this repository are licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution notice

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in
the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without
any additional terms or conditions.
