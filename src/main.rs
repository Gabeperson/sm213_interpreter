use sm213_interpreter::*;
use sm213_parser::*;
fn main() {
    let s = std::fs::read_to_string("program.s").unwrap();
    let program = parse(&s).unwrap();
    dbg!(&second_pass(&program));
    let memory = compile(&program).unwrap();
    dbg!(&memory);
}
