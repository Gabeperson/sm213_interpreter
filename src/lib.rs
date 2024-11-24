use parser::span::Span;
use sm213_parser::*;
use std::{
    collections::{BTreeMap, HashMap},
    hash::{BuildHasher, Hasher},
    ops::BitOr,
};

#[derive(Clone)]
pub struct Memory {
    // This takes up 5x the amount but makes it easy to detect overwrites, so we use it
    // Also because the actual simulator only supports memory up to ~1mb, this should
    // have an absolute max of 5mb
    buffer: HashMap<i32, u8, IdentityHasherGenerator>,
}

impl std::fmt::Debug for Memory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let treemap = self.buffer.iter().collect::<BTreeMap<_, _>>();
        writeln!(f, "\nMemory: {{")?;
        for (k, v) in treemap {
            writeln!(f, "    {k:x}: {v:0>2x}")?;
        }
        writeln!(f, "}}\n")?;
        Ok(())
    }
}

#[derive(Clone, Debug, Copy)]
pub struct IdentityHasherGenerator;

impl BuildHasher for IdentityHasherGenerator {
    type Hasher = IdentityHasher;

    fn build_hasher(&self) -> Self::Hasher {
        IdentityHasher(0)
    }
}

pub struct IdentityHasher(i32);

impl Hasher for IdentityHasher {
    fn finish(&self) -> u64 {
        self.0 as u64
    }

    fn write(&mut self, _bytes: &[u8]) {
        unimplemented!("Only implemented for i32")
    }

    fn write_i32(&mut self, i: i32) {
        self.0 = i
    }
}

#[derive(Debug, Clone, Copy)]
pub enum WriteType {
    Regular,
    Overwrite,
}

impl WriteType {
    fn is_overwrite(&self) -> bool {
        matches!(self, WriteType::Overwrite)
    }
}

impl std::ops::BitOr for WriteType {
    type Output = WriteType;

    fn bitor(self, rhs: Self) -> Self::Output {
        if let (WriteType::Regular, WriteType::Regular) = (self, rhs) {
            WriteType::Regular
        } else {
            WriteType::Overwrite
        }
    }
}

impl std::ops::BitOrAssign for WriteType {
    fn bitor_assign(&mut self, rhs: Self) {
        *self = self.bitor(rhs)
    }
}

#[derive(Debug, Clone)]
pub enum MemoryAccessError {
    UnalignedAccess(i32),
    NegativeAddress(i32),
    BufferOverflow(i32),
}

impl Memory {
    pub fn new() -> Self {
        Self {
            buffer: HashMap::with_hasher(IdentityHasherGenerator),
        }
    }
    // Must only be called when address is valid
    #[inline(always)]
    fn read_byte(&self, address: i32) -> u8 {
        *self.buffer.get(&address).unwrap_or(&0)
    }
    // Must only be called when address is valid
    #[inline(always)]
    fn write_byte(&mut self, address: i32, byte: u8) -> WriteType {
        let prev = self.buffer.insert(address, byte);
        if prev.is_some() {
            WriteType::Overwrite
        } else {
            WriteType::Regular
        }
    }
    pub fn read_i32(&self, address: i32) -> Result<i32, MemoryAccessError> {
        self.err_on_address(address, 4, 4)?;
        let b1 = self.read_byte(address);
        let b2 = self.read_byte(address + 1);
        let b3 = self.read_byte(address + 2);
        let b4 = self.read_byte(address + 3);
        Ok(i32::from_be_bytes([b1, b2, b3, b4]))
    }
    pub fn write_i32(&mut self, address: i32, data: i32) -> Result<WriteType, MemoryAccessError> {
        self.err_on_address(address, 4, 4)?;
        let [b1, b2, b3, b4] = data.to_be_bytes();
        let mut write_type = WriteType::Regular;
        write_type |= self.write_byte(address, b1);
        write_type |= self.write_byte(address + 1, b2);
        write_type |= self.write_byte(address + 2, b3);
        write_type |= self.write_byte(address + 3, b4);
        Ok(write_type)
    }

    pub fn write_i32_unaligned(
        &mut self,
        address: i32,
        data: i32,
    ) -> Result<WriteType, MemoryAccessError> {
        self.err_on_address(address, 4, 1)?;
        let [b1, b2, b3, b4] = data.to_be_bytes();
        let mut write_type = WriteType::Regular;
        write_type |= self.write_byte(address, b1);
        write_type |= self.write_byte(address + 1, b2);
        write_type |= self.write_byte(address + 2, b3);
        write_type |= self.write_byte(address + 3, b4);
        Ok(write_type)
    }

    pub fn read_instruction(&self, address: i32) -> Result<(u16, i32), MemoryAccessError> {
        self.err_on_address(address, 2, 2)?;
        let b1 = self.read_byte(address);
        let b2 = self.read_byte(address + 1);
        // This will technically cause issues if the user manually sets bytes at
        // between 0xffffc to 0xffffff to a ld immediate or jump immediate
        // then jumps there, but we turn a blind eye to this.
        let imm = self.read_i32(address + 2).unwrap_or(0);
        let ins = u16::from_be_bytes([b1, b2]);
        Ok((ins, imm))
    }

    pub fn write_instruction2(
        &mut self,
        address: i32,
        inst: u16,
    ) -> Result<WriteType, MemoryAccessError> {
        self.err_on_address(address, 2, 2)?;
        let [b1, b2] = inst.to_be_bytes();
        let mut write_type = WriteType::Regular;
        write_type |= self.write_byte(address, b1);
        write_type |= self.write_byte(address + 1, b2);
        Ok(write_type)
    }

    pub fn write_instruction6(
        &mut self,
        address: i32,
        inst: u16,
        imm: i32,
    ) -> Result<WriteType, MemoryAccessError> {
        self.err_on_address(address, 6, 2)?;
        let [b1, b2] = inst.to_be_bytes();
        let mut write_type = WriteType::Regular;
        write_type |= self.write_byte(address, b1);
        write_type |= self.write_byte(address + 1, b2);
        write_type |= self.write_i32_unaligned(address + 2, imm)?;
        Ok(write_type)
    }

    /// Returns `Ok` or `Err` depending on a few factors.
    /// - If address is negative (out of buffer), it will return `Err`
    /// - If address is not aligned, it will return `Err`
    /// - If address + size overflows the buffer, it will return `Err`
    /// - Otherwise, returns `Ok`
    pub fn err_on_address(
        &self,
        address: i32,
        size: i32,
        align: i32,
    ) -> Result<(), MemoryAccessError> {
        if address < 0 {
            return Err(MemoryAccessError::NegativeAddress(address));
        }
        if address % align != 0 {
            return Err(MemoryAccessError::UnalignedAccess(address));
        }
        // Original SM213 simulator only allows
        // memory addresses from 0x0 to 0xf_ffff to be valid
        if address > 0xf_ffff - size + 1 {
            return Err(MemoryAccessError::BufferOverflow(address));
        }
        Ok(())
    }
}

impl Default for Memory {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug)]
pub enum CompileError {
    InstructionOverwritten {
        overwritten: usize,
        overwriter: usize,
    },
    MemoryAccessError {
        instruction_index: usize,
        err: MemoryAccessError,
    },
    BranchTooFar {
        inst_index: usize,
        branch_from: i32,
        branch_to: i32,
        branch_to_label: Option<(String, Span)>,
    },
}

pub fn compile(program: &Program) -> Result<Memory, CompileError> {
    let compiler = Compiler::new();
    compiler.compile(program)
}

#[derive(Clone, Debug)]
struct Compiler<'a> {
    loc: i32,
    labels: HashMap<&'a str, i32>,
    memory: Memory,
    label_queue: Vec<(&'a str, i32)>,
    branch_label_queue: Vec<(Label<'a>, i32, usize)>,
    loc_map: HashMap<i32, usize>,
    iter_index: usize,
}

impl<'a> Compiler<'a> {
    fn new() -> Self {
        Self {
            loc: 0,
            labels: HashMap::new(),
            memory: Memory::new(),
            label_queue: Vec::new(),
            branch_label_queue: Vec::new(),
            loc_map: HashMap::new(),
            iter_index: 0,
        }
    }
}

impl<'a> Compiler<'a> {
    fn compile(mut self, input: &'a Program) -> Result<Memory, CompileError> {
        for (iter_index, line) in input.inner.iter().enumerate() {
            self.iter_index = iter_index;
            let statement = match line {
                Line::Code(s) | Line::CodeAndComment(s, _) => s,
                _ => continue,
            };
            if let Some(((l, _), _)) = statement.label_and_comment {
                self.labels.insert(l.0, self.loc);
            }
            match statement.instruction.inst {
                ref inst @ Instruction::Load {
                    from: (from, _),
                    to,
                } => match from {
                    LoadFrom::ImmediateNumber(n) => {
                        self.push_instruction6(inst_from_parts(0, to.value(), 0xF, 0xF), n.value)?;
                    }
                    LoadFrom::ImmediateLabel(l) => {
                        self.label_queue.push((l.0, self.loc + 2));
                        self.push_instruction6(inst_from_parts(0, to.value(), 0xF, 0xF), 0)?;
                    }
                    _ => self.push_instruction2(inst.to_quad_hexit())?,
                },
                Instruction::Branch { to: (to, _) } => match to {
                    BranchLocation::Address(a) => {
                        let branch_offset =
                            self.check_branch(self.loc, a.value, self.iter_index, None)?;
                        self.push_instruction2(inst_from_parts(
                            8,
                            0xF,
                            (branch_offset >> 4) & 0xF,
                            branch_offset & 0xF,
                        ))?;
                    }
                    BranchLocation::Label(l) => {
                        self.branch_label_queue
                            .push((l, self.loc + 1, self.iter_index));
                        self.push_instruction2(0x8FFF)?;
                    }
                },
                Instruction::BranchIfEqual { reg, to: (to, _) } => match to {
                    BranchLocation::Address(a) => {
                        let branch_offset =
                            self.check_branch(self.loc, a.value, self.iter_index, None)?;
                        self.push_instruction2(inst_from_parts(
                            9,
                            reg.value(),
                            (branch_offset >> 4) & 0xF,
                            branch_offset & 0xF,
                        ))?;
                    }
                    BranchLocation::Label(l) => {
                        self.branch_label_queue
                            .push((l, self.loc + 1, self.iter_index));
                        self.push_instruction2(inst_from_parts(9, reg.value(), 0xF, 0xF))?;
                    }
                },
                Instruction::BranchIfGreater { reg, to: (to, _) } => match to {
                    BranchLocation::Address(a) => {
                        let branch_offset =
                            self.check_branch(self.loc, a.value, self.iter_index, None)?;
                        self.push_instruction2(inst_from_parts(
                            0xA,
                            reg.value(),
                            (branch_offset >> 4) & 0xF,
                            branch_offset & 0xF,
                        ))?;
                    }
                    BranchLocation::Label(l) => {
                        self.branch_label_queue
                            .push((l, self.loc + 1, self.iter_index));
                        self.push_instruction2(inst_from_parts(0xA, reg.value(), 0xF, 0xF))?;
                    }
                },
                ref inst @ Instruction::Jump { to: (jump_loc, _) } => match jump_loc {
                    JumpLocation::Label(l) => {
                        self.label_queue.push((l.0, self.loc + 2));
                        self.push_instruction6(inst_from_parts(0xB, 0xF, 0xF, 0xF), 0)?;
                    }
                    JumpLocation::Addr(a) => {
                        self.push_instruction6(inst_from_parts(0xB, 0xF, 0xF, 0xF), a.value)?;
                    }
                    _ => self.push_instruction2(inst.to_quad_hexit())?,
                },
                Instruction::DirectiveLong { value: (value, _) } => match value {
                    DirectiveLongValue::Number(n) => {
                        self.set_num(self.loc, n.value, iter_index)?;
                    }
                    DirectiveLongValue::Label(l) => {
                        self.label_queue.push((l.0, self.loc));
                        self.set_num(self.loc, 0, iter_index)?;
                    }
                },
                Instruction::DirectivePos { loc: (loc, _) } => self.loc = loc.value,
                ref inst => self.push_instruction2(inst.to_quad_hexit())?,
            }
        }
        for (label, loc) in self.label_queue.iter() {
            let label_loc = *self.labels.get(label).unwrap();
            self.memory.write_i32_unaligned(*loc, label_loc).unwrap();
        }
        for (label, loc, index) in self.branch_label_queue.iter() {
            let label_loc = *self.labels.get(label.0).unwrap();
            let offset = self.check_branch(*loc, label_loc, *index, Some(*label))?;
            self.memory.write_byte(*loc, offset);
        }

        Ok(self.memory)
    }
}

impl<'a> Compiler<'a> {
    fn check_branch(
        &self,
        from: i32,
        to: i32,
        instruction_index: usize,
        label: Option<Label<'_>>,
    ) -> Result<u8, CompileError> {
        let distance = to - from;
        if !(-254..=256).contains(&distance) {
            return Err(CompileError::BranchTooFar {
                inst_index: instruction_index,
                branch_from: from,
                branch_to: to,
                branch_to_label: label.map(|s| (s.0.to_owned(), s.1)),
            });
        }
        let branch_offset = ((distance + 254) / 2) as u8;
        Ok(branch_offset)
    }
    fn check_not_overlap(
        &self,
        kind: WriteType,
        loc: i32,
        inst: usize,
    ) -> Result<(), CompileError> {
        if kind.is_overwrite() {
            return Err(CompileError::InstructionOverwritten {
                overwritten: *self.loc_map.get(&loc).unwrap(),
                overwriter: inst,
            });
        } else {
            Ok(())
        }
    }
    fn update_loc_map(&mut self, size: i32, instruction_index: usize) {
        for i in 0..size {
            self.loc_map.insert(self.loc + i, instruction_index);
        }
    }

    fn set_num(
        &mut self,
        loc: i32,
        num: i32,
        instruction_index: usize,
    ) -> Result<(), CompileError> {
        let kind =
            self.memory
                .write_i32(loc, num)
                .map_err(|e| CompileError::MemoryAccessError {
                    instruction_index,
                    err: e,
                })?;
        self.check_not_overlap(kind, loc, instruction_index)?;
        self.update_loc_map(4, instruction_index);
        self.loc += 4;
        Ok(())
    }

    fn push_instruction2(&mut self, inst: u16) -> Result<(), CompileError> {
        let kind = self
            .memory
            .write_instruction2(self.loc, inst)
            .map_err(|e| CompileError::MemoryAccessError {
                instruction_index: self.iter_index,
                err: e,
            })?;
        self.check_not_overlap(kind, self.loc, self.iter_index)?;
        self.update_loc_map(2, self.iter_index);
        self.loc += 2;
        Ok(())
    }
    fn push_instruction6(&mut self, inst: u16, imm: i32) -> Result<(), CompileError> {
        let kind = self
            .memory
            .write_instruction6(self.loc, inst, imm)
            .map_err(|e| CompileError::MemoryAccessError {
                instruction_index: self.iter_index,
                err: e,
            })?;
        self.check_not_overlap(kind, self.loc, self.iter_index)?;
        self.update_loc_map(6, self.iter_index);
        self.loc += 6;
        Ok(())
    }
}

trait InstructionToOp {
    fn to_quad_hexit(&self) -> u16;
}

impl InstructionToOp for Instruction<'_> {
    fn to_quad_hexit(&self) -> u16 {
        match self {
            Instruction::Load {
                from: (from, _),
                to,
            } => match from {
                LoadFrom::ImmediateNumber(_n) => {
                    unimplemented!("Should not be called on a load immediate instruction")
                }
                LoadFrom::ImmediateLabel(_l) => {
                    unimplemented!("Should not be called on a load immediate instruction")
                }
                LoadFrom::Offset { offset, base } => inst_from_parts(
                    1,
                    (offset.map(|n| n.value).unwrap_or(0) / 4) as u8,
                    base.value(),
                    to.value(),
                ),
                LoadFrom::Indexed { base, index } => {
                    inst_from_parts(2, base.value(), index.value(), to.value())
                }
            },
            Instruction::Store { from, to: (to, _) } => match to {
                StoreTo::Offset { offset, base } => inst_from_parts(
                    3,
                    from.value(),
                    (offset.map(|n| n.value).unwrap_or(0) / 4) as u8,
                    base.value(),
                ),
                StoreTo::Indexed { base, index } => {
                    inst_from_parts(4, from.value(), base.value(), index.value())
                }
            },
            Instruction::Halt => 0xF000,
            Instruction::Nop => 0xFF00,
            Instruction::Mov { from, to } => inst_from_parts(6, 0, from.value(), to.value()),
            Instruction::Add { from, to } => inst_from_parts(6, 1, from.value(), to.value()),
            Instruction::And { from, to } => inst_from_parts(6, 2, from.value(), to.value()),
            Instruction::Inc { reg } => inst_from_parts(6, 3, 0xF, reg.value()),
            Instruction::IncAddr { reg } => inst_from_parts(6, 4, 0xF, reg.value()),
            Instruction::Dec { reg } => inst_from_parts(6, 5, 0xF, reg.value()),
            Instruction::DecAddr { reg } => inst_from_parts(6, 6, 0xF, reg.value()),
            Instruction::Not { reg } => inst_from_parts(6, 7, 0xF, reg.value()),
            Instruction::ShiftLeft { amt, reg } => {
                let amt = amt.value as u8;
                inst_from_parts(7, reg.value(), (amt >> 4) & 0xF, amt & 0xF)
            }
            Instruction::ShiftRight { amt, reg } => {
                let amt = amt.value as u8;
                let amt = (-(amt as i8)) as u8;
                inst_from_parts(7, reg.value(), (amt >> 4) & 0xF, amt & 0xF)
            }
            Instruction::Branch { .. }
            | Instruction::BranchIfEqual { .. }
            | Instruction::BranchIfGreater { .. } => {
                unimplemented!("Should not be called on a branch-like instruction")
            }
            Instruction::GetProgramCounter { offset, to } => {
                inst_from_parts(6, 0xF, (offset.value / 2) as u8, to.value())
            }
            Instruction::Jump { to: (location, _) } => match location {
                JumpLocation::Label(_) => {
                    unimplemented!("Should not be called on a jump label instruction")
                }
                JumpLocation::Addr(_a) => {
                    unimplemented!("Should not be called on a jump addr instruction")
                }
                JumpLocation::Indirect { offset, to } => {
                    let offset = offset.map(|n| n.value).unwrap_or(0);
                    let offset = (offset / 2) as u8;
                    inst_from_parts(0xC, to.value(), (offset >> 4) & 0xF, offset & 0xF)
                }
                JumpLocation::DoubleIndirect(DoubleIndirectMethod::Offset { offset, to }) => {
                    let offset = offset.map(|n| n.value).unwrap_or(0);
                    let offset = (offset / 4) as u8;
                    inst_from_parts(0xD, to.value(), (offset >> 4) & 0xF, offset & 0xF)
                }
                JumpLocation::DoubleIndirect(DoubleIndirectMethod::Indexed { base, index }) => {
                    inst_from_parts(0xE, base.value(), index.value(), 0xF)
                }
            },
            Instruction::Syscall { typ: (typ, _) } => inst_from_parts(
                0xF,
                1,
                0,
                match typ {
                    SyscallType::Read => 0,
                    SyscallType::Write => 1,
                    SyscallType::Exec => 2,
                },
            ),
            Instruction::DirectiveLong { .. } | Instruction::DirectivePos { .. } => {
                unimplemented!("Should not be called on directives!")
            }
        }
    }
}

fn inst_from_parts(opcode: u8, op0: u8, op1: u8, op2: u8) -> u16 {
    (((opcode as u16) & 0b1111) << 12)
        | (((op0 as u16) & 0b1111) << 8)
        | (((op1 as u16) & 0b1111) << 4)
        | ((op2 as u16) & 0b1111)
}
