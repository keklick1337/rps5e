// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Embedded vertex fetch: finds the attribute-table and buffer-table walks of
//! a vertex prolog and the buffer loads they end in, so the host can bind
//! them as fixed-function vertex attributes.

use crate::inline_constants;
use crate::ir_types::*;
use crate::opcode::Opcode;
use std::collections::{HashMap, HashSet};

/// One buffer load of the prolog that reads an attribute's stream.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EmbeddedVertexFetchLoad {
    pub pc: u32,
    pub attribute_id: i32,
    pub components: u32,
    /// The scalar table loads that produced its descriptor.
    pub prolog_loads: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EmbeddedVertexFetchPlan {
    pub loads: Vec<EmbeddedVertexFetchLoad>,
    /// User SGPR carrying the draw's vertex offset (-1 when none).
    pub vertex_offset_scalar_register: i32,
    pub instance_offset_scalar_register: i32,
}

impl Default for EmbeddedVertexFetchPlan {
    fn default() -> Self {
        EmbeddedVertexFetchPlan { loads: Vec::new(), vertex_offset_scalar_register: -1, instance_offset_scalar_register: -1 }
    }
}

impl EmbeddedVertexFetchPlan {
    /// Pcs of the fetch loads (pass to [`ShaderResourcePlan::extract`](super::ShaderResourcePlan::extract)).
    pub fn load_pcs(&self) -> HashSet<u32> {
        self.loads.iter().map(|l| l.pc).collect()
    }

    /// The program with the replaced table loads turned into S_NOPs.
    /// Instruction addresses are kept so branch targets and vertex-input
    /// bindings still refer to the same code.
    pub fn remove_replaced_table_loads(&self, program: &Program) -> Program {
        let replaced: HashSet<u32> = self.loads.iter().flat_map(|l| l.prolog_loads.iter().copied()).collect();
        let instructions = program
            .instructions
            .iter()
            .map(|i| {
                if replaced.contains(&i.pc) && matches!(i.control, Some(Control::ScalarMemory(_))) {
                    Instruction {
                        encoding: Encoding::Sopp,
                        opcode: Opcode::SNop,
                        sources: vec![],
                        destinations: vec![],
                        control: None,
                        ..i.clone()
                    }
                } else {
                    i.clone()
                }
            })
            .collect();
        Program::new(program.address, instructions)
    }
}

const SCALAR_REGISTER_COUNT: u32 = 108;
const VECTOR_REGISTER_COUNT: u32 = 256;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Ty {
    #[default]
    Unknown,
    Constant,
    AttributeTable,
    Attribute,
    BufferTable,
    Buffer,
    Index,
}

#[derive(Clone, Debug, Default)]
struct ScalarInfo {
    ty: Ty,
    attribute_id: i32,
    value: u32,
    prolog_loads: Option<Vec<u32>>,
}

fn tracked(o: Option<Operand>) -> bool {
    matches!(o, Some(o) if o.kind == OperandKind::ScalarRegister && o.value < SCALAR_REGISTER_COUNT)
}

fn vector(o: Option<Operand>) -> bool {
    matches!(o, Some(o) if o.kind == OperandKind::VectorRegister)
}

fn constant(scalars: &[ScalarInfo], o: Option<Operand>) -> Option<u32> {
    let o = o?;
    match o.kind {
        OperandKind::LiteralConstant => Some(o.value),
        OperandKind::EncodedConstant => {
            if o.value == 125 {
                Some(0)
            } else {
                inline_constants::decode(o.value)
            }
        }
        OperandKind::ScalarRegister if o.value < SCALAR_REGISTER_COUNT && scalars[o.value as usize].ty == Ty::Constant => {
            Some(scalars[o.value as usize].value)
        }
        _ => None,
    }
}

fn clear(scalars: &mut [ScalarInfo], destination: Operand, count: u32) {
    if destination.kind != OperandKind::ScalarRegister {
        return;
    }
    for i in 0..count {
        if destination.value + i < SCALAR_REGISTER_COUNT {
            scalars[(destination.value + i) as usize] = ScalarInfo::default();
        }
    }
}

fn lane(lane: u32, wave_size: u32) -> u32 {
    if wave_size == 32 || wave_size == 64 {
        lane % wave_size
    } else {
        lane
    }
}

/// Detect the fetch loads. `attribute_table_register` / `buffer_table_register`
/// are the SGPR pairs holding the attribute-descriptor and vertex-buffer
/// table pointers (user-data base + the AGC direct-resource offsets).
pub fn detect(
    program: &Program,
    attribute_table_register: i32,
    buffer_table_register: i32,
    user_data_base: u32,
    user_data_count: u32,
    wave_size: u32,
) -> EmbeddedVertexFetchPlan {
    let mut plan = EmbeddedVertexFetchPlan::default();
    let (mut vertex_candidate, mut instance_candidate) = (-1i32, -1i32);
    let (mut vertex_conflict, mut instance_conflict) = (false, false);
    let mut scalars = vec![ScalarInfo::default(); SCALAR_REGISTER_COUNT as usize];
    let mut vectors = vec![Ty::Unknown; VECTOR_REGISTER_COUNT as usize];
    let mut lanes: HashMap<(u32, u32), ScalarInfo> = HashMap::new();
    let track_lanes = !program.instructions.iter().any(|i| {
        use Opcode::*;
        matches!(i.opcode, SSetpcB64 | SBranch | SCbranchScc0 | SCbranchScc1 | SCbranchVccz | SCbranchVccnz | SCbranchExecz | SCbranchExecnz)
    });
    for register in [attribute_table_register, attribute_table_register + 1] {
        if (0..SCALAR_REGISTER_COUNT as i32).contains(&register) {
            scalars[register as usize].ty = Ty::AttributeTable;
        }
    }
    for register in [buffer_table_register, buffer_table_register + 1] {
        if (0..SCALAR_REGISTER_COUNT as i32).contains(&register) {
            scalars[register as usize].ty = Ty::BufferTable;
        }
    }
    for instruction in &program.instructions {
        let destination = instruction.destinations.first().copied();
        let (s0, s1, s2) = (instruction.sources.first().copied(), instruction.sources.get(1).copied(), instruction.sources.get(2).copied());
        // Fetch shaders accumulate the vertex offset in v0 (or v5, and the
        // instance offset in v8, under the user-data-at-s8 layout).
        let dst_value = destination.map_or(u32::MAX, |d| d.value);
        let vertex_accumulator = vector(destination) && (dst_value == 0 || (user_data_base == 8 && dst_value == 5));
        let instance_accumulator = vector(destination) && dst_value == if user_data_base == 8 { 8 } else { 3 };
        let index_offset_add = (vertex_accumulator || instance_accumulator)
            && tracked(s0)
            && ((instruction.opcode == Opcode::VAddI32 && vector(s1) && s1.unwrap().value == dst_value)
                || (user_data_base == 8
                    && (dst_value == 5 || dst_value == 8)
                    && instruction.opcode == Opcode::VSadU32
                    && vector(s2)
                    && s2.unwrap().value == dst_value
                    && constant(&scalars, s1) == Some(0)));
        if plan.loads.is_empty() && index_offset_add {
            let register = s0.unwrap().value;
            if register >= user_data_base && register - user_data_base < user_data_count {
                let (candidate, conflict) = if vertex_accumulator {
                    (&mut vertex_candidate, &mut vertex_conflict)
                } else {
                    (&mut instance_candidate, &mut instance_conflict)
                };
                if *candidate >= 0 && *candidate != register as i32 {
                    *conflict = true;
                } else {
                    *candidate = register as i32;
                }
            }
        }
        match instruction.opcode {
            Opcode::VWritelaneB32 => {
                if vector(destination) && dst_value < VECTOR_REGISTER_COUNT {
                    vectors[dst_value as usize] = Ty::Unknown;
                }
                match (track_lanes && vector(destination) && tracked(s0), constant(&scalars, s1)) {
                    (true, Some(l)) => {
                        lanes.insert((dst_value, lane(l, wave_size)), scalars[s0.unwrap().value as usize].clone());
                    }
                    _ => {
                        if vector(destination) {
                            lanes.retain(|k, _| k.0 != dst_value);
                        }
                    }
                }
            }
            Opcode::VReadlaneB32 => {
                match (track_lanes && tracked(destination) && vector(s0), constant(&scalars, s1)) {
                    (true, Some(l)) => {
                        scalars[dst_value as usize] = lanes.get(&(s0.unwrap().value, lane(l, wave_size))).cloned().unwrap_or_default();
                    }
                    _ => {
                        if tracked(destination) {
                            clear(&mut scalars, destination.unwrap(), 1);
                        }
                    }
                }
            }
            Opcode::SMovB32 => {
                if tracked(destination) && tracked(s0) {
                    scalars[dst_value as usize] = scalars[s0.unwrap().value as usize].clone();
                } else if tracked(destination) {
                    match constant(&scalars, s0) {
                        Some(value) => scalars[dst_value as usize] = ScalarInfo { ty: Ty::Constant, value, ..Default::default() },
                        None => clear(&mut scalars, destination.unwrap(), 1),
                    }
                }
            }
            Opcode::SMovkI32 => {
                if tracked(destination) {
                    scalars[dst_value as usize] = ScalarInfo {
                        ty: Ty::Constant,
                        value: instruction.sources[0].value as u16 as i16 as i32 as u32,
                        ..Default::default()
                    };
                }
            }
            opcode => {
                let name = opcode.name();
                if let (Some(Control::ScalarMemory(load)), true) = (&instruction.control, name.starts_with("SLoadDword")) {
                    apply_scalar_load(instruction, load, &mut scalars);
                } else if opcode == Opcode::VCndmaskB32 {
                    if vector(destination) && dst_value < VECTOR_REGISTER_COUNT {
                        lanes.retain(|k, _| k.0 != dst_value);
                        if vector(s0) && s0.unwrap().value == 8 && vector(s1) && s1.unwrap().value == 5 {
                            vectors[dst_value as usize] = Ty::Index;
                        }
                    }
                } else if matches!(opcode, Opcode::SBfeU32 | Opcode::SAndB32 | Opcode::SAddI32 | Opcode::SAddU32 | Opcode::SLshlB32) {
                    if tracked(destination) && tracked(s0) && scalars[s0.unwrap().value as usize].ty == Ty::Attribute {
                        scalars[dst_value as usize] = scalars[s0.unwrap().value as usize].clone();
                    } else if tracked(destination) {
                        match (constant(&scalars, s0), constant(&scalars, s1)) {
                            (Some(l), Some(r)) => {
                                let value = match opcode {
                                    Opcode::SAndB32 => l & r,
                                    Opcode::SLshlB32 => l << (r & 31),
                                    Opcode::SBfeU32 => l >> (r & 31),
                                    _ => l.wrapping_add(r),
                                };
                                scalars[dst_value as usize] = ScalarInfo { ty: Ty::Constant, value, ..Default::default() };
                            }
                            _ => clear(&mut scalars, destination.unwrap(), 1),
                        }
                    }
                } else if let Some(Control::BufferMemory(buffer)) = &instruction.control {
                    let fetch = matches!(
                        opcode,
                        Opcode::BufferLoadFormatX | Opcode::BufferLoadFormatXy | Opcode::BufferLoadFormatXyz | Opcode::BufferLoadFormatXyzw
                    );
                    if fetch
                        && buffer.vector_address < VECTOR_REGISTER_COUNT
                        && vectors[buffer.vector_address as usize] == Ty::Index
                        && buffer.scalar_resource < SCALAR_REGISTER_COUNT
                        && scalars[buffer.scalar_resource as usize].ty == Ty::Buffer
                    {
                        let table = &scalars[buffer.scalar_resource as usize];
                        if plan.loads.is_empty() {
                            if !vertex_conflict {
                                plan.vertex_offset_scalar_register = vertex_candidate;
                            }
                            if !instance_conflict {
                                plan.instance_offset_scalar_register = instance_candidate;
                            }
                        }
                        plan.loads.push(EmbeddedVertexFetchLoad {
                            pc: instruction.pc,
                            attribute_id: table.attribute_id,
                            components: buffer.dword_count.max(1),
                            prolog_loads: table.prolog_loads.clone().unwrap_or_default(),
                        });
                    }
                }
            }
        }
        if instruction.opcode == Opcode::VMovreldB32 {
            lanes.clear();
        } else if instruction.opcode != Opcode::VWritelaneB32 {
            for written in &instruction.destinations {
                if written.kind == OperandKind::VectorRegister {
                    lanes.retain(|k, _| k.0 != written.value);
                }
            }
        }
    }
    plan
}

/// A load through the attribute table yields attribute records; one through
/// the buffer table, or through a loaded attribute, yields buffer records.
fn apply_scalar_load(instruction: &Instruction, control: &ScalarMemoryControl, scalars: &mut [ScalarInfo]) {
    let destination = instruction.destinations.first().copied();
    let (s0, s1) = (instruction.sources.first().copied(), instruction.sources.get(1).copied());
    if !tracked(destination) {
        return;
    }
    let destination = destination.unwrap();
    let register = destination.value;
    let count = control.destination_count.max(1);
    let raw_offset = |scalars: &[ScalarInfo]| -> Option<u32> {
        let base = match s1 {
            Some(o) => constant(scalars, Some(o))?,
            None => 0,
        };
        let value = base as u64 + control.immediate_offset_bytes as u32 as u64;
        (value <= u32::MAX as u64).then_some(value as u32)
    };
    let source_ty = if tracked(s0) { scalars[s0.unwrap().value as usize].ty } else { Ty::Unknown };
    if source_ty == Ty::AttributeTable {
        match raw_offset(scalars) {
            Some(raw) => {
                let index = (raw / 4) as i32;
                for c in 0..count {
                    if register + c < SCALAR_REGISTER_COUNT {
                        scalars[(register + c) as usize] =
                            ScalarInfo { ty: Ty::Attribute, attribute_id: index + c as i32, value: 0, prolog_loads: Some(vec![instruction.pc]) };
                    }
                }
            }
            None => clear(scalars, destination, count),
        }
    } else if source_ty == Ty::BufferTable {
        if let Some(raw) = raw_offset(scalars) {
            for c in 0..count {
                if register + c < SCALAR_REGISTER_COUNT {
                    scalars[(register + c) as usize] = ScalarInfo {
                        ty: Ty::Buffer,
                        attribute_id: ((raw + c * 4) / 16) as i32,
                        value: 0,
                        prolog_loads: Some(vec![instruction.pc]),
                    };
                }
            }
        } else if tracked(s1) && scalars[s1.unwrap().value as usize].ty == Ty::Attribute && control.immediate_offset_bytes & 3 == 0 {
            let loaded = scalars[s1.unwrap().value as usize].clone();
            for c in 0..count {
                if register + c < SCALAR_REGISTER_COUNT {
                    let mut loads = loaded.prolog_loads.clone().unwrap_or_default();
                    loads.push(instruction.pc);
                    scalars[(register + c) as usize] =
                        ScalarInfo { ty: Ty::Buffer, attribute_id: loaded.attribute_id, value: 0, prolog_loads: Some(loads) };
                }
            }
        } else {
            clear(scalars, destination, count);
        }
    } else {
        clear(scalars, destination, count);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A buffer-descriptor load (s_load_dwordx4 s[0:3] from the buffer
    /// table at the attribute record held in s_n) whose destination
    /// overlaps s_n keeps the attribute and the load chain of that record.
    #[test]
    fn aliased_attribute_descriptor_keeps_its_provenance() {
        for wave in [32u32, 64] {
            for index_register in [0u32, 1, 2, 3, 16] {
                let code = [
                    0xf400_0004 | (index_register << 6),
                    0xfa00_000c,
                    0xf408_0005,
                    index_register << 25,
                    0x0214_0b08,
                    0xe00c_2000,
                    0x8000_0b0a,
                    0xbf81_0000,
                ];
                let bytes: Vec<u8> = code.iter().flat_map(|w| w.to_le_bytes()).collect();
                let program = crate::decode_blob(&bytes, 0x1000).unwrap();
                // Attribute table in s[8:9], buffer table in s[10:11] (the
                // merged layout's user data from s8).
                let plan = detect(&program, 8, 10, 0, 16, wave);
                assert_eq!(plan.loads.len(), 1, "wave{wave} s{index_register}: {plan:?}");
                let load = &plan.loads[0];
                assert_eq!((load.attribute_id, load.components, load.pc), (3, 4, 20), "wave{wave} s{index_register}");
                assert_eq!(load.prolog_loads, vec![0, 8], "wave{wave} s{index_register}");
            }
        }
    }
}
