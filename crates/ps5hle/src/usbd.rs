// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! libSceUsbd without host USB passthrough.
//!
//! The library follows libusb: devices, handles and descriptors are objects
//! the library hands out, and a pointer it did not return is a title bug.
//! No host devices are exposed, so the device list is always empty and no
//! device, handle or config descriptor is ever handed out: calls with NULL
//! arguments report what the console library reports, calls on any other
//! pointer log the misuse and return `SCE_USBD_ERROR_NO_DEVICE`. Transfer
//! structures (the libusb_transfer layout) and control setup packets are
//! plain guest memory and are filled exactly.
//!
//! Errors are libusb's codes mapped to `0x80240000 - code`.

#[cfg(windows)]
#[allow(unused_imports)]
use crate::host::c as libc;
use std::collections::HashSet;
use std::sync::Mutex;

use crate::context::{CpuContext, CpuRegister};
use crate::registry::{HleCallback, HleRegistry, HleResult};

const fn usbd_error(code: u32) -> HleResult {
    code as i32 as i64
}

/// LIBUSB_ERROR_INVALID_PARAM.
const ERR_INVALID_ARG: HleResult = usbd_error(0x8024_0002);
/// LIBUSB_ERROR_NO_DEVICE.
const ERR_NO_DEVICE: HleResult = usbd_error(0x8024_0004);
/// LIBUSB_ERROR_NOT_FOUND.
const ERR_NOT_FOUND: HleResult = usbd_error(0x8024_0005);

/// SceUsbdTransfer (the libusb_transfer layout): device handle +0, flags +8,
/// endpoint +9, type +10, timeout +12, status +16, length +20, actual
/// length +24, callback +32, user data +40, buffer +48, iso packet count
/// +56, iso packet descriptors (length, actual length, status; 12 bytes
/// each) from +60.
const T_DEVICE: u64 = 0;
const T_ENDPOINT: u64 = 9;
const T_TIMEOUT: u64 = 12;
const T_LENGTH: u64 = 20;
const T_CALLBACK: u64 = 32;
const T_USER: u64 = 40;
const T_BUFFER: u64 = 48;
const T_ISO_COUNT: u64 = 56;
const T_ISO_DESCRIPTORS: u64 = 60;
const ISO_DESCRIPTOR_SIZE: u64 = 12;

const TYPE_CONTROL: u8 = 0;
const TYPE_ISOCHRONOUS: u8 = 1;
const TYPE_BULK: u8 = 2;
const TYPE_INTERRUPT: u8 = 3;
/// Size of the control setup packet at the start of a control buffer.
const CONTROL_SETUP_SIZE: u64 = 8;

/// Empty device lists handed out by sceUsbdGetDeviceList and not yet freed.
static LISTS: Mutex<Option<HashSet<u64>>> = Mutex::new(None);

/// Stack argument `index` (0 = the first one past the six registers).
fn stack_arg(ctx: &CpuContext, index: u64) -> Option<u64> {
    ctx.read_u64(ctx.get(CpuRegister::Rsp) + 8 + 8 * index)
}

/// A device that sceUsbdGetDeviceList never returned (it returns none).
fn unknown_device(function: &str, device: u64) -> HleResult {
    log::error!("[usbd] {function}({device:#x}): the device was not returned by sceUsbdGetDeviceList (no host USB devices are exposed)");
    ERR_NO_DEVICE
}

/// A handle that sceUsbdOpen never returned (it opens nothing).
fn unknown_handle(function: &str, handle: u64) -> HleResult {
    log::error!("[usbd] {function}({handle:#x}): the handle was not returned by sceUsbdOpen (no host USB devices are exposed)");
    ERR_NO_DEVICE
}

/// sceUsbdInit() / sceUsbdExit(): nothing to set up without host devices.
fn cb_ok(_ctx: &mut CpuContext) -> HleResult {
    0
}

/// Poll interval of sceUsbdHandleEventsTimeout while no USB events exist.
const USBD_POLL_INTERVAL: std::time::Duration = std::time::Duration::from_millis(16);

/// sceUsbdHandleEventsTimeout(...): no USB events are ever pending; wait
/// one poll interval so event loops do not spin.
fn cb_handle_events_timeout(_ctx: &mut CpuContext) -> HleResult {
    crate::signals::sleep(USBD_POLL_INTERVAL);
    0
}

/// sceUsbdGetDeviceList(device***): an empty, NULL-terminated list and a
/// device count of 0.
fn cb_get_device_list(ctx: &mut CpuContext) -> HleResult {
    let out = ctx.rdi();
    if out == 0 {
        return ERR_INVALID_ARG;
    }
    let list = crate::libc::heap_block(ctx, 8);
    if list == 0 || !ctx.write_u64(list, 0) {
        crate::libc::heap_release(list);
        return usbd_error(0x8024_000B); // LIBUSB_ERROR_NO_MEM
    }
    if !ctx.write_u64(out, list) {
        crate::libc::heap_release(list);
        return ERR_INVALID_ARG;
    }
    LISTS.lock().unwrap_or_else(|p| p.into_inner()).get_or_insert_with(HashSet::new).insert(list);
    0
}

/// sceUsbdFreeDeviceList(list, unrefDevices): a list this library returned
/// is released; NULL is ignored.
fn cb_free_device_list(ctx: &mut CpuContext) -> HleResult {
    let list = ctx.rdi();
    if list == 0 {
        return 0;
    }
    let known = LISTS.lock().unwrap_or_else(|p| p.into_inner()).get_or_insert_with(HashSet::new).remove(&list);
    if known {
        crate::libc::heap_release(list);
    } else {
        log::error!("[usbd] sceUsbdFreeDeviceList({list:#x}): list was not returned by sceUsbdGetDeviceList");
    }
    0
}

/// sceUsbdAllocTransfer(int isoPackets) -> a zeroed transfer holding its
/// iso packet count, NULL for a negative count or no memory.
fn cb_alloc_transfer(ctx: &mut CpuContext) -> HleResult {
    let iso = ctx.rdi() as i32;
    if iso < 0 {
        return 0;
    }
    let size = T_ISO_DESCRIPTORS + ISO_DESCRIPTOR_SIZE * iso as u64;
    let t = crate::libc::heap_block(ctx, size);
    if t == 0 || !ctx.write_bytes(t, &vec![0u8; size as usize]) || !ctx.write_u32(t + T_ISO_COUNT, iso as u32) {
        return 0;
    }
    t as HleResult
}

/// sceUsbdFreeTransfer(transfer): NULL is ignored; nothing is ever in
/// flight.
fn cb_free_transfer(ctx: &mut CpuContext) -> HleResult {
    crate::libc::heap_release(ctx.rdi());
    0
}

/// The fields every libusb_fill_*_transfer sets.
#[allow(clippy::too_many_arguments)]
fn fill_transfer(ctx: &CpuContext, t: u64, device: u64, endpoint: u8, kind: u8, buffer: u64, length: u32, callback: u64, user: u64, timeout: u32) {
    ctx.write_u64(t + T_DEVICE, device);
    ctx.write_bytes(t + T_ENDPOINT, &[endpoint, kind]);
    ctx.write_u32(t + T_TIMEOUT, timeout);
    ctx.write_u32(t + T_LENGTH, length);
    ctx.write_u64(t + T_CALLBACK, callback);
    ctx.write_u64(t + T_USER, user);
    ctx.write_u64(t + T_BUFFER, buffer);
}

/// sceUsbdFill{Interrupt,Bulk}Transfer(transfer, device, u8 endpoint, u8*
/// buffer, int length, callback, userData, u32 timeout): user data and
/// timeout on the stack.
fn fill_endpoint_transfer(ctx: &mut CpuContext, function: &str, kind: u8) -> HleResult {
    let (t, device, endpoint, buffer, length, callback) = (ctx.rdi(), ctx.rsi(), ctx.rdx() as u8, ctx.rcx(), ctx.r8() as u32, ctx.r9());
    if t == 0 {
        log::error!("[usbd] {function}: NULL transfer");
        return 0;
    }
    let (Some(user), Some(timeout)) = (stack_arg(ctx, 0), stack_arg(ctx, 1)) else { return 0 };
    fill_transfer(ctx, t, device, endpoint, kind, buffer, length, callback, user, timeout as u32);
    0
}

fn cb_fill_interrupt_transfer(ctx: &mut CpuContext) -> HleResult {
    fill_endpoint_transfer(ctx, "sceUsbdFillInterruptTransfer", TYPE_INTERRUPT)
}

fn cb_fill_bulk_transfer(ctx: &mut CpuContext) -> HleResult {
    fill_endpoint_transfer(ctx, "sceUsbdFillBulkTransfer", TYPE_BULK)
}

/// sceUsbdFillIsoTransfer(transfer, device, u8 endpoint, u8* buffer, int
/// length, int numIsoPackets, callback, userData, u32 timeout).
fn cb_fill_iso_transfer(ctx: &mut CpuContext) -> HleResult {
    let (t, device, endpoint, buffer, length, packets) = (ctx.rdi(), ctx.rsi(), ctx.rdx() as u8, ctx.rcx(), ctx.r8() as u32, ctx.r9() as u32);
    if t == 0 {
        log::error!("[usbd] sceUsbdFillIsoTransfer: NULL transfer");
        return 0;
    }
    let (Some(callback), Some(user), Some(timeout)) = (stack_arg(ctx, 0), stack_arg(ctx, 1), stack_arg(ctx, 2)) else { return 0 };
    fill_transfer(ctx, t, device, endpoint, TYPE_ISOCHRONOUS, buffer, length, callback, user, timeout as u32);
    ctx.write_u32(t + T_ISO_COUNT, packets);
    0
}

/// sceUsbdFillControlSetup(u8* buffer, u8 requestType, u8 request, u16
/// value, u16 index, u16 length): the 8-byte setup packet, little endian.
fn cb_fill_control_setup(ctx: &mut CpuContext) -> HleResult {
    let buffer = ctx.rdi();
    if buffer == 0 {
        log::error!("[usbd] sceUsbdFillControlSetup: NULL buffer");
        return 0;
    }
    let (value, index, length) = (ctx.rcx() as u16, ctx.r8() as u16, ctx.r9() as u16);
    let mut setup = [ctx.rsi() as u8, ctx.rdx() as u8, 0, 0, 0, 0, 0, 0];
    setup[2..4].copy_from_slice(&value.to_le_bytes());
    setup[4..6].copy_from_slice(&index.to_le_bytes());
    setup[6..8].copy_from_slice(&length.to_le_bytes());
    ctx.write_bytes(buffer, &setup);
    0
}

/// sceUsbdFillControlTransfer(transfer, device, u8* buffer, callback,
/// userData, u32 timeout): endpoint 0, the length is the setup packet plus
/// its wLength.
fn cb_fill_control_transfer(ctx: &mut CpuContext) -> HleResult {
    let (t, device, buffer, callback, user, timeout) = (ctx.rdi(), ctx.rsi(), ctx.rdx(), ctx.rcx(), ctx.r8(), ctx.r9() as u32);
    if t == 0 {
        log::error!("[usbd] sceUsbdFillControlTransfer: NULL transfer");
        return 0;
    }
    let length = if buffer == 0 {
        0
    } else {
        let Some(w) = ctx.read_bytes(buffer + 6, 2) else { return 0 };
        CONTROL_SETUP_SIZE as u32 + u16::from_le_bytes([w[0], w[1]]) as u32
    };
    fill_transfer(ctx, t, device, 0, TYPE_CONTROL, buffer, length, callback, user, timeout);
    0
}

/// The transfer's buffer pointer, or None for a NULL transfer (logged).
fn transfer_buffer(ctx: &CpuContext, t: u64, function: &str) -> Option<u64> {
    if t == 0 {
        log::error!("[usbd] {function}: NULL transfer");
        return None;
    }
    ctx.read_u64(t + T_BUFFER)
}

/// sceUsbdControlTransferGetData(transfer) -> the data after the setup
/// packet, NULL without a buffer.
fn cb_control_transfer_get_data(ctx: &mut CpuContext) -> HleResult {
    match transfer_buffer(ctx, ctx.rdi(), "sceUsbdControlTransferGetData") {
        Some(b) if b != 0 => (b + CONTROL_SETUP_SIZE) as HleResult,
        _ => 0,
    }
}

/// sceUsbdControlTransferGetSetup(transfer) -> the buffer (setup packet).
fn cb_control_transfer_get_setup(ctx: &mut CpuContext) -> HleResult {
    transfer_buffer(ctx, ctx.rdi(), "sceUsbdControlTransferGetSetup").unwrap_or(0) as HleResult
}

/// sceUsbdSetIsoPacketLengths(transfer, u32 length): every iso packet
/// descriptor gets the length.
fn cb_set_iso_packet_lengths(ctx: &mut CpuContext) -> HleResult {
    let (t, length) = (ctx.rdi(), ctx.rsi() as u32);
    if t == 0 {
        log::error!("[usbd] sceUsbdSetIsoPacketLengths: NULL transfer");
        return 0;
    }
    let packets = ctx.read_u32(t + T_ISO_COUNT).unwrap_or(0) as i32;
    for packet in 0..packets.max(0) as u64 {
        ctx.write_u32(t + T_ISO_DESCRIPTORS + packet * ISO_DESCRIPTOR_SIZE, length);
    }
    0
}

/// sceUsbdGetIsoPacketBuffer(transfer, u32 packet) -> the buffer past the
/// lengths of the packets before it; NULL for a packet out of range or no
/// buffer.
fn cb_get_iso_packet_buffer(ctx: &mut CpuContext) -> HleResult {
    let (t, packet) = (ctx.rdi(), ctx.rsi() as u32);
    let Some(buffer) = transfer_buffer(ctx, t, "sceUsbdGetIsoPacketBuffer") else { return 0 };
    let packets = ctx.read_u32(t + T_ISO_COUNT).unwrap_or(0) as i32;
    if packets < 0 || packet >= packets as u32 || buffer == 0 {
        return 0;
    }
    let mut pointer = buffer;
    for index in 0..packet as u64 {
        pointer = pointer.wrapping_add(ctx.read_u32(t + T_ISO_DESCRIPTORS + index * ISO_DESCRIPTOR_SIZE).unwrap_or(0) as u64);
    }
    pointer as HleResult
}

/// sceUsbdCancelTransfer(transfer): no transfer is ever submitted without a
/// device, so there is nothing to cancel.
fn cb_cancel_transfer(_ctx: &mut CpuContext) -> HleResult {
    ERR_NOT_FOUND
}

/// sceUsbdSubmitTransfer(transfer): a NULL transfer or one without a
/// handle is invalid; any handle is one this library never opened.
fn cb_submit_transfer(ctx: &mut CpuContext) -> HleResult {
    let t = ctx.rdi();
    let handle = if t == 0 { 0 } else { ctx.read_u64(t + T_DEVICE).unwrap_or(0) };
    if handle == 0 {
        return ERR_INVALID_ARG;
    }
    unknown_handle("sceUsbdSubmitTransfer", handle)
}

/// sceUsbdEventHandlingOk(): no other thread handles USB events.
fn cb_event_handling_ok(_ctx: &mut CpuContext) -> HleResult {
    1
}

/// sceUsbdFreeConfigDescriptor(config): no descriptor is ever handed out,
/// so only NULL is expected.
fn cb_free_config_descriptor(ctx: &mut CpuContext) -> HleResult {
    if ctx.rdi() != 0 {
        log::error!("[usbd] sceUsbdFreeConfigDescriptor({:#x}): descriptor was not returned by this library", ctx.rdi());
    }
    0
}

/// Calls on a device handle whose only argument check is a NULL handle:
/// sceUsbdAttachKernelDriver, CheckConnected, ClaimInterface,
/// KernelDriverActive, ReleaseInterface, ResetDevice, SetConfiguration.
fn handle_call(ctx: &CpuContext, function: &str) -> HleResult {
    match ctx.rdi() {
        0 => ERR_INVALID_ARG,
        handle => unknown_handle(function, handle),
    }
}

fn cb_attach_kernel_driver(ctx: &mut CpuContext) -> HleResult {
    handle_call(ctx, "sceUsbdAttachKernelDriver")
}
fn cb_check_connected(ctx: &mut CpuContext) -> HleResult {
    handle_call(ctx, "sceUsbdCheckConnected")
}
fn cb_claim_interface(ctx: &mut CpuContext) -> HleResult {
    handle_call(ctx, "sceUsbdClaimInterface")
}
fn cb_kernel_driver_active(ctx: &mut CpuContext) -> HleResult {
    handle_call(ctx, "sceUsbdKernelDriverActive")
}
fn cb_release_interface(ctx: &mut CpuContext) -> HleResult {
    handle_call(ctx, "sceUsbdReleaseInterface")
}
fn cb_reset_device(ctx: &mut CpuContext) -> HleResult {
    handle_call(ctx, "sceUsbdResetDevice")
}
fn cb_set_configuration(ctx: &mut CpuContext) -> HleResult {
    handle_call(ctx, "sceUsbdSetConfiguration")
}

/// sceUsbd{Bulk,Interrupt}Transfer(handle, u8 endpoint, u8* data, int
/// length, int* transferred, u32 timeout): the transferred count is
/// cleared before the handle is looked up.
fn sync_transfer(ctx: &CpuContext, function: &str) -> HleResult {
    let (handle, data, length, transferred) = (ctx.rdi(), ctx.rdx(), ctx.rcx() as i32, ctx.r8());
    if handle == 0 || (length != 0 && data == 0) {
        return ERR_INVALID_ARG;
    }
    if transferred == 0 {
        log::error!("[usbd] {function}: NULL transferred output");
        return ERR_INVALID_ARG;
    }
    ctx.write_u32(transferred, 0);
    unknown_handle(function, handle)
}

fn cb_bulk_transfer(ctx: &mut CpuContext) -> HleResult {
    sync_transfer(ctx, "sceUsbdBulkTransfer")
}
fn cb_interrupt_transfer(ctx: &mut CpuContext) -> HleResult {
    sync_transfer(ctx, "sceUsbdInterruptTransfer")
}

/// sceUsbdControlTransfer(handle, u8 requestType, u8 request, u16 value,
/// u16 index, u8* data, u16 length, u32 timeout): length and timeout on the
/// stack.
fn cb_control_transfer(ctx: &mut CpuContext) -> HleResult {
    let (handle, data) = (ctx.rdi(), ctx.r9());
    let length = stack_arg(ctx, 0).unwrap_or(0) as u16;
    if handle == 0 || (length != 0 && data == 0) {
        return ERR_INVALID_ARG;
    }
    unknown_handle("sceUsbdControlTransfer", handle)
}

/// sceUsbdGetStringDescriptor(handle, u8 index, u16 languageId, u8* data,
/// int length).
fn cb_get_string_descriptor(ctx: &mut CpuContext) -> HleResult {
    let (handle, data, length) = (ctx.rdi(), ctx.rcx(), ctx.r8() as i32);
    if handle == 0 || data == 0 || length < 1 {
        return ERR_INVALID_ARG;
    }
    unknown_handle("sceUsbdGetStringDescriptor", handle)
}

/// sceUsbdClose(handle): NULL is ignored.
fn cb_close(ctx: &mut CpuContext) -> HleResult {
    if ctx.rdi() != 0 {
        unknown_handle("sceUsbdClose", ctx.rdi());
    }
    0
}

/// sceUsbdOpen(device, handle*): a NULL device is a title bug, a NULL
/// output is invalid; the output is cleared before the device is looked up.
fn cb_open(ctx: &mut CpuContext) -> HleResult {
    let (device, out) = (ctx.rdi(), ctx.rsi());
    if device == 0 {
        return unknown_device("sceUsbdOpen", 0);
    }
    if out == 0 {
        return ERR_INVALID_ARG;
    }
    ctx.write_u64(out, 0);
    unknown_device("sceUsbdOpen", device)
}

/// sceUsbdRefDevice(device) -> the device; NULL for NULL.
fn cb_ref_device(ctx: &mut CpuContext) -> HleResult {
    if ctx.rdi() != 0 {
        unknown_device("sceUsbdRefDevice", ctx.rdi());
    }
    0
}

/// sceUsbdUnrefDevice(device): NULL is ignored.
fn cb_unref_device(ctx: &mut CpuContext) -> HleResult {
    if ctx.rdi() != 0 {
        unknown_device("sceUsbdUnrefDevice", ctx.rdi());
    }
    0
}

/// sceUsbdGetBusNumber / sceUsbdGetDeviceAddress(device) -> u8; 0 for NULL.
fn cb_get_bus_number(ctx: &mut CpuContext) -> HleResult {
    if ctx.rdi() != 0 {
        unknown_device("sceUsbdGetBusNumber", ctx.rdi());
    }
    0
}
fn cb_get_device_address(ctx: &mut CpuContext) -> HleResult {
    if ctx.rdi() != 0 {
        unknown_device("sceUsbdGetDeviceAddress", ctx.rdi());
    }
    0
}

/// sceUsbdGetDeviceDescriptor(device, descriptor*).
fn cb_get_device_descriptor(ctx: &mut CpuContext) -> HleResult {
    if ctx.rdi() == 0 || ctx.rsi() == 0 {
        return ERR_INVALID_ARG;
    }
    unknown_device("sceUsbdGetDeviceDescriptor", ctx.rdi())
}

/// sceUsbdGetConfigDescriptor(device, u8 index, config**).
fn cb_get_config_descriptor(ctx: &mut CpuContext) -> HleResult {
    if ctx.rdi() == 0 || ctx.rdx() == 0 {
        return ERR_INVALID_ARG;
    }
    unknown_device("sceUsbdGetConfigDescriptor", ctx.rdi())
}

/// sceUsbdGetActiveConfigDescriptor(device, config**): a NULL device is a
/// title bug (the library dereferences it), a NULL output is invalid.
fn cb_get_active_config_descriptor(ctx: &mut CpuContext) -> HleResult {
    let (device, out) = (ctx.rdi(), ctx.rsi());
    if device == 0 {
        return unknown_device("sceUsbdGetActiveConfigDescriptor", 0);
    }
    if out == 0 {
        return ERR_INVALID_ARG;
    }
    unknown_device("sceUsbdGetActiveConfigDescriptor", device)
}

const EXPORTS: &[(&str, &str, HleCallback)] = &[
    ("TOhg7P6kTH4", "sceUsbdInit", cb_ok),
    ("Fq6+0Fm55xU", "sceUsbdExit", cb_ok),
    ("+wU6CGuZcWk", "sceUsbdHandleEventsTimeout", cb_handle_events_timeout),
    ("8qB9Ar4P5nc", "sceUsbdGetDeviceList", cb_get_device_list),
    ("EQ6SCLMqzkM", "sceUsbdFreeDeviceList", cb_free_device_list),
    ("0ktE1PhzGFU", "sceUsbdAllocTransfer", cb_alloc_transfer),
    ("-sgi7EeLSO8", "sceUsbdFreeTransfer", cb_free_transfer),
    ("t3J5pXxhJlI", "sceUsbdFillInterruptTransfer", cb_fill_interrupt_transfer),
    ("-KNh1VFIzlM", "sceUsbdCancelTransfer", cb_cancel_transfer),
    ("e7gp1xhu6RI", "sceUsbdEventHandlingOk", cb_event_handling_ok),
    ("Hvd3S--n25w", "sceUsbdFreeConfigDescriptor", cb_free_config_descriptor),
    ("BKMEGvfCPyU", "sceUsbdAttachKernelDriver", cb_attach_kernel_driver),
    ("fotb7DzeHYw", "sceUsbdBulkTransfer", cb_bulk_transfer),
    ("MlW6deWfPp0", "sceUsbdCheckConnected", cb_check_connected),
    ("AE+mHBHneyk", "sceUsbdClaimInterface", cb_claim_interface),
    ("HarYYlaFGJY", "sceUsbdClose", cb_close),
    ("RRKFcKQ1Ka4", "sceUsbdControlTransfer", cb_control_transfer),
    ("XUWtxI31YEY", "sceUsbdControlTransferGetData", cb_control_transfer_get_data),
    ("SEdQo8CFmus", "sceUsbdControlTransferGetSetup", cb_control_transfer_get_setup),
    ("oHCade-0qQ0", "sceUsbdFillBulkTransfer", cb_fill_bulk_transfer),
    ("8KrqbaaPkE0", "sceUsbdFillControlSetup", cb_fill_control_setup),
    ("7VGfMerK6m0", "sceUsbdFillControlTransfer", cb_fill_control_transfer),
    ("xqmkjHCEOSY", "sceUsbdFillIsoTransfer", cb_fill_iso_transfer),
    ("S1o1C6yOt5g", "sceUsbdGetActiveConfigDescriptor", cb_get_active_config_descriptor),
    ("t7WE9mb1TB8", "sceUsbdGetBusNumber", cb_get_bus_number),
    ("Dkm5qe8j3XE", "sceUsbdGetConfigDescriptor", cb_get_config_descriptor),
    ("GjlCrU4GcIY", "sceUsbdGetDeviceAddress", cb_get_device_address),
    ("bhomgbiQgeo", "sceUsbdGetDeviceDescriptor", cb_get_device_descriptor),
    ("vokkJ0aDf54", "sceUsbdGetIsoPacketBuffer", cb_get_iso_packet_buffer),
    ("g2oYm1DitDg", "sceUsbdGetStringDescriptor", cb_get_string_descriptor),
    ("rxi1nCOKWc8", "sceUsbdInterruptTransfer", cb_interrupt_transfer),
    ("RLf56F-WjKQ", "sceUsbdKernelDriverActive", cb_kernel_driver_active),
    ("VJ6oMq-Di2U", "sceUsbdOpen", cb_open),
    ("U1t1SoJvV-A", "sceUsbdRefDevice", cb_ref_device),
    ("REfUTmTchMw", "sceUsbdReleaseInterface", cb_release_interface),
    ("hvMn0QJXj5g", "sceUsbdResetDevice", cb_reset_device),
    ("FhU9oYrbXoA", "sceUsbdSetConfiguration", cb_set_configuration),
    ("dJxro8Nzcjk", "sceUsbdSetIsoPacketLengths", cb_set_iso_packet_lengths),
    ("L0EHgZZNVas", "sceUsbdSubmitTransfer", cb_submit_transfer),
    ("OULgIo1zAsA", "sceUsbdUnrefDevice", cb_unref_device),
];

pub fn register(reg: &mut HleRegistry) {
    for &(nid, name, cb) in EXPORTS {
        reg.register_fn(nid, name, "libSceUsbd", cb);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::CpuRegister::*;

    fn setup(base: u64) -> (ps5vm::VirtualMemory, u64) {
        let vm = ps5vm::VirtualMemory::new();
        vm.map(base, 0x2000, ps5vm::MemProtect::RW).unwrap();
        (vm, base)
    }

    fn call(ctx: &mut CpuContext, cb: HleCallback, args: &[u64]) -> HleResult {
        for (r, v) in [Rdi, Rsi, Rdx, Rcx, R8, R9].into_iter().zip(args) {
            ctx.set(r, *v);
        }
        cb(ctx)
    }

    #[test]
    fn nids_match_names() {
        for &(nid, name, _) in EXPORTS {
            assert_eq!(ps5nid::compute_nid(name), nid, "NID mismatch for {name}");
        }
    }

    /// The device list is a NULL-terminated array of zero devices that
    /// FreeDeviceList releases; a NULL output is invalid (reference
    /// GuestUsbd).
    #[test]
    fn device_list_is_empty_and_null_terminated() {
        let (vm, base) = setup(0x5DA3_0000_0000);
        let mut ctx = CpuContext::new(&vm);
        ctx.write_u64(base, 0x1234);
        assert_eq!(call(&mut ctx, cb_get_device_list, &[base]), 0);
        let list = ctx.read_u64(base).unwrap();
        assert_ne!(list, 0);
        assert_eq!(ctx.read_u64(list), Some(0));
        assert_eq!(call(&mut ctx, cb_free_device_list, &[list, 1]), 0);
        assert!(!LISTS.lock().unwrap().as_ref().unwrap().contains(&list), "the list is released");
        assert_eq!(call(&mut ctx, cb_free_device_list, &[0, 1]), 0);
        assert_eq!(call(&mut ctx, cb_get_device_list, &[0]), ERR_INVALID_ARG);
    }

    /// NULL arguments get the codes the console library returns; any other
    /// device or handle was never handed out and is refused (reference
    /// GuestUsbd without a device).
    #[test]
    fn null_and_foreign_devices_and_handles() {
        let (vm, base) = setup(0x5DA4_0000_0000);
        let mut ctx = CpuContext::new(&vm);
        let foreign = base + 0x100;
        let out = base + 0x200;
        ctx.set(Rsp, base + 0x800);
        ctx.write_u64(out, 0xAAAA);
        assert_eq!(call(&mut ctx, cb_open, &[foreign, 0]), ERR_INVALID_ARG, "null handle output");
        assert_eq!(call(&mut ctx, cb_open, &[0, out]), ERR_NO_DEVICE, "null device");
        assert_eq!(call(&mut ctx, cb_open, &[foreign, out]), ERR_NO_DEVICE, "never listed");
        assert_eq!(ctx.read_u64(out), Some(0), "the output is cleared first");
        assert_eq!(call(&mut ctx, cb_close, &[0]), 0);
        assert_eq!(call(&mut ctx, cb_ref_device, &[0]), 0);
        assert_eq!(call(&mut ctx, cb_ref_device, &[foreign]), 0);
        assert_eq!(call(&mut ctx, cb_unref_device, &[0]), 0);
        assert_eq!(call(&mut ctx, cb_get_bus_number, &[0]), 0);
        assert_eq!(call(&mut ctx, cb_get_device_address, &[0]), 0);
        let desc = base + 0x300;
        assert_eq!(call(&mut ctx, cb_get_device_descriptor, &[0, desc]), ERR_INVALID_ARG);
        assert_eq!(call(&mut ctx, cb_get_device_descriptor, &[foreign, 0]), ERR_INVALID_ARG);
        assert_eq!(call(&mut ctx, cb_get_device_descriptor, &[foreign, desc]), ERR_NO_DEVICE);
        assert_eq!(call(&mut ctx, cb_get_config_descriptor, &[0, 0, out]), ERR_INVALID_ARG);
        assert_eq!(call(&mut ctx, cb_get_config_descriptor, &[foreign, 0, 0]), ERR_INVALID_ARG);
        assert_eq!(call(&mut ctx, cb_get_config_descriptor, &[foreign, 0, out]), ERR_NO_DEVICE);
        assert_eq!(call(&mut ctx, cb_get_active_config_descriptor, &[0, out]), ERR_NO_DEVICE, "null device");
        assert_eq!(call(&mut ctx, cb_get_active_config_descriptor, &[foreign, 0]), ERR_INVALID_ARG);
        for cb in [cb_claim_interface, cb_release_interface, cb_set_configuration, cb_reset_device, cb_check_connected, cb_kernel_driver_active, cb_attach_kernel_driver] {
            assert_eq!(call(&mut ctx, cb, &[0, 1]), ERR_INVALID_ARG);
            assert_eq!(call(&mut ctx, cb, &[foreign, 1]), ERR_NO_DEVICE);
        }
        // Control transfer: length (18) and timeout on the stack.
        ctx.write_u64(base + 0x808, 18);
        ctx.write_u64(base + 0x810, 1000);
        assert_eq!(call(&mut ctx, cb_control_transfer, &[0, 0x80, 6, 0x100, 0, desc]), ERR_INVALID_ARG);
        assert_eq!(call(&mut ctx, cb_control_transfer, &[foreign, 0x80, 6, 0x100, 0, 0]), ERR_INVALID_ARG, "no data");
        assert_eq!(call(&mut ctx, cb_control_transfer, &[foreign, 0x80, 6, 0x100, 0, desc]), ERR_NO_DEVICE);
        ctx.write_u64(base + 0x808, 0);
        assert_eq!(call(&mut ctx, cb_control_transfer, &[foreign, 0x80, 6, 0x100, 0, 0]), ERR_NO_DEVICE, "no data for a zero length");
        assert_eq!(call(&mut ctx, cb_get_string_descriptor, &[0, 1, 0x409, desc, 18]), ERR_INVALID_ARG);
        assert_eq!(call(&mut ctx, cb_get_string_descriptor, &[foreign, 1, 0x409, desc, 0]), ERR_INVALID_ARG, "empty buffer");
        assert_eq!(call(&mut ctx, cb_get_string_descriptor, &[foreign, 1, 0x409, desc, 18]), ERR_NO_DEVICE);
    }

    /// Bulk/interrupt transfers: NULL handle or data is invalid, the
    /// transferred count is cleared before the handle is looked up.
    #[test]
    fn synchronous_transfers() {
        let (vm, base) = setup(0x5DA5_0000_0000);
        let mut ctx = CpuContext::new(&vm);
        let (foreign, data, done) = (base + 0x100, base + 0x200, base + 0x300);
        for cb in [cb_bulk_transfer, cb_interrupt_transfer] {
            ctx.write_u32(done, 77);
            assert_eq!(call(&mut ctx, cb, &[0, 0x81, data, 8, done, 100]), ERR_INVALID_ARG);
            assert_eq!(call(&mut ctx, cb, &[foreign, 0x81, 0, 8, done, 100]), ERR_INVALID_ARG);
            assert_eq!(ctx.read_u32(done), Some(77));
            assert_eq!(call(&mut ctx, cb, &[foreign, 0x81, data, 8, 0, 100]), ERR_INVALID_ARG, "null transferred");
            assert_eq!(call(&mut ctx, cb, &[foreign, 0x81, data, 8, done, 100]), ERR_NO_DEVICE);
            assert_eq!(ctx.read_u32(done), Some(0));
        }
    }

    /// The no-device transfer path: a zeroed transfer with its iso packet
    /// count, the interrupt and bulk fields filled at the libusb offsets,
    /// cancel is NOT_FOUND, submit needs a handle (reference GuestUsbd).
    #[test]
    fn transfers_without_a_device() {
        let (vm, base) = setup(0x7C62_0000_0000);
        let mut ctx = CpuContext::new(&vm);
        let t = call(&mut ctx, cb_alloc_transfer, &[2]) as u64;
        assert_ne!(t, 0);
        assert_eq!(ctx.read_u32(t + 56), Some(2));
        assert_eq!(ctx.read_bytes(t, 56).unwrap(), vec![0; 56]);
        assert_eq!(call(&mut ctx, cb_alloc_transfer, &[(-1i64) as u64]), 0);
        let rsp = base + 0x100;
        ctx.write_u64(rsp + 8, 0xDA7A);
        ctx.write_u64(rsp + 16, 250);
        ctx.set(Rsp, rsp);
        assert_eq!(call(&mut ctx, cb_fill_interrupt_transfer, &[t, 0, 0x81, base + 0x200, 8, 0xCA11]), 0);
        assert_eq!(ctx.read_bytes(t + 9, 2).unwrap(), vec![0x81, TYPE_INTERRUPT]);
        assert_eq!((ctx.read_u32(t + 12), ctx.read_u32(t + 20)), (Some(250), Some(8)));
        assert_eq!((ctx.read_u64(t + 32), ctx.read_u64(t + 40), ctx.read_u64(t + 48)), (Some(0xCA11), Some(0xDA7A), Some(base + 0x200)));
        assert_eq!(call(&mut ctx, cb_cancel_transfer, &[t]), ERR_NOT_FOUND);
        assert_eq!(call(&mut ctx, cb_submit_transfer, &[0]), ERR_INVALID_ARG, "null transfer");
        assert_eq!(call(&mut ctx, cb_submit_transfer, &[t]), ERR_INVALID_ARG, "no handle");
        assert_eq!(call(&mut ctx, cb_fill_bulk_transfer, &[t, base + 0x400, 0x02, base + 0x200, 16, 0xCA12]), 0);
        assert_eq!(ctx.read_bytes(t + 9, 2).unwrap(), vec![0x02, TYPE_BULK]);
        assert_eq!((ctx.read_u64(t), ctx.read_u32(t + 20)), (Some(base + 0x400), Some(16)));
        assert_eq!(call(&mut ctx, cb_submit_transfer, &[t]), ERR_NO_DEVICE, "a handle never opened");
        assert_eq!(call(&mut ctx, cb_free_transfer, &[t]), 0);
        assert_eq!(call(&mut ctx, cb_free_transfer, &[0]), 0);
        assert_eq!(cb_event_handling_ok(&mut ctx), 1);
        assert_eq!(call(&mut ctx, cb_free_config_descriptor, &[0]), 0);
    }

    /// The setup packet is little endian; a control transfer's length is
    /// the setup packet plus wLength and its data follows the setup.
    #[test]
    fn control_setup_and_transfer() {
        let (vm, base) = setup(0x7C63_0000_0000);
        let mut ctx = CpuContext::new(&vm);
        let buf = base + 0x100;
        assert_eq!(call(&mut ctx, cb_fill_control_setup, &[buf, 0x80, 6, 0x0100, 0x0409, 18]), 0);
        assert_eq!(ctx.read_bytes(buf, 8).unwrap(), vec![0x80, 6, 0x00, 0x01, 0x09, 0x04, 18, 0]);
        let t = base + 0x400;
        ctx.write_bytes(t, &[0xFF; 60]);
        assert_eq!(call(&mut ctx, cb_fill_control_transfer, &[t, 0x1111, buf, 0xCA11, 0xDA7A, 500]), 0);
        assert_eq!(ctx.read_bytes(t + 9, 2).unwrap(), vec![0, TYPE_CONTROL]);
        assert_eq!((ctx.read_u32(t + 12), ctx.read_u32(t + 20)), (Some(500), Some(8 + 18)));
        assert_eq!((ctx.read_u64(t), ctx.read_u64(t + 32), ctx.read_u64(t + 40), ctx.read_u64(t + 48)), (Some(0x1111), Some(0xCA11), Some(0xDA7A), Some(buf)));
        assert_eq!(call(&mut ctx, cb_control_transfer_get_setup, &[t]) as u64, buf);
        assert_eq!(call(&mut ctx, cb_control_transfer_get_data, &[t]) as u64, buf + 8);
        assert_eq!(call(&mut ctx, cb_fill_control_transfer, &[t, 0x1111, 0, 0, 0, 0]), 0);
        assert_eq!(ctx.read_u32(t + 20), Some(0), "no buffer, no length");
        assert_eq!(call(&mut ctx, cb_control_transfer_get_data, &[t]), 0);
        assert_eq!(call(&mut ctx, cb_control_transfer_get_setup, &[0]), 0);
    }

    /// Iso fill stores the packet count; packet lengths are set for every
    /// descriptor and a packet's buffer follows the earlier packets.
    #[test]
    fn iso_transfer_packets() {
        let (vm, base) = setup(0x7C64_0000_0000);
        let mut ctx = CpuContext::new(&vm);
        let t = call(&mut ctx, cb_alloc_transfer, &[3]) as u64;
        let rsp = base + 0x100;
        ctx.write_u64(rsp + 8, 0xCA11);
        ctx.write_u64(rsp + 16, 0xDA7A);
        ctx.write_u64(rsp + 24, 40);
        ctx.set(Rsp, rsp);
        let buf = base + 0x800;
        assert_eq!(call(&mut ctx, cb_fill_iso_transfer, &[t, 0x2222, 0x83, buf, 3 * 64, 3]), 0);
        assert_eq!(ctx.read_bytes(t + 9, 2).unwrap(), vec![0x83, TYPE_ISOCHRONOUS]);
        assert_eq!((ctx.read_u32(t + 12), ctx.read_u32(t + 56)), (Some(40), Some(3)));
        assert_eq!((ctx.read_u64(t + 32), ctx.read_u64(t + 40)), (Some(0xCA11), Some(0xDA7A)));
        assert_eq!(call(&mut ctx, cb_set_iso_packet_lengths, &[t, 64]), 0);
        for p in 0..3 {
            assert_eq!(ctx.read_u32(t + 60 + 12 * p), Some(64));
        }
        ctx.write_u32(t + 60 + 12, 100);
        assert_eq!(call(&mut ctx, cb_get_iso_packet_buffer, &[t, 0]) as u64, buf);
        assert_eq!(call(&mut ctx, cb_get_iso_packet_buffer, &[t, 1]) as u64, buf + 64);
        assert_eq!(call(&mut ctx, cb_get_iso_packet_buffer, &[t, 2]) as u64, buf + 164);
        assert_eq!(call(&mut ctx, cb_get_iso_packet_buffer, &[t, 3]), 0, "out of range");
        ctx.write_u64(t + 48, 0);
        assert_eq!(call(&mut ctx, cb_get_iso_packet_buffer, &[t, 0]), 0, "no buffer");
        crate::libc::heap_release(t);
    }
}
