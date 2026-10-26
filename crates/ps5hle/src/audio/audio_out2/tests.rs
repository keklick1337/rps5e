// RustPS5Emu (RPS5E)
// Copyright (C) 2026 Vladislav Tislenko aka keklick1337
// SPDX-License-Identifier: GPL-2.0-only

//! Ports, users, gains, object mixing and queue behaviour against a fake
//! AudioOut that records every batch.

use super::*;
use std::collections::HashMap;
use std::sync::{Arc, Condvar};

#[derive(Default)]
struct Devices {
    live: Vec<i32>,
    device_backed: Vec<i32>,
    output_blocking: Vec<bool>,
    output_pcm: Vec<Vec<u8>>,
    output_gains: Vec<[f32; 2]>,
    capture_bytes: usize,
    next_device: i32,
    open_waiters: usize,
    block_opens: bool,
}

#[derive(Default)]
struct Fake {
    devices: Mutex<Devices>,
    cv: Condvar,
    now: AtomicU64,
}

impl Backend for Arc<Fake> {
    fn open(&self, port_type: i32, _samples_num: u32, _freq: u32, _format: Format) -> i32 {
        let mut d = locked(&self.devices);
        d.next_device += 1;
        let handle = d.next_device;
        d.live.push(handle);
        if port_type != audio_out::PORT_TYPE_VIBRATION {
            d.device_backed.push(handle);
        }
        d.open_waiters += 1;
        self.cv.notify_all();
        while d.block_opens {
            d = self.cv.wait(d).unwrap();
        }
        handle
    }

    fn close(&self, handle: i32) {
        let mut d = locked(&self.devices);
        d.live.retain(|&h| h != handle);
        d.device_backed.retain(|&h| h != handle);
    }

    fn has_device(&self, handle: i32) -> bool {
        locked(&self.devices).device_backed.contains(&handle)
    }

    fn outputs(&self, params: &[OutputParam], blocking: bool) -> u32 {
        let mut d = locked(&self.devices);
        d.output_blocking.push(blocking);
        d.output_gains.clear();
        if d.capture_bytes != 0 {
            for p in params {
                let pcm = p.data.unwrap()[..d.capture_bytes].to_vec();
                d.output_pcm.push(pcm);
                d.output_gains.push(p.gains.map_or([1.0, 1.0], |g| [g[0], g[1]]));
            }
        }
        0
    }

    fn now_us(&self) -> u64 {
        self.now.fetch_add(1000, Ordering::Relaxed)
    }

    fn sleep_us(&self, us: u64) {
        self.now.fetch_add(us, Ordering::Relaxed);
    }
}

/// AudioOut2 with a fake output and a scratch "guest memory" for PCM.
struct Harness {
    out2: AudioOut2<Arc<Fake>>,
    fake: Arc<Fake>,
    memory: Mutex<HashMap<u64, Vec<u8>>>,
    user: u64,
}

const STEREO_FLOAT: u32 = 0x200;

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|s| s.to_le_bytes()).collect()
}

fn i16_bytes(v: &[i16]) -> Vec<u8> {
    v.iter().flat_map(|s| s.to_le_bytes()).collect()
}

fn pcm_attribute(address: u64) -> Attribute {
    Attribute::new(PORT_ATTRIBUTE_ID_PCM, 1, 8, &address.to_le_bytes())
}

fn value_attribute(id: u32, bytes: Vec<u8>) -> Attribute {
    Attribute::new(id, 1, bytes.len() as u64, &bytes)
}

fn check_samples(pcm: &[u8], frame: &[f32], message: &str) {
    assert!(!pcm.is_empty() && pcm.len() % (frame.len() * 4) == 0, "incorrect PCM size: {message}");
    for (i, c) in pcm.chunks_exact(4).enumerate() {
        let sample = f32::from_le_bytes(c.try_into().unwrap());
        assert!((sample - frame[i % frame.len()]).abs() < 0.000001, "{message}: sample {i} = {sample}");
    }
}

impl Harness {
    fn new() -> Self {
        let fake = Arc::new(Fake::default());
        let out2 = AudioOut2::new(fake.clone());
        let mut user = 0;
        assert_eq!(out2.user_create(Some(&mut user)), 0, "test user create failed");
        Self { out2, fake, memory: Mutex::new(HashMap::new()), user }
    }

    fn param(&self, data_format: u32) -> PortParam {
        PortParam { port_type: 0, data_format, sampling_freq: 48000, user_handle: self.user }
    }

    fn context(&self, queue_depth: u32, max_object_ports: u32) -> u64 {
        self.out2
            .context_create(&ContextParam { max_object_ports, queue_depth, num_grains: 512 })
            .expect("context create failed")
    }

    fn port(&self, context: u64, param: &PortParam) -> u64 {
        self.out2.port_create(context, param).expect("port create failed")
    }

    fn write(&self, address: u64, data: Vec<u8>) {
        locked(&self.memory).insert(address, data);
    }

    fn set(&self, port: u64, attributes: &[Attribute]) -> HleResult {
        let memory = locked(&self.memory).clone();
        self.out2.port_set_attributes(port, attributes, &mut |address, buffer| {
            match memory.get(&address).filter(|v| v.len() >= buffer.len()) {
                Some(v) => {
                    buffer.copy_from_slice(&v[..buffer.len()]);
                    true
                }
                None => false,
            }
        })
    }

    fn set_pcm(&self, port: u64, address: u64) {
        assert_eq!(self.set(port, &[pcm_attribute(address)]), 0, "setting PCM failed");
    }

    fn set_attribute(&self, port: u64, id: u32, bytes: Vec<u8>) {
        assert_eq!(self.set(port, &[value_attribute(id, bytes)]), 0, "setting attribute failed");
    }

    fn devices(&self) -> std::sync::MutexGuard<'_, Devices> {
        locked(&self.fake.devices)
    }

    fn live_devices(&self) -> usize {
        self.devices().live.len()
    }

    fn reset_output_calls(&self) {
        self.devices().output_blocking.clear();
    }

    fn output_calls(&self) -> Vec<bool> {
        self.devices().output_blocking.clone()
    }

    fn push_pcm(&self, context: u64, bytes: usize) -> Vec<Vec<u8>> {
        {
            let mut d = self.devices();
            d.capture_bytes = bytes;
            d.output_pcm.clear();
        }
        assert_eq!(self.out2.context_push(context, true), 0, "PCM push failed");
        let mut d = self.devices();
        d.capture_bytes = 0;
        std::mem::take(&mut d.output_pcm)
    }

    fn block_device_opens(&self) {
        let mut d = self.devices();
        d.open_waiters = 0;
        d.block_opens = true;
    }

    fn wait_for_device_opens(&self, count: usize) {
        let mut d = self.devices();
        while d.open_waiters < count {
            d = self.fake.cv.wait(d).unwrap();
        }
    }

    fn release_device_opens(&self) {
        self.devices().block_opens = false;
        self.fake.cv.notify_all();
    }

    fn finish(self) {
        assert_eq!(self.out2.user_destroy(self.user), 0, "test user destroy failed");
    }
}

#[test]
fn user_supported_attributes() {
    let h = Harness::new();
    assert_eq!(h.out2.user_create(None), AUDIO_OUT2_ERROR_INVALID_PARAM, "null user handle output was accepted");
    let mut user = 0;
    assert_eq!(h.out2.user_create(Some(&mut user)), 0, "user create failed");
    let (mut context_attributes, mut port_attributes) = (u32::MAX, u32::MAX);
    assert_eq!(h.out2.user_get_supported_attributes(user, Some(&mut context_attributes), Some(&mut port_attributes)), 0);
    assert!(
        context_attributes == 0 && port_attributes == 0x103,
        "user capabilities do not match implemented attributes"
    );

    context_attributes = u32::MAX;
    port_attributes = u32::MAX;
    assert_eq!(h.out2.user_get_supported_attributes(user, None, Some(&mut port_attributes)), AUDIO_OUT2_ERROR_INVALID_PARAM);
    assert_eq!(
        h.out2.user_get_supported_attributes(user, Some(&mut context_attributes), None),
        AUDIO_OUT2_ERROR_INVALID_PARAM
    );
    assert!(
        context_attributes == u32::MAX && port_attributes == u32::MAX,
        "null capability outputs were accepted or modified"
    );
    for invalid in [0u64, u64::MAX] {
        assert_eq!(
            h.out2.user_get_supported_attributes(invalid, Some(&mut context_attributes), Some(&mut port_attributes)),
            AUDIO_OUT2_ERROR_INVALID_PARAM
        );
        assert!(
            context_attributes == u32::MAX && port_attributes == u32::MAX,
            "invalid user capabilities succeeded or modified outputs"
        );
    }

    let context = h.context(4, 0);
    let mut param = h.param(STEREO_FLOAT);
    param.user_handle = user;
    assert!(h.out2.port_create(context, &param).is_ok(), "user port create failed");
    assert_eq!(h.out2.user_destroy(user), AUDIO_OUT2_ERROR_BUSY, "user was destroyed while owning a port");
    h.out2.context_destroy(context);
    assert_eq!(h.out2.user_destroy(user), 0, "unused user destroy failed");
    assert_eq!(
        h.out2.user_get_supported_attributes(user, Some(&mut context_attributes), Some(&mut port_attributes)),
        AUDIO_OUT2_ERROR_INVALID_PARAM
    );
    assert!(context_attributes == u32::MAX && port_attributes == u32::MAX);
    assert_eq!(h.out2.user_destroy(user), AUDIO_OUT2_ERROR_INVALID_PARAM, "destroyed user remained valid");

    let new_context = h.context(4, 0);
    assert_eq!(
        h.out2.port_create(new_context, &param).err(),
        Some(AUDIO_OUT2_ERROR_INVALID_PARAM),
        "destroyed user acquired a port"
    );
    h.out2.context_destroy(new_context);
    h.finish();
}

#[test]
fn slot_reuse() {
    let h = Harness::new();
    let context = h.context(4, 0);
    let param = h.param(STEREO_FLOAT);
    for _ in 0..300 {
        let port = h.out2.port_create(context, &param).expect("port slot was not reusable");
        assert_ne!(port, 0, "port handle is zero");
        h.out2.port_destroy(port);
    }
    h.out2.context_destroy(context);
    h.finish();
}

#[test]
fn full_table_recovers() {
    let h = Harness::new();
    let context = h.context(4, 0);
    let param = h.param(STEREO_FLOAT);
    let ports: Vec<u64> = (0..256).map(|_| h.out2.port_create(context, &param).expect("port table filled early")).collect();
    assert_eq!(
        h.out2.port_create(context, &param).err(),
        Some(AUDIO_OUT2_ERROR_PORT_FULL),
        "full port table accepted another port"
    );
    for port in ports {
        h.out2.port_destroy(port);
    }
    let port = h.out2.port_create(context, &param).expect("port table did not recover");
    h.out2.port_destroy(port);
    h.out2.context_destroy(context);
    h.finish();
}

#[test]
fn concurrent_creates() {
    const THREADS: usize = 8;
    let h = Harness::new();
    let context = h.context(4, 0);
    let param = h.param(0x800);
    h.block_device_opens();
    let results: Vec<Result<u64, HleResult>> = std::thread::scope(|s| {
        let threads: Vec<_> = (0..THREADS).map(|_| s.spawn(|| h.out2.port_create(context, &param))).collect();
        h.wait_for_device_opens(THREADS);
        h.release_device_opens();
        threads.into_iter().map(|t| t.join().unwrap()).collect()
    });
    for r in results {
        let port = r.expect("concurrent port create failed");
        assert_eq!(h.out2.port_channels(port), Some(8), "concurrent create lost its reserved slot");
        h.out2.port_destroy(port);
    }
    assert_eq!(h.live_devices(), 0, "concurrent create leaked a device");
    h.out2.context_destroy(context);
    h.finish();
}

#[test]
fn context_destroy_cancels_pending_create() {
    let h = Harness::new();
    let context = h.context(4, 0);
    let param = h.param(STEREO_FLOAT);
    h.block_device_opens();
    let result = std::thread::scope(|s| {
        let creator = s.spawn(|| h.out2.port_create(context, &param));
        h.wait_for_device_opens(1);
        h.out2.context_destroy(context);
        h.release_device_opens();
        creator.join().unwrap()
    });
    assert!(result.is_err(), "destroyed context retained a pending port create");
    assert_eq!(h.live_devices(), 0, "cancelled port create leaked a device");
    h.finish();
}

#[test]
fn synchronous_device_push_bypasses_modelled_queue() {
    let h = Harness::new();
    let context = h.context(1, 0);
    let port = h.port(context, &h.param(STEREO_FLOAT));
    h.write(0x1000, vec![0u8; 512 * 2 * 4]);
    h.set_pcm(port, 0x1000);
    h.reset_output_calls();
    assert_eq!(h.out2.context_push(context, true), 0, "first sync push failed");
    assert_eq!(h.out2.context_push(context, true), 0, "device-paced sync push was blocked by modelled queue");
    assert_eq!(h.output_calls(), vec![true, true], "sync pushes did not reach the device backend blocking");
    h.out2.port_destroy(port);
    h.out2.context_destroy(context);
    h.finish();
}

#[test]
fn float_12_channel_port_outputs_pcm() {
    let h = Harness::new();
    let context = h.context(4, 0);
    let port = h.port(context, &h.param(0x0C00));
    assert_eq!(h.live_devices(), 1, "12-channel port did not open an audio device");
    assert_eq!(h.out2.port_channels(port), Some(12), "12-channel port lost its guest channel count");
    h.write(0x1000, vec![0u8; 512 * 12 * 4]);
    h.set_pcm(port, 0x1000);
    h.reset_output_calls();
    assert_eq!(h.out2.context_push(context, true), 0, "12-channel PCM push failed");
    assert_eq!(h.output_calls(), vec![true], "12-channel PCM did not reach the device backend");
    h.out2.port_destroy(port);
    assert_eq!(h.live_devices(), 0, "12-channel port leaked its audio device");
    h.out2.context_destroy(context);
    h.finish();
}

#[test]
fn asynchronous_device_push_keeps_queue_bounded() {
    let h = Harness::new();
    let context = h.context(1, 0);
    let port = h.port(context, &h.param(STEREO_FLOAT));
    h.write(0x1000, vec![0u8; 512 * 2 * 4]);
    h.set_pcm(port, 0x1000);
    h.reset_output_calls();
    assert_eq!(h.out2.context_push(context, false), 0, "first async push failed");
    assert_eq!(
        h.out2.context_push(context, false),
        AUDIO_OUT2_ERROR_NOT_READY,
        "full async queue accepted another buffer"
    );
    assert_eq!(h.output_calls(), vec![false], "rejected async push reached the device backend");
    let (mut queued, mut available) = (0, 0);
    h.out2.context_get_queue_level(context, Some(&mut queued), Some(&mut available));
    assert!(queued == 1 && available == 0, "async queue level does not match accepted pushes");
    h.out2.port_destroy(port);
    h.out2.context_destroy(context);
    h.finish();
}

/// Advanced grains not pushed yet count toward the queue level (capped at
/// the depth); a push consumes one.
#[test]
fn advanced_grains_fill_the_queue_level() {
    let h = Harness::new();
    let context = h.context(4, 0);
    let port = h.port(context, &h.param(STEREO_FLOAT));
    let level = |h: &Harness| {
        let (mut queued, mut available) = (0, 0);
        h.out2.context_get_queue_level(context, Some(&mut queued), Some(&mut available));
        (queued, available)
    };
    assert_eq!(level(&h), (0, 4));
    for _ in 0..3 {
        h.out2.context_advance(context);
    }
    assert_eq!(level(&h), (3, 1));
    for _ in 0..5 {
        h.out2.context_advance(context);
    }
    assert_eq!(level(&h), (4, 0), "capped at the depth");
    h.write(0x1000, vec![0u8; 512 * 2 * 4]);
    h.set_pcm(port, 0x1000);
    assert_eq!(h.out2.context_push(context, false), 0);
    let (queued, _) = level(&h);
    assert_eq!(queued, 4);
    h.out2.port_destroy(port);
    h.out2.context_destroy(context);
    h.finish();
}

/// A burst of 64 grains set and advanced before their pushes plays every
/// grain, in order.
#[test]
fn advance_burst_keeps_every_grain() {
    let h = Harness::new();
    let context = h.context(4, 0);
    let port = h.port(context, &h.param(STEREO_FLOAT));
    for i in 0..64u64 {
        let address = 0x10_0000 + i * 0x1_0000;
        h.write(address, f32_bytes(&vec![i as f32; 512 * 2]));
        h.set_pcm(port, address);
        h.out2.context_advance(context);
    }
    for i in 0..64 {
        let pcm = h.push_pcm(context, 8);
        assert_eq!(pcm.len(), 1);
        check_samples(&pcm[0], &[i as f32, i as f32], "burst grain");
    }
    h.out2.port_destroy(port);
    h.out2.context_destroy(context);
    h.finish();
}

#[test]
fn handle_without_pcm_does_not_bypass_queue() {
    let h = Harness::new();
    let context = h.context(1, 0);
    let port = h.port(context, &h.param(STEREO_FLOAT));
    h.reset_output_calls();
    assert_eq!(h.out2.context_push(context, true), 0, "empty sync push failed");
    assert_eq!(
        h.out2.context_push(context, false),
        AUDIO_OUT2_ERROR_NOT_READY,
        "handle without PCM bypassed queue backpressure"
    );
    assert!(h.output_calls().is_empty(), "empty push reached the device backend");
    h.out2.port_destroy(port);
    h.out2.context_destroy(context);
    h.finish();
}

#[test]
fn pcm_copied_before_scratch_buffer_reuse() {
    let h = Harness::new();
    let context = h.context(4, 0);
    let first = h.port(context, &h.param(STEREO_FLOAT));
    let second = h.port(context, &h.param(STEREO_FLOAT));
    let first_pcm = f32_bytes(&[0.25; 1024]);
    let second_pcm = f32_bytes(&[-0.5; 1024]);
    h.write(0x1000, first_pcm.clone());
    h.set_pcm(first, 0x1000);
    h.write(0x1000, second_pcm.clone());
    h.set_pcm(second, 0x1000);
    h.write(0x1000, vec![0u8; 4096]);

    let output = h.push_pcm(context, 4096);
    assert_eq!(output.len(), 2, "shared-buffer push did not output both ports");
    assert_eq!(output[0], first_pcm, "first port lost PCM when scratch buffer was reused");
    assert_eq!(output[1], second_pcm, "second port lost PCM when scratch buffer was reused");
    h.out2.port_destroy(first);
    h.out2.port_destroy(second);
    h.out2.context_destroy(context);
    h.finish();
}

#[test]
fn port_gain_and_validation() {
    let h = Harness::new();
    let context = h.context(4, 0);
    let port = h.port(context, &h.param(STEREO_FLOAT));
    h.write(0x1000, f32_bytes(&[0.25; 1024]));
    h.set_pcm(port, 0x1000);
    let check_output = |gain: [f32; 2], expected: f32| {
        check_samples(&h.push_pcm(context, 4096)[0], &[expected], "gain modified stored PCM");
        assert_eq!(h.devices().output_gains, vec![gain], "incorrect per-channel gain forwarded to backend");
    };
    check_output([1.0, 1.0], 0.25);
    for gain in [[2.0, 0.5], [0.0, 0.0], [1.0, 1.0]] {
        h.set_attribute(port, PORT_ATTRIBUTE_ID_GAIN, f32_bytes(&gain));
        check_output(gain, 0.25);
        check_output(gain, 0.25);
    }

    let gain = [0.5f32, 0.25];
    h.set_attribute(port, PORT_ATTRIBUTE_ID_GAIN, f32_bytes(&gain));
    h.write(0x1000, f32_bytes(&[0.75; 1024]));
    h.set_pcm(port, 0x1000);
    check_output(gain, 0.75);

    h.write(0x2000, f32_bytes(&[-0.25; 1024]));
    for invalid in [-1.0f32, f32::INFINITY, f32::NAN] {
        let attributes = [pcm_attribute(0x2000), value_attribute(PORT_ATTRIBUTE_ID_GAIN, f32_bytes(&[1.0, invalid]))];
        assert_eq!(h.set(port, &attributes), AUDIO_OUT2_ERROR_INVALID_PARAM, "invalid gain was accepted");
        check_output(gain, 0.75);
    }
    let null_value = Attribute::new(PORT_ATTRIBUTE_ID_GAIN, 0, 8, &[]);
    let short_value = value_attribute(PORT_ATTRIBUTE_ID_GAIN, f32_bytes(&gain[..1]));
    for attribute in [null_value, short_value] {
        assert_eq!(h.set(port, &[attribute]), AUDIO_OUT2_ERROR_INVALID_PARAM, "malformed gain was accepted");
    }
    let duplicates = [
        pcm_attribute(0x2000),
        value_attribute(PORT_ATTRIBUTE_ID_GAIN, f32_bytes(&gain)),
        value_attribute(PORT_ATTRIBUTE_ID_GAIN, f32_bytes(&gain)),
    ];
    assert_eq!(h.set(port, &duplicates), AUDIO_OUT2_ERROR_INVALID_PARAM, "duplicate gain attributes were accepted");
    check_output(gain, 0.75);
    // A gain array longer than the port's channels (a full speaker layout)
    // is accepted; the leading entries apply.
    let wide = [0.5f32, 0.75, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0];
    assert_eq!(h.set(port, &[pcm_attribute(0x1000), value_attribute(PORT_ATTRIBUTE_ID_GAIN, f32_bytes(&wide))]), 0);
    check_output([0.5, 0.75], 0.75);
    h.out2.context_destroy(context);
    h.finish();
}

#[test]
fn object_gain_and_ambisonics() {
    const HALF: f32 = 0.70710678;
    let cases: [(u32, f32, f32); 9] = [
        (AMBISONICS_NONE, HALF, HALF),
        (0, 1.0, 1.0),
        (1, 0.0, 0.0),
        (2, HALF, -HALF),
        (3, 0.0, 0.0),
        (64, HALF, HALF),
        (65, HALF, -HALF),
        (66, 0.0, 0.0),
        (99, 0.0, 0.0),
    ];
    for format in [0x100u32, 0x101] {
        let h = Harness::new();
        let context = h.context(1, 1);
        let mut param = h.param(format);
        param.port_type = 0x100;
        let object = h.port(context, &param);
        assert_eq!(h.live_devices(), 1, "object-only context did not create one output device");
        let pcm = |loud: bool| {
            if format == 0x100 {
                f32_bytes(&[if loud { 0.5 } else { 0.0 }; 512])
            } else {
                i16_bytes(&[if loud { 16384 } else { 0 }; 512])
            }
        };
        const OUTPUT_BYTES: usize = 512 * 2 * 4;
        h.write(0x1000, pcm(true));
        h.set_pcm(object, 0x1000);
        h.write(0x1000, pcm(false));
        h.set_attribute(object, PORT_ATTRIBUTE_ID_GAIN, f32_bytes(&[0.0]));
        h.set_attribute(object, PORT_ATTRIBUTE_ID_AMBISONICS, 65u32.to_le_bytes().to_vec());
        for _ in 0..2 {
            let output = h.push_pcm(context, OUTPUT_BYTES);
            assert_eq!(output.len(), 1, "object-only push did not output exactly once");
            check_samples(
                &output[0],
                &[0.5 * HALF, 0.5 * HALF],
                "object PCM ownership, default gain, or attribute freeze failed",
            );
        }
        h.write(0x1000, pcm(true));
        for (channel, left, right) in cases {
            h.out2.context_advance(context);
            let attributes = [
                pcm_attribute(0x1000),
                value_attribute(PORT_ATTRIBUTE_ID_GAIN, f32_bytes(&[0.5])),
                value_attribute(PORT_ATTRIBUTE_ID_AMBISONICS, channel.to_le_bytes().to_vec()),
            ];
            assert_eq!(h.set(object, &attributes), 0, "object attributes could not be set atomically with PCM first");
            for _ in 0..2 {
                check_samples(
                    &h.push_pcm(context, OUTPUT_BYTES)[0],
                    &[0.25 * left, 0.25 * right],
                    "object gain or ambisonics decoded incorrectly or accumulated across pushes",
                );
            }
        }
        h.out2.context_advance(context);
        h.set_attribute(object, PORT_ATTRIBUTE_ID_AMBISONICS, AMBISONICS_NONE.to_le_bytes().to_vec());
        h.set_attribute(object, PORT_ATTRIBUTE_ID_GAIN, f32_bytes(&[0.0]));
        h.set_pcm(object, 0x1000);
        check_samples(&h.push_pcm(context, OUTPUT_BYTES)[0], &[0.0], "zero gain did not mute object");
        h.out2.context_advance(context);
        h.set_pcm(object, 0x1000);
        check_samples(&h.push_pcm(context, OUTPUT_BYTES)[0], &[0.0], "object gain did not persist");
        h.out2.context_destroy(context);
        assert_eq!(h.live_devices(), 0, "object context leaked its output device");
        h.finish();
    }
}

#[test]
fn object_bus_routing() {
    let h = Harness::new();
    let context = h.context(1, 2);
    let other_context = h.context(1, 1);
    let mut other_user = 0;
    assert_eq!(h.out2.user_create(Some(&mut other_user)), 0, "second user create failed");
    let first = h.port(context, &h.param(STEREO_FLOAT));
    let second = h.port(context, &h.param(STEREO_FLOAT));
    h.write(0x1000, f32_bytes(&[0.125; 1024]));
    h.set_pcm(first, 0x1000);
    h.set_pcm(second, 0x1000);
    let mut param = h.param(0x100);
    param.port_type = 0x100;
    h.write(0x2000, f32_bytes(&[0.25; 512]));
    h.set_pcm(h.port(context, &param), 0x2000);
    param.user_handle = other_user;
    h.set_pcm(h.port(context, &param), 0x2000);
    h.write(0x2000, f32_bytes(&[1.0; 512]));
    h.set_pcm(h.port(other_context, &param), 0x2000);
    const BYTES: usize = 512 * 2 * 4;
    for _ in 0..2 {
        let output = h.push_pcm(context, BYTES);
        assert_eq!(output.len(), 3, "object bus was missing or queued more than once");
        check_samples(&output[0], &[0.125], "objects modified first bed");
        check_samples(&output[1], &[0.125], "objects modified second bed");
        check_samples(&output[2], &[0.35355339], "MAIN objects failed to mix across users or leaked contexts");
    }
    h.set_pcm(first, 0);
    h.set_pcm(second, 0);
    check_samples(&h.push_pcm(context, BYTES)[0], &[0.35355339], "object output depended on bed PCM");
    h.out2.context_destroy(context);
    assert_eq!(h.live_devices(), 1, "context destroy closed another context's object device");
    check_samples(&h.push_pcm(other_context, BYTES)[0], &[0.70710678], "other context lost its object PCM");
    h.out2.context_destroy(other_context);
    assert_eq!(h.out2.user_destroy(other_user), 0, "second user destroy failed");
    assert_eq!(h.live_devices(), 0, "object bus device leaked");
    h.finish();
}

/// MAIN port formats as measured on a console: 1, 2, 8 or 12 channels with
/// sample type 0 or 1; bit 7 without 8 channels is an invalid parameter;
/// bits 12-31 do not matter. Rejections store an invalid handle.
#[test]
fn measured_main_port_formats() {
    const OK: u32 = 0;
    const FLAG: u32 = 0x8026_8001;
    const FORMAT: u32 = 0x8026_800E;
    let cases: &[(u32, u32)] = &[
        (0x000, FORMAT), (0x001, FORMAT), (0x080, FLAG),
        (0x100, OK), (0x101, OK), (0x102, FORMAT), (0x17F, FORMAT),
        (0x180, FLAG), (0x181, FLAG), (0x1FF, FLAG),
        (0x200, OK), (0x201, OK), (0x202, FORMAT), (0x27F, FORMAT),
        (0x280, FLAG), (0x281, FLAG), (0x2FF, FLAG),
        (0x300, FORMAT), (0x301, FORMAT), (0x380, FLAG),
        (0x400, FORMAT), (0x401, FORMAT), (0x500, FORMAT),
        (0x600, FORMAT), (0x601, FORMAT), (0x680, FLAG),
        (0x700, FORMAT), (0x701, FORMAT), (0x780, FLAG),
        (0x800, OK), (0x801, OK), (0x802, FORMAT), (0x87F, FORMAT),
        (0x880, OK), (0x881, OK), (0x882, FORMAT), (0x8FF, FORMAT),
        (0x900, FORMAT), (0x901, FORMAT), (0x980, FLAG),
        (0xA00, FORMAT), (0xB00, FORMAT), (0xC00, OK), (0xC01, OK),
        (0xC02, FORMAT), (0xC7F, FORMAT), (0xC80, FLAG),
        (0xC81, FLAG), (0xD00, FORMAT), (0xE00, FORMAT),
        (0xF00, FORMAT), (0xFFF, FLAG),
        (0xFFFF_F100, OK), (0xFFFF_F201, OK), (0xFFFF_F800, OK), (0xFFFF_FC01, OK),
        (0xFFFF_F600, FORMAT), (0xFFFF_F202, FORMAT), (0xFFFF_F280, FLAG),
    ];
    let h = Harness::new();
    let context = h.context(4, 0);
    for &(format, expected) in cases {
        let r = h.out2.port_create(context, &h.param(format));
        match r {
            Ok(port) => {
                assert_eq!(expected, OK, "format {format:#x} was accepted");
                assert!(port != 0 && port != u64::MAX);
                h.out2.port_destroy(port);
            }
            Err(e) => assert_eq!(e as u32, expected, "format {format:#x}"),
        }
        assert_eq!(validate_main_port_format(format).err().map_or(OK, |e| e as u32), expected, "format {format:#x}");
    }
    for format in [0x100u32, 0x201, 0x800, 0xC01] {
        for bit in 12..32 {
            let port = h.port(context, &h.param(format | 1 << bit));
            h.out2.port_destroy(port);
        }
    }
    h.out2.context_destroy(context);
    h.finish();
}

#[test]
fn data_formats_and_port_types() {
    assert_eq!(data_format_channels(0x0000), 2);
    assert_eq!(data_format_channels(0x0C00), 12);
    assert_eq!(data_format_channels(0x1100), 16);
    assert_eq!(data_format_to_audio_format(0x0880), Format::Float8ChStd);
    assert_eq!(data_format_to_audio_format(0x0800), Format::Float8Ch);
    assert_eq!(data_format_to_audio_format(0x0C80), Format::Unknown);
    assert_eq!(data_format_to_audio_format(0x0101), Format::S16Mono);
    assert_eq!(data_format_to_audio_format(0x0102), Format::Unknown);
    assert_eq!(port_type_to_audio_out_type(3), audio_out::PORT_TYPE_PADSPK);
    assert_eq!(port_type_to_audio_out_type(4), audio_out::PORT_TYPE_PERSONAL);
    assert_eq!(port_type_to_audio_out_type(6), audio_out::PORT_TYPE_VIBRATION);
    assert_eq!(port_type_to_audio_out_type(0x100), audio_out::PORT_TYPE_MAIN);
}

#[test]
fn latency_table_and_mastering_states() {
    let h = Harness::new();
    assert_eq!(h.out2.set_3d_latency(1, 3, 100), AUDIO_OUT2_ERROR_INVALID_PARAM);
    for user in 0..LATENCIES_MAX as u32 {
        assert_eq!(h.out2.set_3d_latency(user, 0, 100), 0);
    }
    assert_eq!(h.out2.set_3d_latency(0, 0, 200), 0, "existing entry was not updated");
    assert_eq!(h.out2.set_3d_latency(99, 0, 100), AUDIO_OUT2_ERROR_PORT_FULL);
    let s = mastering_states(MASTERING_STATES_ID_DEFAULT).unwrap();
    assert_eq!(s.len(), 336);
    assert_eq!(u32_at(&s, 0x10), MASTERING_STRUCT_ID_COMPRESSOR_DEFAULT);
    assert_eq!(u32_at(&s, 0xE0), MASTERING_STRUCT_ID_LIMITER_DEFAULT);
    let s = mastering_states(MASTERING_STATES_ID_V2).unwrap();
    assert_eq!(s.len(), 400);
    assert_eq!(u32_at(&s, 0x120), MASTERING_STRUCT_ID_LIMITER_DEFAULT);
    assert!(mastering_states(3).is_none());
    h.finish();
}

#[test]
fn speaker_arrays_and_coefficients() {
    assert_eq!(speaker_array_memory_size(2, false, false), 0x400 + 2 * 0x40);
    assert_eq!(speaker_array_memory_size(0, true, false), 0x400 + 0x40 + 0x200, "speaker count clamps to 1");
    assert_eq!(speaker_array_memory_size(99, false, true), 0x400 + 32 * 0x100, "speaker count clamps to 32");
    // Fill the table, then recover one slot.
    let mut handles = Vec::new();
    while let Some(h) = speaker_array_create() {
        assert!(h != 0 && !handles.contains(&h));
        handles.push(h);
    }
    assert_eq!(handles.len(), SPEAKER_ARRAYS_MAX);
    let last = handles.pop().unwrap();
    speaker_array_destroy(last);
    assert_eq!(speaker_array_create(), Some(last), "destroyed slot was not reused");
    handles.push(last);
    for h in handles {
        speaker_array_destroy(h);
    }
    let c: Vec<f32> = (0..4).map(|i| speaker_coefficient(None, i)).collect();
    assert_eq!(c, [1.0, 1.0, 0.0, 0.0]);
    for (channel, w) in [(0, 0.707_106_77), (64, 0.707_106_77), (1, 1.0), (65, 1.0)] {
        assert_eq!(speaker_coefficient(Some(channel), 0), w);
        assert_eq!(speaker_coefficient(Some(channel), 1), 0.0);
    }
}

/// Every PCM buffer set between two pushes plays, one per grain and in
/// order; a port whose buffers move on plays silence once they run out, a
/// port always given the same buffer repeats it.
#[test]
fn port_buffers_set_between_pushes_play_in_order() {
    const BYTES: usize = 512 * 2 * 4;
    let h = Harness::new();
    let context = h.context(1, 0);
    let port = h.port(context, &h.param(STEREO_FLOAT));
    let level = |i: u64| 0.01 * i as f32;
    for i in 1..=24u64 {
        h.write(i * 0x1000, f32_bytes(&[level(i); 1024]));
    }
    let played = |h: &Harness| -> Option<f32> {
        let output = h.push_pcm(context, BYTES);
        assert!(output.len() <= 1, "one port queued more than one buffer per grain");
        output.first().map(|pcm| f32::from_le_bytes(pcm[0..4].try_into().unwrap()))
    };

    // Three buffers set before one push play in order.
    for i in 1..=3 {
        h.set_pcm(port, i * 0x1000);
    }
    for i in 1..=3 {
        assert_eq!(played(&h), Some(level(i)), "buffer {i} of a burst was dropped or reordered");
    }
    assert_eq!(played(&h), None, "an advancing port repeated a stale grain");

    // A burst after a push, then a gap: every buffer plays once.
    h.set_pcm(port, 0x4000);
    assert_eq!(played(&h), Some(level(4)));
    h.set_pcm(port, 0x5000);
    h.set_pcm(port, 0x6000);
    assert_eq!(played(&h), Some(level(5)));
    assert_eq!(played(&h), Some(level(6)));
    assert_eq!(played(&h), None, "a gap after a burst repeated its last grain");

    // At most QUEUED_PCM_MAX buffers wait: the oldest go.
    let total = QUEUED_PCM_MAX as u64 + 4;
    let address = |n: u64| (n % 24 + 1) * 0x1000;
    for n in 0..total {
        h.set_pcm(port, address(n));
    }
    for n in 4..total {
        assert_eq!(played(&h), Some(level(address(n) / 0x1000)), "the queue did not keep the newest buffers");
    }
    assert_eq!(played(&h), None);
    h.out2.port_destroy(port);

    // A port always given the same buffer repeats it until a null set.
    let port = h.port(context, &h.param(STEREO_FLOAT));
    h.set_pcm(port, 0x7000);
    assert_eq!(played(&h), Some(level(7)));
    assert_eq!(played(&h), Some(level(7)), "a port reusing one buffer did not repeat it");
    h.set_pcm(port, 0x7000);
    h.set_pcm(port, 0x7000);
    for _ in 0..3 {
        assert_eq!(played(&h), Some(level(7)));
    }
    h.set_pcm(port, 0);
    assert_eq!(played(&h), None, "a null PCM set did not silence the port");
    assert_eq!(played(&h), None);
    h.out2.port_destroy(port);
    h.out2.context_destroy(context);
    h.finish();
}

#[test]
fn chat_enable_and_disable_are_accepted() {
    let vm = crate::sync::test_util::vm_arena(0x5C48_0000_0000);
    let mut ctx = CpuContext::new(vm);
    assert_eq!(cb_enable_chat(&mut ctx), 0);
    assert_eq!(cb_enable_chat(&mut ctx), 0, "enabled twice");
    assert_eq!(cb_disable_chat(&mut ctx), 0);
    assert_eq!(cb_disable_chat(&mut ctx), 0, "disabled twice");
    assert_eq!(cb_enable_chat(&mut ctx), 0);
    assert!(CHAT_ENABLED.load(Ordering::SeqCst));
    assert_eq!(cb_disable_chat(&mut ctx), 0);
}

/// Speaker info as the console reports it; the call is refused before the
/// AudioOut2 initialisation, and the mastering and debug-state setters
/// refuse a null parameter.
#[test]
fn speaker_info_and_null_parameters() {
    let vm = crate::sync::test_util::vm_arena(0x5C49_0000_0000);
    let mut ctx = CpuContext::new(vm);
    let out = 0x5C49_0000_0000u64;
    ctx.write_bytes(out, &[0xEE; SPEAKER_INFO_SIZE]);
    ctx.set(crate::context::CpuRegister::Rdi, out);
    assert_eq!(cb_get_speaker_info(&mut ctx), AUDIO_OUT2_ERROR_NOT_INITIALIZED);
    crate::audio::AUDIO_OUT_INITIALIZED.store(true, std::sync::atomic::Ordering::Release);
    assert_eq!(cb_get_speaker_info(&mut ctx), 0);
    let b = ctx.read_bytes(out, SPEAKER_INFO_SIZE).unwrap();
    assert_eq!(b[0], SPEAKER_TYPE_TV);
    assert_eq!(u32::from_le_bytes(b[4..8].try_into().unwrap()), 3);
    let angle = |i: usize| {
        let a = SPEAKER_INFO_ANGLE_OFFSET + i * 4;
        (i16::from_le_bytes([b[a], b[a + 1]]), i16::from_le_bytes([b[a + 2], b[a + 3]]))
    };
    assert_eq!((angle(0), angle(1)), ((30, 0), (-30, 0)));
    for i in 2..16 {
        assert_eq!(angle(i), (-360, -360));
    }
    assert_eq!(cb_mastering_set_param(&mut ctx), 0);
    assert_eq!(cb_set_system_debug_state(&mut ctx), 0);
    ctx.set(crate::context::CpuRegister::Rdi, 0);
    assert_eq!(cb_mastering_set_param(&mut ctx), AUDIO_OUT2_ERROR_MASTERING_INVALID_API_PARAM);
    assert_eq!(cb_set_system_debug_state(&mut ctx), 0x8026_0502u32 as i32 as i64);
    crate::audio::AUDIO_OUT_INITIALIZED.store(false, std::sync::atomic::Ordering::Release);
}
