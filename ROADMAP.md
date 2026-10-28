# Roadmap

The main directions of RustPS5Emu (RPS5E) for the next releases. Plans change as games show what they need; the [CHANGELOG](CHANGELOG.md) lists what is done.

## 0.1.x: correct pictures in the tested games

- Grand Theft Auto V: no rendering glitches in the city (reflections, depth of field), fewer stalls when the CPU waits for the GPU, a higher frame rate in the city.
- Demon's Souls: textures and foliage that flicker or disappear.
- Marvel's Spider-Man 2: models rendered broken and without textures, the long freeze after the logos.
- System libraries that games call during start-up and loading (PlayGo and others), each checked against the console's behaviour.
- Every fix is checked on all tested games before a release, so nothing that worked breaks.

## 0.2: speed

- Hardware ray tracing: games that use the PS5's ray-tracing instructions run their ray tests on the GPU's ray-tracing hardware (Vulkan ray queries) instead of shader code.
- Shaders that read data the GPU built in the same frame (descriptors, counts) without waiting for the GPU to finish first.
- Draw preparation spread over more CPU threads.
- Faster 16-bit float shader paths on GPUs without native support for the console's exact rounding.
- Shorter shader-compile stutter: more of the work done ahead of time and kept in the cache.

## Later

- Running PS5 fake packages directly, without extracting them first.
- More games tested and listed in the compatibility table.
- Frame pacing: variable refresh rate, HDR output, smoother presentation.
- Wider hardware coverage: AMD and Intel GPUs, Windows on par with Linux.

Contributions to any of these are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md).
