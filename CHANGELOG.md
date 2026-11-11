# Changelog

All notable changes to RustPS5Emu (RPS5E) are listed here. Versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

- The emulator's own small writes into GPU-watched guest memory (memcpy/memset inside one page, packets of the command-buffer builders) mark the page written before the store instead of taking a page fault (`RPS5E_PREMARK_SMALL`). GTA V's city in Performance mode: write faults per frame ~1550 → ~320, in-run A/B +11.6 % frames (about 38 → 41 fps).
- Less overhead per draw in heavy scenes: the recording thread wakes the command interpreter once per half queue instead of once per command (`RPS5E_QUEUE_WAKE_BATCH`), and streams of page-sized buffer blocks share one buffer mirror instead of one per block (`RPS5E_STREAM_ADJACENT`). GTA V gains a few percent.
- Shader programs are now decoded and planned on a worker thread as soon as the game creates the shader object (`sceAgcCreateShader`), not inside the first draw that uses them; the draw takes the finished program (`RPS5E_CREATE_TIME_SHADERS=0` turns it off).
- Fixed shader planning that could take seconds (or never finish) on programs whose buffer descriptors are computed through long chains of shared values.
- Shader objects the game creates (`sceAgcCreateShader`) now move the prewarm work of their programs to the front of the shader prewarm list while it builds in the background, so those modules and pipelines are ready before the draws that use them (`RPS5E_CREATE_TIME_SHADERS=0` turns it off). Modules are still made per draw permutation: what a module depends on (resources, render targets, vertex layout) is only known at the draw.
- GRAND THEFT AUTO V's city: the GPU recording thread polls the GPU timeline briefly before blocking in the driver (`RPS5E_GPU_WAIT_SPIN`, in-run A/B +5.2 % flips), lookups of write-back spans skip long spans far from the range (`RPS5E_THROUGH_LONG_BOUND`, +3.5 % flips, recorder CPU per frame -5 %), and image eviction and repeated GPU-write records do less work. Frames of the captured city frames are byte-identical.
- `RPS5E_HW_RT=1` (off by default, experimental): hardware ray tracing for guest BVH traversals on devices with VK_KHR_ray_query. The host mirrors the BVHs the game's ray-tracing shaders traverse as Vulkan acceleration structures (triangles from the guest's triangle nodes, opacity and instance flags carried over), updated once per frame and refitting only the BLASes whose triangles changed (~1 ms of GPU a frame in GTA V's city, was ~4). The shader translator recognises the game's traversal loops (a node stack kept in VGPR lanes over `image_bvh_intersect_ray` queries) and gives them a copy whose nodes come from host ray queries: the RT cores find the candidates, the game's own instance, alpha-test and hit code runs on them unchanged, and instances the host has no mirror for are walked by the game's own loop. In GTA V's city the shadow traversal takes less GPU time in streets and alleys (~5 vs 7-10 ms a frame) but more in foliage-heavy views, so it stays off. The host queries cull the same triangle faces as the game's facing test (they culled the opposite ones, so non-opaque single-sided geometry such as Franklin cast no shadow with it). `RPS5E_HW_RT_PRUNE=1` keeps the earlier box pruning for traversals without a copy; `RPS5E_HW_RT=2` adds counters (diagnostics); `RPS5E_HW_RT_DUMP=<dir>` writes the guest BVHs it sees.
- GRAND THEFT AUTO V's city runs about 35 % faster with all the changes below (user's city recording, frames after the 3rd Triangle, quiet machine: 0.1.4-alpha 8.0-8.3 fps → 11.0). Compute pipelines whose shader the GPU driver compiles with register spills are built again in the background from the same program translated with the emulated f32 denormal flush kept, and that build replaces the first one when it spills less (`RPS5E_SPILL_VARIANTS`, needs VK_KHR_pipeline_executable_properties). Both compute the same results; on an RTX 5090 GTA V's ray-tracing traversal no longer spills (168 → 223 registers, 160 bytes of spills → none) and takes 40-65 % less GPU time (city 37 → 20 ms per frame, open road 121 → 44 ms).
- End-of-pipe labels now reach guest memory once the GPU has finished the work before them, as on the console, while the GPU command processor's own waits see them as soon as that work is recorded (the default; `RPS5E_LABELS_AT_EOP=0` writes them as soon as the work is recorded, as before). A game thread that reads GPU results after such a label then finds them in memory instead of waiting for all queued GPU work while holding the GPU back-end (GTA V's main thread does that every frame: its city runs ~15 % faster, 10.7 → 12.6 fps in two alternating runs each). Marvel's Spider-Man 2 plays its intro videos ~25 % slower (~51 → ~38 fps): its command processor waits there for labels its own command buffers release, which now means waiting for the GPU, as on the console; its menus run as before. A GPU queue that waited for such a label before it landed makes its next memory write only once the label is in memory, as on the console (`RPS5E_LATE_WRITE_ORDER`): before, Demon's Souls and Minecraft, which keep labels inside their command buffers, could see the later write first, reuse the command buffer, and have the label land over their new commands (`bad packet header 0x00000001`, then a freeze).
- GRAND THEFT AUTO V's city: another ~4-5 % (two in-run A/Bs: +5.5 % and +3.7 % flips, ahead in 10 of 14 10-second windows with 3 ties; the GPU's time per frame down ~5 %). In a run of back-to-back indirect dispatches with thread counts where every dispatch is followed by a barrier (GTA V's ray-tracing structure updates, ~300 a frame, each written back to guest memory), the thread-count kernels started a new launch batch at every other dispatch; each batch split the command buffer twice and added two full barriers. Such a run now records its kernels next to the dispatches once a batch held a single kernel (`RPS5E_LAUNCH_BATCH_SINGLES`, default on).
- GRAND THEFT AUTO V's city: another ~10-20 % (three in-run A/Bs: +21 %, +23 % and +12 % flips, ahead in 21 of 24 10-second windows). The copies that bring compute results back to guest memory now follow the whole run of dispatches that wrote them, behind one barrier, instead of each dispatch (`RPS5E_THROUGH_RUNS`, default on); a copy needs the compute work before it finished, so a copy after every dispatch made the GPU finish each dispatch before the next could start. GTA V's ray-tracing structure updates (~2000 small dispatches a frame) were drained that way; the GPU now has ~95 % fewer such barriers and ~28 % fewer copies (a range an earlier copy of the same batch covered gets none: `RPS5E_THROUGH_DEDUPE`), its time per frame is ~10 % lower, and the GPU command recorder's waits for that work (~15 ms a frame) mostly went. Same pictures (capture replay frames byte-identical).
- A new GPU buffer whose pages go into two device-address page tables gets one full barrier after both instead of one after each (`RPS5E_TABLE_BARRIER_MERGE`).
- The GPU profile (`RPS5E_GPU_PROFILE=1`, used by input-test replays) costs the GPU command recorder about half as much: its spans are timed with the CPU's time stamp counter instead of the system clock (`RPS5E_PROFILE_TSC`), and the thread CPU time of draws (a system call) is read on one draw in eight (`RPS5E_PROFILE_CPU_EVERY`). GTA V's capture: recorder time with the profile 42 → 37 ms per frame (without it 34 → 33).
- Shader LDS float minimum and maximum atomics (DS_MIN_F32 / DS_MAX_F32) are one integer atomic when the source is +0, an infinity or a normal number, instead of a compare-exchange loop (`RPS5E_DS_INT_MIN_MAX`, default on; the same words bit for bit). GTA V's per-object ray-tracing structure builds use them to merge child bounds: their GPU time drops ~14 % (capture replay, same frames). Games translate their shaders once more on the first start.
- The GPU command recorder does less bookkeeping per draw and dispatch (GTA V's city: its CPU time per draw ~5 % lower in an in-run A/B, per item ~1-3 % overall; the frame rate moved by less than the run-to-run spread, ~12.4 → ~12.6 fps): before work that reads memory through unbounded device addresses only the buffers the CPU wrote since the previous time are looked at, from the tracker's log of CPU writes, instead of every buffer (`RPS5E_UNRANGED_LOG`); a range is checked for watchability against the page registry before the memory-layout memo (`RPS5E_REGISTERED_FIRST`); the buffer holding a range is found from the per-page index alone (`RPS5E_INDEX_EXTENTS`); long write-back spans no longer widen every span lookup (`RPS5E_THROUGH_LONG`); the recorder's own timing reads the time stamp counter instead of the system clock. All on by default; the same pictures (capture replay frames byte-identical).
- With `RPS5E_AB_PERIOD`, the `[gpu]` line also counts dispatches and their recording CPU time per A/B mode (`ab_dispatches`, `ab_dispatch_cpu_ms`).
- Diagnostics for the GPU command recorder's waits: `RPS5E_WORD_SPECULATION=check` compares every GPU-written word a shader's preparation waits for with the value guest memory held before, times the wait by whether all GPU work writing the word had already finished, and names the draw or dispatch that wrote it; `RPS5E_PM4_BLOCKED_WHY=1` sums the time all GPU command queues were blocked by what each queue waited for; `RPS5E_WRITE_RANGE_CHECKS=1` makes device-address stores count the ones outside the range the host planned for them. All off. Games translate their shaders once more on the first start.
- `RPS5E_WAIT_TRACE_TICKS=1` (with `RPS5E_WAIT_TRACE` and the GPU profile) logs every command buffer's GPU time and the GPU's idle time before it; frame-trace lines (`RPS5E_FRAME_TRACE`) show the guest queue of every draw and dispatch.
- `RPS5E_ITEM_CPU=1` also lists, per program, the barriers, submits and write-through copies each draw or dispatch recorded; `RPS5E_BARRIER_SITES=1` counts full barriers per call site and program; the `[gpu]` line counts compute and transfer-read barriers and write-through copies.
- `RPS5E_PIPELINE_STATS=1` logs the guest program address of each compute pipeline.
- `RPS5E_WAIT_TRACE=<ms>` (diagnostics) logs the GPU command recorder's long waits for the GPU with the submissions they waited for and, with the GPU profile, the GPU time and idle time of each.

### Fixed

- Ray-tracing shaders now intersect BVH triangle nodes of types 2 and 3 (the node's triangles v2 v3 v4 and v2 v4 v0, as the guest's BVH builders lay them out); only types 0 and 1 were tested before, so ~9 % of GTA V's shadow-casting triangles never cast a shadow.
- The GPU profile (`RPS5E_GPU_PROFILE=1`) lost the GPU time of a submission's earlier command buffers and of its launch batches whenever a launch batch split the submission: the time is now counted, and the batches show as `chained`. GTA V's city was under-reported by ~25-30 ms per frame (real GPU time ~80-85 ms per frame, not ~51).
- `RPS5E_FRAME_TIMES` files are appended every second instead of being written at shutdown (and every 1024 frames): a run stopped by a timeout no longer leaves a file cut at 2048 frames.
- `sceKernelMprotect` / `mprotect` over a range that starts in unmapped memory, or that has nothing mapped at all, now changes the mapped pages and returns 0 as the console does (was EINVAL). `sceKernelMtypeprotect` over a range holding flexible memory returns EOPNOTSUPP (0x8002002D) and changes nothing, as on the console.
- `sceKernelSignalSema` with a count of 0 returns EINVAL as on the console (was 0).
- `sceKernelWaitEqueue` checks the event count before the queue and stores 0 events on every failure.
- `sceVideoOutRegisterBuffers` / `sceVideoOutRegisterBuffers2` over buffer slots that are still registered return SLOT_OCCUPIED (0x80290010) instead of RESOURCE_BUSY.
- `sceNetResolverStartNtoa`: a timeout of 0 stands for one second and a timeout under one second is EINVAL; an unknown resolver id is reported before a missing host name or address, as the system library does.
- Shaders: `v_mad_u32_u16` / `v_mad_i32_i16` honour op_sel bits 2 and 3 (high half of the 32-bit addend; result into the high half), checked against RDNA2 hardware results; an f64 result clamp keeps a NaN when DX10_CLAMP is off, like f32 and f16; `v_div_fmas_f32` flushes a denormal result in the f32 modes that flush results (its inputs stay unflushed). Games translate their shaders once more on the first start.
- After a context clear the four clip rectangles are (0, 0)-(8192, 8192), as in the system's register defaults, instead of empty: a clip-rectangle rule set without writing the rectangles no longer hides every draw.
- Windows: guest threads start with the console's x87 control word (64-bit precision) instead of Windows' 53-bit one, so the games' `long double` arithmetic is no longer rounded short.

## [0.1.4-alpha] - 2026-11-01

### Added

- On-screen keyboard for games' text input dialogs (sceImeDialog), drawn over the game: type on the keyboard (Enter accepts, Esc cancels) or use the controller (D-pad and Cross on the keys, Circle cancels, Triangle space, Square delete, L1 / R1 move the caret, L2 shift, R2 or Options accept), or click the keys. The game's pads read released while it is shown, and buttons still held when it (or the in-game menu) closes reach the game only after they are released. Before, the dialog was accepted at once with a default text (Demon's Souls' character name). Runs without a window, scripted runs and input recordings and replays still finish the dialog by themselves; `RPS5E_IME_HEADLESS=1` does so with a window too.

### Changed

- Play mode no longer waits for shaders at all: a draw whose pipeline is still compiling is skipped from the first frame and drawn from the frame its pipeline is ready (the per-frame wait of 3 ms is now 0, `--compile-wait`); a pipeline the pipeline cache already holds is made on the spot instead of being skipped (`RPS5E_INLINE_PROBE`). Draws that write memory, draws without render targets and compute dispatches still wait, since the game and later GPU work read what they write.
- The saved shader list of a game is built in the background while the game runs (on all compile workers, with "Building the shader cache: done / total" in the corner) instead of before the game starts; `--prewarm-wait` (`RPS5E_PREWARM_WAIT=1`) keeps the old behaviour. Tests and input replays still build it first.
- Compiles are ordered by need: the shaders a draw or dispatch needs now, then the saved list, then the compile lookahead's guesses; an eighth of the compile workers is kept free for the shaders needed now.
- The compile lookahead also reads the command buffers already queued after the current one, on every GPU queue, and reads past flips and conditional commands (`RPS5E_LOOKAHEAD_QUEUED`, `RPS5E_LOOKAHEAD_GUESS`); the GPU command thread starts it whenever it meets a new compute program (`RPS5E_LOOKAHEAD_EAGER`), so it also works while draws are recorded on their own thread, where it found nothing before.
- Demon's Souls gameplay runs about 1.4-2x faster (user's play recording, frames after the 3rd Triangle: 4.2 → 6.2-8.8 fps; 3 of 3 A/B pairs faster in one run, 5.97 → 8.50 fps). Shader resources read from a guest memory page that also holds GPU results no longer count as GPU-written when the GPU never wrote those bytes, or when its copy back to guest memory already finished: such reads waited 20-40 ms for all queued GPU work, and descriptor-table words were copied on the GPU, which split almost every draw into its own render pass and tripled the GPU time in runs where the game's tables shared a page with GPU output (`RPS5E_UNWRITTEN_WORDS`, `RPS5E_LANDED_WORDS`). The GPU command thread also prepares the shaders of GPU-indirect draws the way the recording thread records them, so they are no longer prepared twice (`RPS5E_PREFETCH_INDIRECT`), and remembers per page whether it holds GPU results while it reads a shader's resources (`RPS5E_DIRTY_PAGE_MEMO`).
- The recording thread (`RPS5E_GPU_RECORD_THREAD=auto`) no longer switches itself off during loading pauses: a one-second window that recorded almost nothing no longer counts as the recorder waiting (`RPS5E_POLICY_QUIET`). In GTA V's city a probe that fell into such a pause kept the thread off for one to two minutes, and the city then ran at ~6.2 fps instead of ~7.8 (2 of 9 benchmark runs).
- Bindless texture tables cost less to keep current: elements that can never have a view (null, another dimension, a depth compare on a colour format, ...) are no longer looked up again at every guest mapping change (`RPS5E_BINDLESS_WORD_FAILURES`), a changed table is read again only on its changed pages (`RPS5E_BINDLESS_PAGE_READS`), and pinned images are logged once between two reads of the log (`RPS5E_PINNED_DEDUP`). GTA V's two ray-tracing dispatches per frame took 2.2-4 ms of recording each, now 1-1.9 ms (the recording thread's CPU per city frame 79 → 75 ms).
- Shaders are much smaller and build up to twice as fast where the GPU flushes f32 denormals itself: the console's flush of tiny float values (FLOAT_MODE) was emulated with about six extra instructions per float operation; the emulator now measures at start what the GPU's own float operations do with denormals (each shader stage) and leaves the emulated flush out wherever the GPU already gives the same bits, and compares values that cannot be denormal with the GPU's own compare (`RPS5E_HOST_FTZ`, `RPS5E_NATIVE_F32_COMPARE`). On an RTX 5090: building the saved shader lists from scratch takes GRAND THEFT AUTO V 23 s instead of 37 s, ASTRO BOT 18 s instead of 40 s and SILENT HILL: The Short Message 11 s instead of 25 s, and the largest ASTRO BOT shader shrinks from 1.1 M to 0.56 M words; pictures are unchanged. Games translate their shaders once more on the first start. `RPS5E_NO_F32_FLUSH=1` drops the flush emulation entirely (not exact, for measurements).

### Fixed

- DOOM: The Dark Ages gets past "PRESS (X) TO START": `sceNpGetAccountAge` now completes its request as signed out, so the game's `sceNpPollAsync` reports the result (it answered "invalid request" forever and the title screen waited). The game goes on to the legal screen, saves its profile and opens the campaign menu. `RPS5E_NP_REQUEST_COMPLETION=0` restores the old behaviour.
- `cpuset_getaffinity`, `cpuset_setaffinity`, `pthread_getaffinity_np` and `pthread_setaffinity_np` accept sets of 8 to 16 bytes and return ERANGE for other sizes, as measured on the console; `pthread_getaffinity_np` accepted only 32 bytes and the others up to 32.
- Block-compressed 3D textures in tiled thick layouts (4x4 blocks stacked in depth) are read instead of being bound as empty textures ("image layout (swizzle mode 5, D3, 16 bytes per element)" in DOOM: The Dark Ages and Marvel's Spider-Man 2). `RPS5E_BC_VOLUMES=0` restores the old behaviour.
- PlayGo reads the chunk count from `playgo-chunks.xml` in the game folder, so games whose dump has no other PlayGo data no longer get "PlayGo not supported" from `scePlayGoOpen` (DOOM: The Dark Ages crashed right after start).
- DOOM: The Dark Ages no longer loses the GPU ("Vulkan device lost", exit code 3) on the title screen. A slot of the bindless texture set kept its descriptor when a texture's new view got the same handle as its old one, but the driver hands out the handle of a destroyed view again, so the slot still described the destroyed view and sampling it hung the GPU. A slot now belongs to one view of one image.
- Textures selected through large bindless tables share one slot per view between the tables that read them. DOOM: The Dark Ages reads one texture heap through six tables and filled the 131 072-slot set ("bindless set full"); its draws then failed and the title screen stayed black.
- Pipelines of the saved shader list that use the bindless texture set are built at the start too; they failed with no pipeline layout and were built during play in every run (GTA V: 39 of 2302 pipelines, the ray-tracing ones, 2-4 s each) (`RPS5E_PREWARM_BINDLESS`).
- Shaders and pipelines added to the saved shader list are kept within a second (a journal next to the list, `RPS5E_PREWARM_JOURNAL`), so a run that crashes or is killed no longer loses the last 30 s of them.
- Pipelines of shaders that read their own code address and turned out not relocatable are saved under the shader's own key, so the next run finds them in the list.
- `sceKernelCreateSema` accepts the attribute bit 0x100 (values 0x100-0x102), as the console does; it returned EINVAL.
- `sceKernelGetSystemSwVersion` and `sceKernelGetProsperoSystemSwVersion` accept any value in the block's size field and leave it as it is, and return 0 for a null pointer, as the console does; they returned EINVAL and EFAULT.
- `scePthreadCondSignalto` returns ESRCH for a handle that names no thread and EPERM for a thread that does not wait on the condition variable, as the console does; it returned success.
- `sceKernelHasTrinityMode` is implemented and reports a base PS5 (0); it was missing, and its unimplemented-call error read as "PS5 Pro mode available".
- Block-compressed textures whose descriptor has the compression bit set are no longer read through colour-compression metadata at the descriptor's (meaningless) metadata address.

## [0.1.3-alpha] - 2026-10-31

### Fixed

- Frame captures (`RPS5E_CAPTURE_FRAMES`) of GTA V replay again instead of hanging the GPU. While a capture recorded, GPU results copied straight into guest memory were later taken for CPU writes and recorded; the replay wrote them over its own GPU results, which differ where GPU work orders its atomics differently, and the mixed ray-tracing tree sent a shader into an endless loop. GPU results now reach guest memory through copies the capture sees while it records.
- Silent Hill 2 (Remake) no longer hangs the GPU after START GAME. A small compute write (a 4-byte counter in reused transient memory) at an address where a render target once kept its fast-clear metadata was taken for a metadata clear, which overwrote the counter and left a GPU loop running for billions of iterations; only writes that cover a whole metadata slice count as clears now (`RPS5E_WHOLE_METADATA_WRITES`).
- Silent Hill 2 (Remake): the ray-tracing acceleration-structure build after START GAME no longer loops forever on the GPU. Compute shaders whose buffer record count comes from a value computed at run time (here an atomic counter read back with `v_readfirstlane`) were skipped, so the sort before the tree build never ran; they are translated now (`RPS5E_RUNTIME_BUFFER_RECORDS`). The game loads and reaches "PRESS ANY BUTTON TO PLAY".
- Silent Hill 2 (Remake) no longer freezes on the loading screen after START GAME (about one run in six). `sceAudioOut2ContextPush` in blocking mode returned with the context queue still full, so `sceAudioOut2ContextGetQueueLevel` right after it reported no free grain; Wwise renders only the free grains and pushes only rendered ones, so its audio thread stopped, and the garbage collector waited forever for an audio bank to unload while holding the loading screen's lock. A blocking push now returns once the queue has room for the next grain.
- Silent Hill 2 (Remake): the picture after START GAME is no longer black; the opening cutscene and the lookout are drawn. Unreal Engine 5 clears the compression keys of its render targets with a small compute kernel that loads the clear code from a constant buffer; the emulator runs that kernel as a plain memory fill and read the code as out of range (it counted the buffer's records as bytes instead of 16-byte units), so the separate-translucency target was cleared to transparent black instead of (0, 0, 0, 1) and the anti-aliasing pass multiplied the whole scene by 0.
- Silent Hill 2 (Remake) no longer crashes in the audio heap a few minutes into play. The `sceAcm` convolution-reverb shared-input job read its output array with a guessed layout and cleared the memory those words pointed at, which corrupted the Wwise heap; the job now only advances its batch cursor, like the other ACM job builders.

- Shader subroutine calls through a table of code addresses are found when the shader keeps the table's descriptor in vector register lanes between loading it and the call (Silent Hill 2 (Remake)), and calls through user data when the code address was copied before a branch. A callee that returns with `s_swappc_b64` into the null register is linked too, and every table call's dispatch code comes before all callee code. The Silent Hill 2 shader that needed this still does not compile: its tables link more callee code (about 30 000 words per table) than a 16-bit call reaches.
- `sceVideoOutIsOutputSupported` answers 0 for the 119.88 Hz output mode when the output has no 120 Hz, as the console does, instead of an error.

- Flips that a game reserves with `sceVideoOutSubmitEopFlip` and releases with `sceAgcCbReleaseMem` are performed; the release packet our `sceAgcCbReleaseMem` writes was not recognised, so the flip never happened.
- `COPY_DATA` packets that read memory on the prefetch engine read memory instead of GDS, and can load registers; the prefetch engine's atomic result and the clocks can be copied too. A `RELEASE_MEM` to an unsupported destination still raises its interrupt.
- GPU results are no longer lost when a game unmaps part of the memory a GPU buffer covers (`RPS5E_UNMAP_WHOLE_BUFFERS`), or maps the same memory again at the same address (`RPS5E_SAME_BACKING_REMAP`).
- Stencil add and subtract operations use the operation value from `DB_STENCILREFMASK` (a value of 0 leaves the stencil unchanged) (`RPS5E_STENCIL_ADD_OPVAL`).
- Primitive restart honours `MATCH_ALL_BITS`: a restart index wider than the index type never restarts.
- Fast clears of 5_6_5, 1_5_5_5, 5_5_5_1 and 4_4_4_4 render targets are decoded.
- Multisampled render targets and depth buffers take their real size in memory (`RPS5E_MSAA_BLOCKS`); they were taken as larger than they are and seen to overlap the memory after them.

- `sceKernelMunmap` rounds the range to whole pages like `munmap`; an unaligned length left part of a page mapped and charged to the flexible memory budget.
- A texture with a shorter mip chain than a cached image at the same address no longer samples that image's levels, which lie at other bytes (`RPS5E_ALIAS_SAME_MIPS`).
- Indirect indexed draws stop at the end of the index buffer (`INDEX_BUFFER_SIZE` counted from the index base); records with a first index read indices past it.
- Shader loops that wait for EXEC to empty end when the shader enabled lanes the host does not run (partial waves, wave64 programs on 32-lane GPUs) (`RPS5E_PRESENT_EXEC`).
- XOR wave reductions in graphics shaders are computed over the lanes the host subgroup has.
- Small render targets that the CPU reads back (a linear target, or a tiled one of at most 64x64 texels) are copied to guest memory after draws and clears, not only after shader stores (`RPS5E_READBACK_TARGETS`).
- Depth-compare samples of colour array textures that the shader compares itself (formats without a depth view) take the layer rounded half to even, as the GPU sampler does (a slice of 0.49999997 read layer 1); a not-equal compare passes against NaN.
- `sceAcmBatchJobPriority` is implemented and `sceAcmBatchJobNotification` adds its 16-byte job, clears the notification block and checks its arguments as the console's library does (Silent Hill 2 (Remake) adds priority jobs to its reverb batches).
- `sceNpTrophy2GetRewardIcon` reports that the icon does not exist, like the other trophy icon calls (Terminator 2D: NO FATE).

### Changed

- Less CPU time on the GPU command thread (GTA V's city frame is that thread's work): on Linux, the pages the GPU caches watch for game writes are write-protected with userfaultfd instead of memory protection changes (no split mappings, cheaper faults and protection changes; falls back to memory protection where the kernel lacks it; `RPS5E_UFFD_WATCH`); GPU objects are destroyed on a helper thread (`RPS5E_ASYNC_RETIRE`); GPU buffers copy their results into guest-memory imports that already exist instead of importing their own range each (`RPS5E_CHUNK_HOST_VIEWS`); the image cache no longer walks all images to sum their sizes every 256 draws, and its memory-pressure collection no longer sorts every idle image at each completed GPU submission. GTA V in the city: the command thread's kernel time per frame about 40 % lower; frame rate +5-6 % per change in in-run A/B measurements.
- Less GPU time and fewer GPU waits in GTA V's city: compute dispatches the game issues back to back with nothing between them are no longer forced to wait for each other (the console runs them at the same time; `RPS5E_GUEST_COMPUTE_ORDER`); the small kernels that turn the thread counts of the ray-tracing refit's indirect dispatches into group counts run in one batch ahead of the dispatches (`RPS5E_LAUNCH_BATCH`) or before the dispatch's own barrier (`RPS5E_EARLY_LAUNCH`), and GPU-written values copied into shader resource tables use narrow barriers (`RPS5E_NARROW_TABLE_COPY`). GPU busy time per frame in the alley scene dropped by about a third (about 90 ms to about 60 ms), the frame rate by 5-15 % in the same scene. Optional (`RPS5E_READ_THROUGHS=1`): GPU results larger than 16 KiB are copied back to guest memory only where the CPU reads them (another ~10 % less GPU time in GTA V; off by default because ASTRO BOT then stalls for seconds).
- On processors with a hybrid or asymmetric-cache layout (Ryzen X3D, Intel P/E cores) the GPU command thread gets one physical core of the fast ones to itself: it runs on one CPU there, and no other emulator thread uses that core (`RPS5E_RECORDER_CORE`). GTA V in the city: about 7 % more frames per second, measured A/B in one run.
- Small tiled render targets the game reads back on the CPU are copied to guest memory at the game's next submit after they are drawn, so the CPU read waits for that copy only, not for all GPU work queued by then (`RPS5E_EAGER_TARGET_READBACK`). GTA V reads four 1x1 targets every frame.
- Shorter stalls the first time a game meets new compute shaders: the compile lookahead now also reads past wait packets in the command stream, so more compute shaders start compiling before a dispatch needs them (`RPS5E_LOOKAHEAD_WAITS`). GTA V with empty shader caches: the total time the GPU thread waited for compiles went from 25 s to 15 s and the longest freeze from 18 s to 13 s.
- Graphics pipelines can be linked from separately compiled vertex-side and pixel-side parts where the GPU driver supports it (`VK_EXT_graphics_pipeline_library`, opt-in: `RPS5E_GPL=1`): the two parts compile in parallel and pipelines that differ only in blending, target formats, vertex layout or rasterizer state reuse them.
- The emulator's memcpy/memset into guest memory that the GPU caches watch marks the pages written in one step instead of one page fault per page (about 5-8 % fewer write faults in Silent Hill 2 and GTA V) (`RPS5E_PREMARK_WRITES`).
- Shader `max(0, x)` (a float maximum with a constant +0) is two instructions instead of the general NaN and signed-zero handling; results are the same bit for bit (`RPS5E_ZERO_MAX`).
- Shader f32 add, subtract, multiply and fused multiply-add test their result for NaN once instead of once per input, with the same results: 2-4 % less GPU time in f32-heavy compute shaders (Marvel's Spider-Man 2 capture) (`RPS5E_NAN_ONCE`).
- Less CPU work on the GPU command thread, the thread GTA V's city frame rate waits for: on Linux the emulator's own allocations can use mimalloc (`RPS5E_MIMALLOC=1`, opt-in: the allocator's share of the thread goes from about 10 % to 2 %, but Silent Hill 2 (Remake) stalls after START GAME with it); copies of game-written memory into the GPU upload buffer run on helper threads (`RPS5E_UPLOAD_THREADS`); the fill-kernel check of compute dispatches remembers its answer per shader (`RPS5E_FILL_DETECT_MEMO`); GPU-written ranges are marked as copied back in place (`RPS5E_WRITTEN_WORDS_IN_PLACE`). GTA V city, measured A/B in one run: 2-5 % less command-thread CPU per draw, within the scene-to-scene spread.
- GPU commands are recorded by two threads when that pays: the command interpreter decodes the game's command buffers and prepares the shader stages of the next draws while a recording thread (on the CPU kept for the recorder) records them (`RPS5E_GPU_RECORD_THREAD`, now `auto` by default: on while the recorder is the busy end, off where one thread is faster). Silent Hill's heavy room +10-20 % fps; light scenes stay at their one-thread speed. Fixes that made the two-thread mode pay: every vertex stage prepared ahead was thrown away (it was prepared with another wave size than the recorder uses); the interpreter's prepared stages are remembered like the recorder's (`RPS5E_PREFETCH_MEMO`); a remembered stage whose pages changed is reused when the bytes it was built from are unchanged (`RPS5E_CONTENT_CHECKS`, also for the one-thread memo); occlusion query dumps no longer make the interpreter wait for the recorder (`RPS5E_ASYNC_OCCLUSION`, 5 000 waits a second in Silent Hill); the recorder wakes the interpreter only while it waits for room.
- Fewer memory protection changes on the GPU command thread: re-watching scattered pages of one 4 MiB block takes one `mprotect` over them and the pages between that already have that protection (`RPS5E_PROTECT_BRIDGE`, Linux; GTA V city: 0.4-3 % less CPU per frame). Re-watching pages the game rewrote in a large buffer window was ~75 % of the time of those uploads.
- Fewer GPU waits on the GPU recording thread: a descriptor word whose last GPU write was already copied back to guest memory is read there, instead of copying its page out again and waiting for the GPU, also when the record of that page's copy-backs was dropped (`RPS5E_WORD_LANDED_ALONE`; GTA V's ray-tracing refit waited ~80 times per frame for such words). GTA V city, A/B in one run: slightly faster on average (within the scene-to-scene spread; the frame is bound by the recording thread's CPU time).
- Faster image refreshes between views of one array surface that start at different layers (a texture or render target addressed at a layer of an array, as Unreal Engine 5 does with its 4-layer 8K atlases): the shared layers are copied image to image instead of going through the guest memory layout (`RPS5E_LAYER_COPY`): about a quarter more frames per second in Silent Hill 2's gameplay, measured A/B in one run.

## [0.1.2-alpha] - 2026-10-28

### Added

- Games run straight from disk images of game dumps, without extracting them: exFAT images (`.exfat`, also inside an MBR or GPT partition), `.ffpfsc` containers (a PS5 PFS image holding a compressed exFAT image) and UFS2 images (`.ffpkg` dumps made on the console). The game folder is found anywhere inside the image.
- The launcher lists `.exfat` and `.ffpfsc` files with their title, version and icon, and its file pickers accept them; `info-pkg` and `install-pkg` read them too.
- PS5 fake packages ("\x7FFIH" packages made in plain-text mode) run straight from the `.pkg`, without extracting them: the Kraken-compressed game image is decoded on demand, and the package's metadata (title, version, icon, `sce_sys/param.json`) is read from it. `info-pkg` and the launcher show them; PS5 retail packages are still reported as encrypted. Silent Hill 2 (Remake) starts from its 48 GB package and reaches the main menu and the New Game settings at about 1 fps; it can still crash right after the start screen (with `RPS5E_LABELS_AT_EOP=through` it does not), and loading a new game stalls.
- Kraken-compressed PFSC files (the PS5 per-file compression) are read; other PS5 compression formats are reported as unsupported instead of failing to read.
- `imports`, `info` and `load` on a package read its `eboot.bin` instead of loading the whole package file.
- libSceNpAuthAuthorizedAppDialog (open, status, result, close), libSceHttp2 WebSockets (offline: requests fail as without a network), `sceSaveDataPrepareForTransferring`, libSceTextToSpeech2 `Speak`/`Close`/`Terminate`, and the libc thread-destructor calls `_sceLibcInternalThreadDtors` and `_sceLibcInternalForceTlsDestructor`.

### Fixed

- Minecraft no longer crashes right at start: system libraries now provide their data objects (the IPv6 addresses `in6addr_any`, `in6addr_loopback` and the others of libSceNet) with real memory. Minecraft reaches the menus and gameplay.
- libSceNpAuth checks its arguments in the console's order and returns its error codes (invalid argument, invalid size, signed out) without writing the outputs on failure.
- Thread-local destructors run the way the console runs them: newest first, in up to four passes, and only while their module is still loaded.
- Minecraft: the sky and distant terrain are no longer pink or magenta and the screen is no longer split in two shades. A 16-bit colour export into an integer render target now keeps its bits, as the console's colour block stores them.
- Minecraft: the daytime sky is blue instead of black, and the large red and white glow bands are gone. Fragment depth is clamped to the viewport's depth range, as on the console, so a depth clear drawn with a polygon offset stores the far plane exactly.
- Unreal Engine 5 games (Silent Hill 2) no longer stop at start with "Command buffer overflow": a command buffer whose end cursor is 0 has room for any packet, as the console's library computes it.
- With `RPS5E_LABELS_AT_EOP=through`, end-of-pipe labels land in the order the queues wrote them; before, an older value could overwrite a newer one and leave both GPU queues waiting forever.
- Demon's Souls: foliage no longer vanishes for single frames, stray stretched triangles no longer flash across the screen and the tunnel exit no longer blows out. A CPU read of GPU results that the game makes after a command-buffer label now waits for those results to reach memory, as on the console; before, it could see an earlier frame's results.

## [0.1.1-alpha] - 2026-10-28

### Added

- Display resolution setting (720p, 1080p or 2160p) in the launcher, the "Run with..." dialog, `--display-resolution` and `RPS5E_DISPLAY_RESOLUTION`: games that ask the console for the TV's resolution are told the chosen one, as a PS5 on that TV would answer; the default stays 2160p.
- PlayGo: `scePlayGoRequestOptionalChunk`, `scePlayGoGetRequiredDisc`, `scePlayGoRequestNextChunk`, `scePlayGoSnapshot` and `scePlayGoSetInstallSpeed2` are implemented. The library now tracks its initialized and open-handle state (`NOT_INITIALIZED`, `ALREADY_INITIALIZED`, `BAD_HANDLE` after close), remembers the install speed that was set, and answers a chunk-id count query made without a list buffer.

### Performance

- GTA V in the city: the longest stalls are much shorter (about 2.5 s instead of 10 s) and driving stutters far less. The frame rate in the city is still low (about 7 fps now that the ray-traced reflections are drawn) and far from playable.
- Wave64 compute shaders on 32-wide GPUs (NVIDIA): lane stacks and register spills stay in registers, ballots share one exchange, and programs from 16 VGPRs run split; the ray traversal of GTA V takes about a third of its former GPU time.
- A CPU read of a descriptor the GPU wrote waits only for the copy that brought that word to memory, not for all queued GPU work.
- Texture uploads that do not fit the full staging ring read guest memory directly (the stall when GTA V enters the city: about 4 s → 1.5 s).
- Ray-tracing shaders read BVH nodes with vector loads: the ray traversal of GTA V is about 8 % faster.

### Fixed

- GTA V: car paint, mirrors and wheel rims no longer look pink or magenta, and the square patches in car reflections are gone. The pass that shades ray-traced reflections never ran, so its magenta placeholders stayed in the picture.
- GTA V: no more grey dots, dotted trails and smears on signs, cars and characters. A render target the game clears with a "single colour" DCC clear was never cleared, so old depth values piled up and broke depth of field.
- Windows: shader and pipeline caches are kept in `%LOCALAPPDATA%\rps5e`; before, they needed `HOME` and were missing without it.
- `PS5_HLE_TRACE` with a relative path writes to that path instead of a fixed file in `/tmp`.
- Default keyboard layout: the left and right halves of the touch pad are on Z and X (they had no key).
- `--refresh-rate` takes the same values the emulator uses (whole Hz, 30 to 1000), and `--language` accepts language 30 (Ukrainian) like `RPS5E_SYSTEM_LANGUAGE`.

### Removed

- `RPS5E_PLAYGO_HACK`: games without PlayGo data get the console's answer (PlayGo not supported) instead of a made-up chunk list.

### Documentation

- `docs/COMMANDS.md`: every program, command and option; `docs/ENVIRONMENT.md`: every environment variable and what it does.

## [0.1.0-alpha.2] - 2026-10-27

### Fixed

- ASTRO BOT: tile-shaped holes, black sky and ghost images left from earlier frames ([#1](https://github.com/keklick1337/rps5e/issues/1)).
- Torn frames in videos played through AvPlayer (ASTRO BOT intro).
- Demon's Souls: a crash with an endless memory fault after a CPU write raced the GPU image cache.

## [0.1.0-alpha.1] - 2026-10-26

First public alpha.

### Emulation

- SELF/ELF loading of decrypted games and modules, relocation and import linking; the game's own modules (libc.prx, middleware) run natively.
- Native execution of guest x86-64 code with patched `fs:` TLS access; AMD-only instructions emulated or patched on Intel CPUs.
- HLE system libraries: kernel (memory, threads, synchronisation, files, events, sockets), libc, AGC GPU driver and command processor, video out, audio (AudioOut, AJM, Ngs2, ATRAC9, MP3), AvPlayer and video decoding (FFmpeg), pad, user and system services, save data, trophies, fonts, IME, SSL, HTTP, PlayGo and more.
- GPU: PM4 command processing on Vulkan 1.3, RDNA shader recompiler to SPIR-V, image cache with aliasing and CPU write tracking, BC decompression, mesh-shader primitive paths, HDR10 output.
- Fake PKG, PFS and PFSC containers: info, install and direct run.
- Windows build (x86_64-pc-windows-gnu) besides Linux.

### Launcher and usability

- Launcher (`rps5e`): game library with folders, packages, covers, search, grid and list views; per-game emulator settings; run logs.
- Controls: keyboard, mouse and SDL3 controllers with press-to-bind, global bindings with per-game input profiles, up to 4 players, DualSense adaptive triggers, light bar, touch pad and motion.
- In-game menu (F1 / PS button): controls with the DualSense picture, graphics, audio and system settings.
- Shader precompilation before the game starts, with progress; portable per-game shader cache export and import (CLI and launcher).
- Input recording and replay (`--record-input`, `--replay-input`, `scripts/input-test.sh`).
- Report issue: a prefilled GitHub issue (game, versions, OS, GPU, log tail) from the game menu and after a crash.
- Performance overlay (F10), frame dumps, GPU frame capture and replay.
- Application icon in the windows and in the Windows executables; Linux desktop entry installed by `make install`.

### Project

- The project is named RustPS5Emu (RPS5E): binaries `rps5e` and `rps5e-cli`, environment variables `RPS5E_*`, data, settings and caches under `rps5e` folders.
- Release packages: `scripts/package.sh` (Linux tarball, Windows zip) with a `dist` build profile without debug info.

[Unreleased]: https://github.com/keklick1337/rps5e/compare/v0.1.4-alpha...HEAD
[0.1.4-alpha]: https://github.com/keklick1337/rps5e/releases/tag/v0.1.4-alpha
[0.1.3-alpha]: https://github.com/keklick1337/rps5e/releases/tag/v0.1.3-alpha
[0.1.2-alpha]: https://github.com/keklick1337/rps5e/releases/tag/v0.1.2-alpha
[0.1.1-alpha]: https://github.com/keklick1337/rps5e/releases/tag/v0.1.1-alpha
[0.1.0-alpha.2]: https://github.com/keklick1337/rps5e/releases/tag/0.1.0-alpha.2
[0.1.0-alpha.1]: https://github.com/keklick1337/rps5e/releases/tag/0.1.0-alpha.1
