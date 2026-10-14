# Moshi: a speech-text foundation model for real time dialogue

![precommit badge](https://github.com/kyutai-labs/moshi/workflows/precommit/badge.svg)
![rust ci badge](https://github.com/kyutai-labs/moshi/workflows/Rust%20CI/badge.svg)

[[Read the paper]][moshi] [[Demo]](https://moshi.chat) [[Hugging Face]](https://huggingface.co/collections/kyutai/moshi-v01-release-66eaeaf3302bef6bd9ad7acd)

[Moshi][moshi] is a speech-text foundation model and **full-duplex** spoken dialogue framework.
It uses [Mimi][moshi], a state-of-the-art streaming neural audio codec.
[Talk to Moshi](https://moshi.chat) now in our live demo.

## Start the customized web UI and Moshi server

This repository's customized audio-loop UI and the Python Moshi backend must be
served together. From inside the development container, use the repository
launcher:

```bash
cd /workspaces/moshi
./start-server.sh
```

The launcher performs two required steps:

1. It builds `client/` into `client/dist/`.
2. It starts `python -m moshi.server` with `--static client/dist`, binding to
   `0.0.0.0:8998`.

There is only one normal web address and one process. The same server provides
the UI at `/` and the streaming WebSocket at `/api/chat`. Do **not** start Vite
on port `5173` for normal use, and do not run `python -m moshi.server` without
the `--static` option: either approach can show the wrong UI or leave the UI
without a working model stream.

Wait until the terminal prints both `moshi loaded` and:

```text
Running on http://0.0.0.0:8998
```

Then open the Docker/container host's LAN address from the remote computer:

```text
http://<container-host-LAN-IP>:8998
```

For the current host this may look like `http://10.1.100.112:8998`; use the
actual LAN address assigned to the machine running the container. Port `8998`
must be published by the container and allowed by the host firewall.

The Python server uses plain HTTP unless TLS was explicitly enabled. Opening an
HTTP server with an `https://` URL produces `ERR_SSL_PROTOCOL_ERROR`. Audio-file
and loop-machine input work over HTTP, but browsers generally require a secure
context for microphone or system-audio capture from a remote address.

To enable HTTPS, create a certificate trusted by the remote computer with
`cert.pem` and `key.pem`, then start with:

```bash
MOSHI_SSL_DIR=/absolute/path/to/certificates ./start-server.sh
```

Only in that case should the remote browser use
`https://<container-host-LAN-IP>:8998`.

For automatic container startup, `.devcontainer/start-services.sh` calls the
same launcher in background mode. Its model log is
`/tmp/moshi-server-8998.log`. To start it manually in the background:

```bash
./start-server.sh --background
tail -f /tmp/moshi-server-8998.log
```

Quick diagnostics:

```bash
# The custom UI title should be present.
curl --fail http://127.0.0.1:8998/ | grep "Moshi Live Lab"

# The server must listen on every container interface for remote access.
ss -ltnp 'sport = :8998'

# Run this from the remote computer; HTTP 200 confirms network reachability.
curl --fail --output /dev/null --write-out '%{http_code}\n' \
  http://<container-host-LAN-IP>:8998/
```

The model is ready only after the page status changes from
`Opening model stream` to `Model stream live`. If the page loads but the stream
does not turn live, check the server terminal—or
`/tmp/moshi-server-8998.log` for a background launch—for model-loading,
WebSocket, CUDA, or out-of-memory errors.

### Hidden-state trajectory experiments

The **Desviá.** module (code `TRAJ`) in the bottom control deck adds a seeded cyclic
trajectory to the residual hidden state **after a selected temporal-transformer
block**, before text prediction and audio-depth generation. It is an inference
experiment, not a model-training change or an audio effect applied afterwards.

- **ON** switch: enable or immediately bypass injection. It starts OFF after
  a browser reload; the other settings and MIDI mappings are saved locally.
- **TRAJ** keys (`ORB`, `PUL`, `WLK`) choose one of three presets:
  - **Smooth orbit**: the original closed Fourier trajectory in seeded random
    directions, not independently sampled noise each frame.
  - **Cycle-start pulse**: a fixed seeded unit-RMS vector is injected for only
    the first model frame of each cycle (80 ms at 12.5 Hz); every other frame
    has exactly zero injection, with no smoothing tail. At high rates, cycle
    starts are quantized to the next model frame. Zero injection does not mean
    zero output audio: Moshi still runs and retains the effects in its context.
  - **Looping random walk**: start at a seeded unit-RMS vector, add one new
    seeded random-direction increment each model frame, then jump back to the
    first vector at the next cycle. The same direction sequence repeats each
    cycle; rate changes preserve phase and the current walk position until
    the next boundary. **STEP** (walk step) sets each increment's RMS (0–1) before
    overall strength; zero holds the first vector. Drift accumulates, so longer
    cycles and larger steps can produce much larger perturbations.
  Preset selection and STEP support MIDI learn; the selector uses CC
  0–42 for orbit, 43–84 for pulse, and 85–127 for walk. The selected preset,
  its settings, and MIDI mappings survive browser reloads (injection starts OFF).
- **STR** (strength): overall trajectory gain relative to that frame's hidden-state RMS,
  from 0 (exact bypass) to 10. Start around 0.05–0.2; large values can produce
  incoherent speech, silence, or unstable sounds. Strength ramps over about
  150 ms for orbit/walk, except OFF/zero, which bypass immediately. Pulse uses
  the full chosen strength immediately at each kick. Its base vector and the
  smooth orbit have unit RMS; the walk's magnitude varies as steps accumulate.
- **RATE** (cycle rate): 0.01–6 Hz (100–0.167 seconds per orbit). Phase advances with model
  frames, including silent input, not a separate wall-clock timer. Rate changes
  keep phase continuous; if inference stops, phase stops too.
  The phase ring in the module heading receives one update per inference step (12.5 updates/second
  at real-time speed), including startup steps before output audio is available.
  Its marker is interpolated at the browser animation rate with up to one
  model step of display lag, rather than snapping between packets. Cycle wraps
  are interpolated forward; the display stops at the last received phase if
  inference pauses. This visual smoothing does not alter the model clock.
- **LAYER**: one-based transformer block index. **HARM** (harmonics, smooth orbit only): 1–6 Fourier
  harmonics in seeded random orthonormal directions (a 2–12 dimensional
  subspace of the full hidden-state space). The direction repeats exactly;
  its magnitude follows the current activation RMS. Higher harmonics can
  exceed the model-frame Nyquist frequency and alias, intentionally permitting
  more irregular experimental motion at fast cycle rates.
- **New path** (the dice key in the module heading) changes the seed/directions
  without resetting phase. **Phase ↺** (the arrow key beside it) restarts only the trajectory, not the conversation. Preset changes
  also preserve phase and context. Layer/path/preset changes
  can introduce abrupt changes; strength/rate modulation is smoother.

All controls apply on upcoming model steps without clearing KV caches or
recapturing CUDA graphs. Turning injection off does not undo effects already
stored in the conversation; use **FLUSH** (in **Olvidá.**) for a fresh baseline
(trajectory settings are preserved, but its phase is reset). Every control's full
explanation is in its tooltip; knobs take Shift + drag for fine adjustment.

The forcing is periodic, but generated sound need not be: autoregressive
context and sampling can respond non-periodically. Implementation:
[`moshi/moshi/perturbation.py`](moshi/moshi/perturbation.py), wired into
[`moshi/moshi/server.py`](moshi/moshi/server.py) before CUDA graph warmup and
updated before each generation step.

### Runtime entropy and the 2D/3D state view

The central **ESTADO** panel shows time-series oscilloscopes alongside the
diffusion map: three compact signal columns and a fourth column for the map.
Each **L01…L32** channel reports the L2 norm of the full residual vector after
that temporal-transformer block, including any injected perturbation. **TXT**
and **A1…A8** report sampling entropy for text and each generated audio codebook.
Entropy applies to these probability distributions; residual states are not
treated as probabilities. These are token-distribution entropies, not waveform
entropy or the surprise of the chosen token. **Compará.** shows only the
input/output envelope Lissajous.

Choose **‖h‖**, **H · bits**, or **‖h‖ + H** to display norms, entropies, or both.
Each signal occupies a compact 16–18 px row within its column, with time
increasing left to right and the latest frame at the right edge. On desktop,
the 41 channels occupy 14 rows; narrower screens use fewer signal columns and
stack the map below. The layout reserves enough height for every signal, so the
scopes need no internal scrolling. All channels share the selected
20/50-frame time window; hover a trace for its exact value and observed range.
Numbers retain their original values, while each curve is centered and scaled
independently to show variations. The display range follows the median and
10th/90th percentiles of the latest 20 samples, adapting gradually on each
received model frame. Tiny entropy changes remain visible; older outliers
are compressed at the edges with a smooth bounded transform. Constant values
remain flat, and the range stops moving when inference stops. Very small values
use scientific notation in the readout. Projection pause stops trajectory
collection but norms and entropy continue to update. Changing the observed layer
highlights its norm channel without discarding measurement history.

The monologue starts in a compact view with only the latest two lines of large
text; the small history lines are removed. Its **+ / −** button expands or shrinks
the text area. Compact mode gives the freed height to the state/oscilloscope view.

Press **Space** (or the square stop key in the header) to stop the engine:
sample sequencing, microphone/system capture, model input streaming, and queued
input/output playback stop together. Scope and trajectory updates stop as well.
The model connection and conversation are retained. Trigger a pad, start the
sequence, or activate capture to resume. Space is ignored while typing or using
a select/input field, and holding it does not retrigger the engine.

Moshi measures Shannon entropy, `−Σ p log₂(p)`, at every sampling procedure,
after temperature, text PAD bias, and top-k/top-p filtering and renormalisation.
Greedy sampling has zero entropy. Each inference frame reports text plus all
eight generated audio codebooks, including startup frames. Measurement does
not consume sampling RNG. Tensor buffers update during CUDA graph replay;
audio sampling control changes refresh only the depth graph, preserving
conversation context.

The map's **2D/3D** views project the last **20 or 50 frames** of a
selected temporal-transformer layer, after any perturbation at that block.
Select **2D** or **3D**, the layer with **L**, and the window. Switching dimensions
reprojects the collected frames into two or three diffusion coordinates without
resetting model context. Drag the 3D drawing to rotate the camera; 2D uses a
flat view. The pause key suspends collection/projection while inference and
entropy measurement continue. Layer/window changes begin a fresh window;
**FLUSH** and a new connection also clear the trajectory. At 12.5 model frames
per second, the buffers cover approximately 1.6 and 4 seconds. No new frames
means no new motion in the plot.

The worker uses [diffusion maps](https://doi.org/10.1016/j.acha.2006.04.006), a
nonlinear spectral reduction with an adaptive-distance kernel over the full
hidden vectors. Overlapping frames align the two or three axes between windows to
reduce arbitrary rotations/reflections. Windows are centered and scaled for
display; axes have no fixed semantic meaning, and distances between different
windows are not directly comparable. The plot is a local view of model state.

A dedicated CPU process runs projection, with one native BLAS thread and at
most one job in flight. Inference never waits for a projection result; the next
job always uses the newest window. On startup, the worker benchmarks 20 and 50
frames at the loaded model's hidden dimension. It recommends 50 if its p95
projection time fits within half a model frame, otherwise 20, and caps updates
at 5 Hz. Manual window selection overrides that recommendation. Hover the
view heading for calibration, or its timing readout for live worker latency.

Local tests with Moshiko 7B, 4096-dimensional states and an RTX 3090 measured:

| Window | Approximate history | Live projection p95 |
| --- | --- | --- |
| 20 frames | 1.6 s | 0.8 ms |
| 50 frames | 4 s | 1.6–1.7 ms |

The 50-frame worker measured about 3 ms p95 including queue transfer. A local
120-frame silence test averaged 35.8 ms per complete frame with telemetry,
within the 80 ms real-time frame budget; the earlier baseline averaged 35.2 ms.
These are local timings, not guarantees for another machine or network.
The default is therefore **50 frames at up to 5 Hz**. To repeat the standalone
projection calibration with representative correlated vectors:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python - <<'PY'
from moshi.projection import benchmark_projection
print(benchmark_projection(4096, repeats=60))
PY
```

Implementation: [`moshi/moshi/telemetry.py`](moshi/moshi/telemetry.py),
[`moshi/moshi/projection.py`](moshi/moshi/projection.py), and the sampling
instrumentation in [`moshi/moshi/utils/sampling.py`](moshi/moshi/utils/sampling.py).

## Organisation of the repository

There are three separate versions of the Moshi inference stack in this repo.


- **[PyTorch](#pytorch-implementation): for research and tinkering.** The code is in the [`moshi/`](moshi/) directory.
- **[MLX](#mlx-implementation-for-local-inference-on-macos): for on-device inference on iPhone and Mac.** The code is in the [`moshi_mlx/`](moshi_mlx/) directory.
- **[Rust](#rust-implementation): for production.** The code is in the [`rust/`](rust/) directory.
    This contains in particular a Mimi implementation in Rust, with Python bindings available
    as `rustymimi`.

Finally, the code for the web UI client used in the [Moshi demo](https://moshi.chat) is provided in the [`client/`](client/) directory.

If you want to fine tune Moshi, head out to [kyutai-labs/moshi-finetune](https://github.com/kyutai-labs/moshi-finetune).

### Other Kyutai models

The Moshi codebase is also used to run related models from Kyutai that use a multi-stream architecture similar to Moshi:
- **Hibiki: simultaneous speech translation.** Check out the [Hibiki repo](https://github.com/kyutai-labs/hibiki) for more info.
- **Kyutai Text-To-Speech and Speech-To-Text.** Check out the [Delayed Streams Modeling repo](https://github.com/kyutai-labs/delayed-streams-modeling) for more info.

## Model architecture

Moshi models **two streams of audio**: one corresponds to Moshi speaking, and the other one to the user speaking.
Along with these two audio streams, Moshi predicts text tokens corresponding to its own speech, its **inner monologue**,
which greatly improves the quality of its generation.
A small Depth Transformer models inter-codebook dependencies for a given time step,
while a large, 7B-parameter Temporal Transformer models the temporal dependencies. Moshi achieves a theoretical latency
of 160ms (80ms for the frame size of Mimi + 80ms of acoustic delay), with a practical overall latency as low as 200ms on an L4 GPU.

<p align="center">
<img src="./moshi.png" alt="Schema representing the structure of Moshi. Moshi models two streams of audio:
    one corresponds to Moshi, and the other one to the user. At inference, the audio stream of the user is taken from the audio input, and the audio stream for Moshi is sampled from the model's output. Along that, Moshi predicts text tokens corresponding to its own speech for improved accuracy. A small Depth Transformer models inter codebook dependencies for a given step."
width="650px"></p>

### Mimi

Mimi is a neural audio codec that processes 24 kHz audio, down to a 12.5 Hz representation
with a bandwidth of 1.1 kbps, in a fully streaming manner (latency of 80ms, the frame size),
yet performs better than existing, non-streaming, codecs like
[SpeechTokenizer](https://github.com/ZhangXInFD/SpeechTokenizer) (50 Hz, 4kbps), or [SemantiCodec](https://github.com/haoheliu/SemantiCodec-inference) (50 Hz, 1.3kbps).

Mimi builds on previous neural audio codecs such as [SoundStream](https://arxiv.org/abs/2107.03312)
and [EnCodec](https://github.com/facebookresearch/encodec), adding a Transformer both in the encoder and decoder,
and adapting the strides to match an overall frame rate of 12.5 Hz. This allows Mimi to get closer to the
average frame rate of text tokens (~3-4 Hz), and limit the number of autoregressive steps in Moshi.
Similarly to SpeechTokenizer, Mimi uses a distillation loss so that the first codebook tokens match
a self-supervised representation from [WavLM](https://arxiv.org/abs/2110.13900), which allows modeling semantic and acoustic information with a single model. Finally, and similarly to [EBEN](https://arxiv.org/pdf/2210.14090),
Mimi uses **only an adversarial training loss**, along with feature matching, showing strong improvements in terms of
subjective quality despite its low bitrate.

<p align="center">
<img src="./mimi.png" alt="Schema representing the structure of Mimi, our proposed neural codec. Mimi contains a Transformer
in both its encoder and decoder, and achieves a frame rate closer to that of text tokens. This allows us to reduce
the number of auto-regressive steps taken by Moshi, thus reducing the latency of the model."
width="800px"></p>

## Models

We release three models:
- Moshi fine-tuned on a male synthetic voice (Moshiko),
- Moshi fine-tuned on a female synthetic voice (Moshika),
- Mimi, our speech codec.

Depending on the backend, the file format and quantization available will vary. Here is the list
of the HuggingFace repo with each model. Mimi is bundled in each of those, and always use the same checkpoint format.

- Moshika for PyTorch (bf16, int8): [kyutai/moshika-pytorch-bf16](https://huggingface.co/kyutai/moshika-pytorch-bf16), [kyutai/moshika-pytorch-q8](https://huggingface.co/kyutai/moshika-pytorch-q8) (experimental).
- Moshiko for PyTorch (bf16, int8): [kyutai/moshiko-pytorch-bf16](https://huggingface.co/kyutai/moshiko-pytorch-bf16), [kyutai/moshiko-pytorch-q8](https://huggingface.co/kyutai/moshiko-pytorch-q8) (experimental).
- Moshika for MLX (int4, int8, bf16): [kyutai/moshika-mlx-q4](https://huggingface.co/kyutai/moshika-mlx-q4), [kyutai/moshika-mlx-q8](https://huggingface.co/kyutai/moshika-mlx-q8),  [kyutai/moshika-mlx-bf16](https://huggingface.co/kyutai/moshika-mlx-bf16).
- Moshiko for MLX (int4, int8, bf16): [kyutai/moshiko-mlx-q4](https://huggingface.co/kyutai/moshiko-mlx-q4), [kyutai/moshiko-mlx-q8](https://huggingface.co/kyutai/moshiko-mlx-q8),  [kyutai/moshiko-mlx-bf16](https://huggingface.co/kyutai/moshiko-mlx-bf16).
- Moshika for Rust/Candle (int8, bf16): [kyutai/moshika-candle-q8](https://huggingface.co/kyutai/moshika-candle-q8),  [kyutai/moshika-mlx-bf16](https://huggingface.co/kyutai/moshika-candle-bf16).
- Moshiko for Rust/Candle (int8, bf16): [kyutai/moshiko-candle-q8](https://huggingface.co/kyutai/moshiko-candle-q8),  [kyutai/moshiko-mlx-bf16](https://huggingface.co/kyutai/moshiko-candle-bf16).

All models are released under the CC-BY 4.0 license.

## Requirements

You will need at least Python 3.10, with 3.12 recommended. For specific requirements, please check the individual backends
directories. You can install the PyTorch and MLX clients with the following:

```bash
pip install -U moshi      # moshi PyTorch, from PyPI
pip install -U moshi_mlx  # moshi MLX, from PyPI, best with Python 3.12.
# Or the bleeding edge versions for Moshi and Moshi-MLX.
pip install -U -e "git+https://git@github.com/kyutai-labs/moshi.git#egg=moshi&subdirectory=moshi"
pip install -U -e "git+https://git@github.com/kyutai-labs/moshi.git#egg=moshi_mlx&subdirectory=moshi_mlx"

pip install rustymimi  # mimi, rust implementation with Python bindings from PyPI
```

If you are not using Python 3.12, you might get an error when installing
`moshi_mlx` or `rustymimi` (which `moshi_mlx` depends on). Then, you will need to install the [Rust toolchain](https://rustup.rs/), or switch to Python 3.12.

While we hope that the present codebase will work on Windows, we do not provide official support for it.
We have tested the MLX version on a MacBook Pro M3. At the moment, we do not support quantization
for the PyTorch version, so you will need a GPU with a significant amount of memory (24GB).

For using the Rust backend, you will need a recent version of the [Rust toolchain](https://rustup.rs/).
To compile GPU support, you will also need the [CUDA](https://developer.nvidia.com/cuda-toolkit) properly installed for your GPU, in particular with `nvcc`.

## PyTorch implementation

The PyTorch based API can be found in the `moshi` directory. It provides a streaming
version of the audio tokenizer (mimi) and the language model (moshi).

In order to run in interactive mode, you need to start a server which will run
the model. For this repository's customized UI, use `./start-server.sh` as
described at the top of this README.

The equivalent explicit command, after building `client/`, is:

```bash
cd /workspaces/moshi
npm --prefix client run build
.venv/bin/python -m moshi.server \
  --host 0.0.0.0 \
  --port 8998 \
  --static client/dist \
  --hf-repo kyutai/moshiko-pytorch-bf16
```

Omitting `--static client/dist` makes Moshi download and serve the upstream
Kyutai client instead of the customized UI in this checkout. Omitting
`--host 0.0.0.0` binds only to localhost and prevents direct LAN access.

Then access the web UI on [localhost:8998](http://localhost:8998), or use the
container host's LAN IP from another computer.
If your GPU is on a distant machine this will not work because for security reasons, websites using HTTP
are not allowed to use the microphone. There are two ways to get around this:
- Forward the remote 8998 port to your localhost using ssh `-L` flag. Then
  connects to [localhost:8998](http://localhost:8998) as mentioned previously.
- Use the `--gradio-tunnel` argument, setting up a tunnel with a URL accessible from anywhere.
  Keep in mind that this tunnel goes through the US and can add significant
  latency (up to 500ms from Europe). You can use `--gradio-tunnel-token` to set a
  fixed secret token and reuse the same address over time.

You can use `--hf-repo` to select a different pretrained model, by setting the proper Hugging Face repository.

Accessing a server that is not localhost via http may cause issues with using
the microphone in the web UI (in some browsers this is only allowed using
https).

A command-line client is also available, as
```bash
python -m moshi.client [--url URL_TO_GRADIO]
```
However note that, unlike the web browser, this client is barebones: it does not perform any echo cancellation,
nor does it try to compensate for a growing lag by skipping frames.

For more information, in particular on how to use the API directly, please
checkout [moshi/README.md](moshi/README.md).

## MLX implementation for local inference on macOS

Once you have installed `moshi_mlx`, you can run
```bash
python -m moshi_mlx.local -q 4   # weights quantized to 4 bits
python -m moshi_mlx.local -q 8   # weights quantized to 8 bits
# And using a different pretrained model:
python -m moshi_mlx.local -q 4 --hf-repo kyutai/moshika-mlx-q4
python -m moshi_mlx.local -q 8 --hf-repo kyutai/moshika-mlx-q8
# be careful to always match the `-q` and `--hf-repo` flag.
```

This command line interface is also barebone. It does not perform any echo cancellation,
nor does it try to compensate for a growing lag by skipping frames.

Alternatively you can run `python -m moshi_mlx.local_web` to use
the web UI, the connection is via http and will be at [localhost:8998](http://localhost:8998).


## Rust implementation

In order to run the Rust inference server, use the following command from within
the `rust` directory:

```bash
cargo run --features cuda --bin moshi-backend -r -- --config moshi-backend/config.json standalone
```

When using macOS, you can replace `--features cuda` with `--features metal`.

Alternatively you can use `config-q8.json` rather than `config.json` to use the
quantized q8 model. You can select a different pretrained model, e.g. Moshika,
by changing the `"hf_repo"` key in either file.

Once the server has printed 'standalone worker listening', you can use the web
UI. By default the Rust server uses https so it will be at
[localhost:8998](https://localhost:8998).

You will get warnings about the site being unsafe. When using chrome you
can bypass these by selecting "Details" or "Advanced", then "Visit this unsafe
site" or "Proceed to localhost (unsafe)".

## Clients

We recommend using the web UI as it provides additional echo cancellation that helps
the overall model quality. Note that most commands will directly serve this UI
in the provided URL, and there is in general nothing more to do.

Alternatively, we provide command line interfaces
for the Rust and Python versions, the protocol is the same as with the web UI so
there is nothing to change on the server side.

For reference, here is the list of clients for Moshi.

### Web UI

The web UI can be built from this repo via the
following steps (these will require `npm` being installed).
```bash
cd client
npm install
npm run build
```

The web UI can then be found in the `client/dist` directory.

### Rust Command Line

From within the `rust` directory, run the following:
```bash
cargo run --bin moshi-cli -r -- tui --host localhost
```

### Python with PyTorch

```bash
python -m moshi.client
```

### Gradio Demo

You can launch a Gradio demo locally with the following command:

```bash
python -m moshi.client_gradio --url <moshi-server-url>
```

Prior to running the Gradio demo, please install `gradio-webrtc>=0.0.18`.

### Docker Compose (CUDA only)

```bash
docker compose up --build
```

This runs only the model service; no public tunnel is created. By default Docker
publishes it on all host interfaces at port `8998`, so another machine on the
same LAN can use:

```text
http://<model-host-lan-ip>:8998
```

The current Compose image uses Moshi's upstream downloaded client. Use
`./start-server.sh` inside the development container when you need the
customized audio-loop UI from `client/`.

Downloaded Hugging Face models are stored on the Docker host at
`/mnt/data/huggingface_moshi/` and reused by both Compose and the development
container. Set `HF_CACHE_DIR` for Compose if you need to use a different host
path.

You can verify the route without loading the model UI:

```bash
# On the model host, confirm Docker published 0.0.0.0:8998.
docker compose ps

# On another LAN machine, confirm that the HTTP endpoint is reachable.
curl --fail --output /dev/null http://<model-host-lan-ip>:8998/
```

If the second command cannot connect, allow inbound TCP port `8998` from your
LAN subnet in the model host's firewall. Router port forwarding is neither
required nor recommended. Moshi does not authenticate clients, so restrict the
firewall rule to trusted machines or the local subnet.

The bind address, published port, and model are configurable:

```bash
MOSHI_BIND_ADDRESS=192.168.1.50 MOSHI_PORT=8998 \
HF_REPO=kyutai/moshika-pytorch-bf16 \
docker compose up --build
```

Binding to the host's LAN address instead of `0.0.0.0` avoids listening on any
other host network. Use `127.0.0.1` when LAN access is not wanted.

Browsers do not normally grant microphone access to an HTTP page loaded from a
non-localhost address. For the web voice UI, create a certificate trusted by the
client machine, containing the model host's LAN hostname or IP, as
`.certs/cert.pem` and `.certs/key.pem`. Then use Moshi's native TLS support:

```bash
docker compose -f docker-compose.yml -f docker-compose.tls.yml up --build
```

Open `https://<model-host-lan-name-or-ip>:8998` and make sure the browser reports
a trusted connection before granting microphone access. A local CA tool such as
`mkcert` can create the files, but its root CA must also be installed as trusted
on every client machine. Set `MOSHI_TLS_DIR` if the certificate directory is
elsewhere.

Docker Compose requires an NVIDIA GPU and
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

## Development

### Agent development container

The repository includes a [development container](.devcontainer/README.md) for
running coding agents and development tools with less access to the host. It
uses a non-root user, drops Linux capabilities, disables privilege escalation,
mounts this repository and `/mnt/data/huggingface_moshi/`, and does not mount the
host Docker socket or home directory. NVIDIA GPU access and LAN ports `8998` and
`5173` remain enabled for model and web UI development.

If you wish to install from a clone of this repository, maybe to further develop Moshi, you can do the following:
```bash
# From the root of the clone of the repo
pip install -e 'moshi[dev]'
pip install -e 'moshi_mlx[dev]'
pre-commit install
```

If you wish to build locally `rustymimi` (assuming you have Rust properly installed):
```bash
pip install maturin
maturin dev -r -m rust/mimi-pyo3/Cargo.toml
```

## FAQ

Checkout the [Frequently Asked Questions](FAQ.md) section before opening an issue.


## License

The present code is provided under the MIT license for the Python parts, and Apache license for the Rust backend.
The web client code is provided under the MIT license.
Note that parts of this code is based on [AudioCraft](https://github.com/facebookresearch/audiocraft), released under
the MIT license.

The weights for the models are released under the CC-BY 4.0 license.

## Citation

If you use either Mimi or Moshi, please cite the following paper,

```
@techreport{kyutai2024moshi,
      title={Moshi: a speech-text foundation model for real-time dialogue},
      author={Alexandre D\'efossez and Laurent Mazar\'e and Manu Orsini and
      Am\'elie Royer and Patrick P\'erez and Herv\'e J\'egou and Edouard Grave and Neil Zeghidour},
      year={2024},
      eprint={2410.00037},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2410.00037},
}
```

[moshi]: https://arxiv.org/abs/2410.00037
