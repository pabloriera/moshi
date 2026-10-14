# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import asyncio
from contextlib import suppress
from dataclasses import dataclass
import inspect
import json
import random
import os
from pathlib import Path
import tarfile
import time
import secrets
import sys
import aiohttp
from aiohttp import web
from huggingface_hub import hf_hub_download
import numpy as np
import sentencepiece
import sphn
import torch
from .client_utils import log
from .models import loaders, MimiModel, LMModel, LMGen
from .run_inference import get_condition_tensors
from .perturbation import CyclicPerturbation
from .telemetry import RuntimeTelemetry


def seed_all(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


@dataclass
class ServerState:
    model_type: str
    mimi: MimiModel
    text_tokenizer: sentencepiece.SentencePieceProcessor
    lm_gen: LMGen
    lock: asyncio.Lock

    def __init__(self, model_type: str, mimi: MimiModel, text_tokenizer: sentencepiece.SentencePieceProcessor,
                 lm: LMModel, cfg_coef: float, device: str | torch.device, **kwargs):
        self.model_type = model_type
        self.mimi = mimi
        self.text_tokenizer = text_tokenizer
        self.perturbation = CyclicPerturbation(
            lm.transformer, lm.text_linear.in_features, mimi.frame_rate,
        )
        condition_tensors = get_condition_tensors(model_type, lm, batch_size=1, cfg_coef=cfg_coef)
        kwargs["measure_entropy"] = True
        self.lm_gen = LMGen(lm, cfg_coef=cfg_coef, condition_tensors=condition_tensors, **kwargs)
        # Register after perturbation hooks, so the displayed state includes
        # any live injection at the selected block.
        self.telemetry = RuntimeTelemetry(lm.transformer, lm.text_linear.in_features, mimi.frame_rate)

        self.device = device
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        self.lock = asyncio.Lock()
        self.active_ws: web.WebSocketResponse | None = None
        self.connection_generation = 0

        self.mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)

    def warmup(self):
        for chunk in range(4):
            chunk = torch.zeros(1, 1, self.frame_size, dtype=torch.float32, device=self.device)
            codes = self.mimi.encode(chunk)
            for c in range(codes.shape[-1]):
                tokens = self.lm_gen.step(codes[:, :, c: c + 1])
                if tokens is None:
                    continue
                _ = self.mimi.decode(tokens[:, 1:])

        torch.cuda.synchronize()
        self.telemetry.start()

    async def publish_telemetry(self, ws):
        while not ws.closed:
            for report in self.telemetry.poll():
                await ws.send_bytes(b"\x04" + json.dumps(report).encode("utf8"))
            await asyncio.sleep(0.04)

    def update_sampling_params(self, data: dict) -> dict:
        """Update generation sampling without resetting the active stream."""
        params = {
            "text_temperature": ("temp_text", float, 0.0, 1.2),
            "text_topk": ("top_k_text", int, 1, 500),
            "audio_temperature": ("temp", float, 0.0, 1.2),
            "audio_topk": ("top_k", int, 1, 500),
            "pad_mult": ("pad_mult", float, -10.0, 10.0),
        }
        applied = {}
        audio_changed = False
        for key, (attribute, cast, minimum, maximum) in params.items():
            if key not in data:
                continue
            try:
                value = cast(data[key])
            except (TypeError, ValueError):
                continue
            value = max(minimum, min(maximum, value))
            if attribute in ("temp", "top_k") and value != getattr(self.lm_gen, attribute, None):
                audio_changed = True
            setattr(self.lm_gen, attribute, value)
            applied[key] = value
        # Python sampling parameters are constants in the captured depth graph.
        # Recapture only that graph after an audio control change; keep temporal
        # KV caches and the observed trajectory intact. Text samples outside it.
        state = self.lm_gen._streaming_state
        if audio_changed and state is not None and state.graphed_depth is not None:
            state.graphed_depth.reset()
        return applied

    async def decode_and_send(
        self,
        tokens: torch.Tensor,
        ws: web.WebSocketResponse,
        opus_writer: sphn.OpusStreamWriter
    ):
        assert tokens.shape[1] == self.lm_gen.lm_model.dep_q + 1
        main_pcm = self.mimi.decode(tokens[:, 1:])
        main_pcm = main_pcm.cpu()
        opus_bytes = opus_writer.append_pcm(main_pcm[0, 0].numpy())
        if len(opus_bytes) > 0:
            await ws.send_bytes(b"\x01" + opus_bytes)
        text_token = tokens[0, 0, 0].item()
        if text_token not in (0, self.lm_gen.lm_model.text_padding_token_id):
            _text = self.text_tokenizer.id_to_piece(text_token)  # type: ignore
            _text = _text.replace("▁", " ")
            msg = b"\x02" + bytes(_text, encoding="utf8")
            log("info", f"text token '{_text}'")
            await ws.send_bytes(msg)

    async def recv_loop(
        self,
        ws: web.WebSocketResponse,
        opus_reader: sphn.OpusStreamReader,
        opus_writer: sphn.OpusStreamWriter
    ):
        all_pcm_data = None
        skip_frames = 1
        try:
            async for message in ws:
                if message.type == aiohttp.WSMsgType.ERROR:
                    log("error", f"{ws.exception()}")
                    break
                elif message.type == aiohttp.WSMsgType.CLOSED:
                    break
                elif message.type != aiohttp.WSMsgType.BINARY:
                    log("error", f"unexpected message type {message.type}")
                    continue
                message = message.data
                if not isinstance(message, bytes):
                    log("error", f"unsupported message type {type(message)}")
                    continue
                if len(message) == 0:
                    log("warning", "empty message")
                    continue
                kind = message[0]
                if kind == 1:  # audio
                    payload = message[1:]
                    pcm = opus_reader.append_bytes(payload)
                    if pcm.shape[-1] == 0:
                        continue
                    if all_pcm_data is None:
                        all_pcm_data = pcm
                    else:
                        all_pcm_data = np.concatenate((all_pcm_data, pcm))
                    while all_pcm_data.shape[-1] >= self.frame_size:
                        be = time.time()
                        chunk = all_pcm_data[: self.frame_size]
                        all_pcm_data = all_pcm_data[self.frame_size:]
                        chunk = torch.from_numpy(chunk)
                        chunk = chunk.to(device=self.device)[None, None]
                        codes = self.mimi.encode(chunk)
                        if skip_frames:
                            # The first input audio frame is ignored, as from the point of
                            # view of the model it is in the past. We still `mimi.encode` for simplicity,
                            # however as the first encoded frame has a specific structure (due to the left padding),
                            # we reset the streaming state of the encoder to reapply the padding on the next call.
                            self.mimi.reset_streaming()
                            skip_frames -= 1
                        for c in range(codes.shape[-1]):
                            self.perturbation.prepare_step()
                            tokens = self.lm_gen.step(codes[:, :, c: c + 1])
                            # Report every inference step, including startup
                            # steps with no output tokens yet. Keep this small:
                            # full configuration is sent only on control changes.
                            report = json.dumps({
                                "perturbation_phase": self.perturbation.phase,
                                "runtime_measurements": self.telemetry.measure(self.lm_gen.sampling_entropy),
                            })
                            await ws.send_bytes(b"\x04" + report.encode("utf8"))
                            # Yield even for a large incoming audio packet so
                            # completed projections can be published promptly.
                            await asyncio.sleep(0)
                            if tokens is None:
                                continue
                            await self.decode_and_send(tokens, ws, opus_writer)
                        log("info", f"frame handled in {1000 * (time.time() - be):.1f}ms")
                elif kind == 4:  # metadata
                    try:
                        metadata = json.loads(message[1:].decode("utf8"))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        log("warning", "invalid metadata message")
                        continue
                    if not isinstance(metadata, dict):
                        log("warning", "metadata message must be an object")
                        continue
                    telemetry = metadata.get("telemetry")
                    if isinstance(telemetry, dict):
                        report = {"telemetry_status": self.telemetry.update(telemetry), "trajectory_reset": True}
                        await ws.send_bytes(b"\x04" + json.dumps(report).encode("utf8"))
                    sampling = metadata.get("sampling")
                    if isinstance(sampling, dict):
                        applied = self.update_sampling_params(sampling)
                        log("info", f"updated sampling params: {applied}")
                        acknowledgement = json.dumps({
                            "sampling_applied": applied,
                            "stream_reset": False,
                            "pad_token_id": self.lm_gen.lm_model.text_padding_token_id,
                        }).encode("utf8")
                        await ws.send_bytes(b"\x04" + acknowledgement)
                    perturbation = metadata.get("perturbation")
                    if isinstance(perturbation, dict):
                        applied = self.perturbation.update(perturbation)
                        acknowledgement = json.dumps({
                            "perturbation_status": applied,
                            "stream_reset": False,
                        }).encode("utf8")
                        await ws.send_bytes(b"\x04" + acknowledgement)
                    if metadata.get("perturbation_reset_phase") is True:
                        self.perturbation.reset_phase()
                        report = json.dumps({"perturbation_status": self.perturbation.status()})
                        await ws.send_bytes(b"\x04" + report.encode("utf8"))
                elif kind == 3:  # control
                    if len(message) < 2:
                        log("warning", "empty control message")
                        continue
                    action = message[1]
                    if action != 3:  # restart
                        log("warning", f"unsupported control action {action}")
                        continue

                    # Keep the WebSocket and its continuous Ogg/Opus stream,
                    # but clear all model-side conversational state.
                    all_pcm_data = None
                    skip_frames = 1
                    self.mimi.reset_streaming()
                    self.lm_gen.reset_streaming()
                    self.perturbation.reset_phase()
                    self.telemetry.reset()
                    log("info", "streaming context reset")
                    acknowledgement = json.dumps({
                        "stream_reset": True,
                        "reason": "user",
                        "perturbation_phase": self.perturbation.phase,
                        "trajectory_reset": True,
                    }).encode("utf8")
                    await ws.send_bytes(b"\x04" + acknowledgement)
                else:
                    log("warning", f"unknown message kind {kind}")
        finally:
            log("info", "connection closed")

    async def handle_chat(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        log("info", "accepted connection")

        # Only one model stream can run at a time. A page refresh used to leave
        # the new socket waiting behind an older tab (or even a socket whose
        # client had already gone away), so the UI remained on CONNECTING.
        # Make the newest browser connection authoritative and release the
        # previous stream before acquiring the model lock.
        self.connection_generation += 1
        connection_generation = self.connection_generation
        previous_ws = self.active_ws
        if previous_ws is not None and not previous_ws.closed:
            log("info", "replacing the previous connection")
            await previous_ws.close(
                code=1001,
                message=b"replaced by a newer Moshi session",
            )

        async with self.lock:
            if connection_generation != self.connection_generation or ws.closed:
                if not ws.closed:
                    await ws.close(
                        code=1001,
                        message=b"superseded before model stream started",
                    )
                log("info", "discarded superseded connection")
                return ws

            self.active_ws = ws
            opus_writer = sphn.OpusStreamWriter(self.mimi.sample_rate)
            opus_reader = sphn.OpusStreamReader(self.mimi.sample_rate)
            self.mimi.reset_streaming()
            self.lm_gen.reset_streaming()
            self.perturbation.update({"enabled": False})
            self.perturbation.reset_phase()
            self.telemetry.reset()
            publisher = None
            try:
                # Send the handshake only after this socket owns the model.
                await ws.send_bytes(b"\x00")
                report = json.dumps({"perturbation_status": self.perturbation.status()})
                await ws.send_bytes(b"\x04" + report.encode("utf8"))
                report = json.dumps({"telemetry_status": self.telemetry.status()})
                await ws.send_bytes(b"\x04" + report.encode("utf8"))
                publisher = asyncio.create_task(self.publish_telemetry(ws))
                await self.recv_loop(ws, opus_reader, opus_writer)
            finally:
                if publisher is not None:
                    publisher.cancel()
                    with suppress(asyncio.CancelledError, ConnectionResetError):
                        await publisher
                if self.active_ws is ws:
                    self.active_ws = None
        log("info", "done with connection")
        return ws


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost", type=str)
    parser.add_argument("--port", default=8998, type=int)
    parser.add_argument("--static", type=str)
    parser.add_argument("--gradio-tunnel", action='store_true', help='Activate a gradio tunnel.')
    parser.add_argument("--gradio-tunnel-token",
                        help='Provide a custom (secret) token here to keep getting the same URL.')

    parser.add_argument("--tokenizer", type=str, help="Path to a local tokenizer file.")
    parser.add_argument("--moshi-weight", type=str, help="Path to a local checkpoint file for Moshi.")
    parser.add_argument("--mimi-weight", type=str, help="Path to a local checkpoint file for Mimi.")
    parser.add_argument("--hf-repo", type=str, default=loaders.DEFAULT_REPO,
                        help="HF repo to look into, defaults Moshiko. "
                             "Use this to select a different pre-trained model.")
    parser.add_argument("--lora-weight", type=str, help="Path to a local checkpoint file for LoRA.", default=None)
    parser.add_argument("--config-path", type=str, help="Path to a local config file.", default=None)
    parser.add_argument("--cfg-coef", type=float, default=1., help="CFG coefficient.")
    parser.add_argument("--device", type=str, default="cuda", help="Device on which to run, defaults to 'cuda'.")
    parser.add_argument("--no_fuse_lora", action="store_false", dest="fuse_lora", default=True,
                        help="Do not fuse LoRA layers intot Linear layers.")
    parser.add_argument("--half", action="store_const", const=torch.float16, default=torch.bfloat16,
                        dest="dtype", help="Run inference with float16, not bfloat16, better for old GPUs.")
    parser.add_argument(
        "--ssl",
        type=str,
        help=(
            "use https instead of http, this flag should point to a directory "
            "that contains valid key.pem and cert.pem files"
        )
    )

    args = parser.parse_args()
    seed_all(42424242)

    setup_tunnel = None
    tunnel_token = ''
    if args.gradio_tunnel:
        try:
            from gradio import networking  # type: ignore
        except ImportError:
            log("error", "Cannot find gradio which is required to activate a tunnel. "
                         "Please install with `pip install gradio`.")
            sys.exit(1)
        setup_tunnel = networking.setup_tunnel
        if args.gradio_tunnel_token is None:
            tunnel_token = secrets.token_urlsafe(32)
        else:
            tunnel_token = args.gradio_tunnel_token

    log("info", "retrieving checkpoint")
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
        args.hf_repo, args.moshi_weight, args.mimi_weight, args.tokenizer,
        lora_weights=args.lora_weight, config_path=args.config_path)
    log("info", "loading mimi")
    mimi = checkpoint_info.get_mimi(device=args.device)
    log("info", "mimi loaded")

    text_tokenizer = checkpoint_info.get_text_tokenizer()

    log("info", "loading moshi")
    lm = checkpoint_info.get_moshi(device=args.device, dtype=args.dtype, fuse_lora=args.fuse_lora)
    log("info", "moshi loaded")

    state = ServerState(checkpoint_info.model_type, mimi, text_tokenizer, lm, args.cfg_coef, args.device,
                        **checkpoint_info.lm_gen_config)
    pad_token_id = state.lm_gen.lm_model.text_padding_token_id
    log(
        "info",
        f"text PAD token: id={pad_token_id}, "
        f"piece={text_tokenizer.id_to_piece(pad_token_id)!r}",
    )
    log("info", "warming up the model")
    state.warmup()
    app = web.Application()

    async def shutdown_stream(_app):
        if state.active_ws is not None and not state.active_ws.closed:
            await state.active_ws.close(code=1001, message=b"server restarting")

    async def cleanup_telemetry(_app):
        state.telemetry.close()

    app.on_cleanup.append(cleanup_telemetry)
    app.on_shutdown.append(shutdown_stream)
    app.router.add_get("/api/chat", state.handle_chat)
    static_path: None | str = None
    if args.static is None:
        log("info", "retrieving the static content")
        dist_tgz = hf_hub_download("kyutai/moshi-artifacts", "dist.tgz")
        dist_tgz = Path(dist_tgz)
        dist = dist_tgz.parent / "dist"
        if not dist.exists():
            with tarfile.open(dist_tgz, "r:gz") as tar:
                tar.extractall(path=dist_tgz.parent)
        static_path = str(dist)
    elif args.static != "none":
        # When set to the "none" string, we don't serve any static content.
        static_path = args.static
    if static_path is not None:
        async def handle_root(_):
            return web.FileResponse(os.path.join(static_path, "index.html"))

        log("info", f"serving static content from {static_path}")
        app.router.add_get("/", handle_root)
        app.router.add_static(
            "/", path=static_path, follow_symlinks=True, name="static"
        )
    protocol = "http"
    ssl_context = None
    if args.ssl is not None:
        import ssl

        ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        cert_file = os.path.join(args.ssl, "cert.pem")
        key_file = os.path.join(args.ssl, "key.pem")
        ssl_context.load_cert_chain(certfile=cert_file, keyfile=key_file)
        protocol = "https"

    log("info", f"Access the Web UI directly at {protocol}://{args.host}:{args.port}")
    if setup_tunnel is not None:
        tunnel_kwargs = {}
        if "share_server_tls_certificate" in inspect.signature(setup_tunnel).parameters:
            tunnel_kwargs["share_server_tls_certificate"] = None
        tunnel = setup_tunnel('localhost', args.port, tunnel_token, None, **tunnel_kwargs)  # type: ignore
        log("info", f"Tunnel started, if executing on a remote GPU, you can use {tunnel}.")
        log("info", "Note that this tunnel goes through the US and you might experience high latency in Europe.")
    web.run_app(app, host=args.host , port=args.port, ssl_context=ssl_context)


if __name__ == "__main__":
    with torch.no_grad():
        main()
