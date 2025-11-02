#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pipecat Local setup Example.

The example runs a simple voice AI bot that you can connect to using your
browser and speak with it. You can also deploy this bot to Pipecat local.

Required AI services:
# Deepgram (Speech-to-Text)
✅ Local Whisper STT (Speech-to-Text)
#X- OpenAI (LLM)
✅ Ollama (LocalLLM
# X Cartesia (Text-to-Speech)
✅ Piper (Text-to-Speech)

Run the bot using::

    uv run botLocal.py
"""

from pipecat.services.whisper.stt import WhisperSTTService, Model
from pipecat.services.piper.tts import PiperTTSService
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.runner.utils import create_transport
from pipecat.runner.types import RunnerArguments
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.pipeline import Pipeline
from pipecat.frames.frames import LLMRunFrame
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
import os
import aiohttp
import nltk
from dotenv import load_dotenv
from loguru import logger

print("🚀 Starting Pipecat bot...")
print("⏳ Loading models and imports (20 seconds, first run only)\n")

logger.info("Loading Local Smart Turn Analyzer V3...")

logger.info("✅ Local Smart Turn Analyzer V3 loaded")
logger.info("Loading Silero VAD model...")

logger.info("✅ Silero VAD model loaded")


logger.info("Loading pipeline components...")

# Ollama Local LLM Service
# Piper Local TTS Service

# whisper local imports

logger.info("✅ All components loaded successfully!")

load_dotenv(override=True)


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot")

    # stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
    stt = WhisperSTTService(
        # Select model size (tiny, base, small, medium, large, etc.)
        model=Model.MEDIUM,
        # Uses GPU (cuda/mlx) if available, otherwise CPU
        device='auto'
    )

    tts = PiperTTSService(
        # You MUST run a separate Piper server at this URL (e.g., port 5002)
        base_url=os.getenv("PIPER_BASE_URL", "http://127.0.0.1:5002"),
        aiohttp_session=aiohttp.ClientSession(),
    )

    # llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))
    llm = OLLamaLLMService(
        model="gemma3n:e4b",
        base_url="http://localhost:11434/v1"
    )

    messages = [
        {
            "role": "system",
            "content": "You are a friendly AI assistant. Respond naturally and keep your answers conversational.",
        },
    ]

    context = LLMContext(messages)
    context_aggregator = LLMContextAggregatorPair(context)

    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            rtvi,  # RTVI processor
            stt,
            context_aggregator.user(),  # User responses
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            context_aggregator.assistant(),  # Assistant spoken responses
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[RTVIObserver(rtvi)],
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        # Kick off the conversation.
        messages.append(
            {"role": "system", "content": "Say hello and briefly introduce yourself."})
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point for the bot starter."""

    transport_params = {
        "webrtc": lambda: TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
            turn_analyzer=LocalSmartTurnAnalyzerV3(),
        ),
    }

    transport = await create_transport(runner_args, transport_params)

    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main
    import nltk
    import ssl

    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    main()
