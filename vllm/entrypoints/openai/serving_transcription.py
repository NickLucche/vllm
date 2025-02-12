# SPDX-License-Identifier: Apache-2.0
import asyncio
import io
import time
from typing import AsyncGenerator, Final, Optional, Union, cast

from fastapi import Request

from vllm.config import ModelConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.protocol import (
    ChatCompletionResponseStreamChoice, ChatCompletionStreamResponse,
    DeltaMessage, ErrorResponse, RequestResponseMetadata, TranscriptionRequest,
    TranscriptionResponse, TranscriptionResponseVerbose, UsageInfo)
from vllm.entrypoints.openai.serving_engine import OpenAIServing
from vllm.entrypoints.openai.serving_models import OpenAIServingModels
from vllm.inputs.data import PromptType
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.utils import PlaceholderModule

try:
    import librosa
except ImportError:
    librosa = PlaceholderModule("librosa")  # type: ignore[assignment]

logger = init_logger(__name__)

# From https://platform.openai.com/docs/guides/speech-to-text/supported-languages#supported-languages
# TODO these configs should live somewhere with the model so we can support
# additional ones

ISO639_1_SUPPORTED_LANGS = {
    "af": "Afrikaans",
    "ar": "Arabic",
    "hy": "Armenian",
    "az": "Azerbaijani",
    "be": "Belarusian",
    "bs": "Bosnian",
    "bg": "Bulgarian",
    "ca": "Catalan",
    "zh": "Chinese",
    "hr": "Croatian",
    "cs": "Czech",
    "da": "Danish",
    "nl": "Dutch",
    "en": "English",
    "et": "Estonian",
    "fi": "Finnish",
    "fr": "French",
    "gl": "Galician",
    "de": "German",
    "el": "Greek",
    "he": "Hebrew",
    "hi": "Hindi",
    "hu": "Hungarian",
    "is": "Icelandic",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "kn": "Kannada",
    "kk": "Kazakh",
    "ko": "Korean",
    "lv": "Latvian",
    "lt": "Lithuanian",
    "mk": "Macedonian",
    "ms": "Malay",
    "mr": "Marathi",
    "mi": "Maori",
    "ne": "Nepali",
    "no": "Norwegian",
    "fa": "Persian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sr": "Serbian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "es": "Spanish",
    "sw": "Swahili",
    "sv": "Swedish",
    "tl": "Tagalog",
    "ta": "Tamil",
    "th": "Thai",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "cy": "Welsh"
}
ISO639_1_OTHER_LANGS = {
    "lo": "Lao",
    "jw": "Javanese",
    "tk": "Turkmen",
    "yi": "Yiddish",
    "so": "Somali",
    "bn": "Bengali",
    "nn": "Norwegian Nynorsk",
    "si": "Sinhala",
    "yo": "Yoruba",
    "sa": "Sanskrit",
    "mi": "Māori",
    "fo": "Faroese",  # codespell:ignore
    "mt": "Maltese",
    "tg": "Tajik",
    "mg": "Malagasy",
    "haw": "Hawaiian",
    "km": "Khmer",
    "br": "Breton",
    "ps": "Pashto",
    "ln": "Lingala",
    "la": "Latin",
    "ml": "Malayalam",
    "sq": "Albanian",
    "su": "Sundanese",
    "eu": "Basque",
    "ka": "Georgian",
    "uz": "Uzbek",
    "sn": "Shona",
    "ht": "Haitian",
    "as": "Assamese",
    "mn": "Mongolian",
    "te": "Telugu",
    "pa": "Panjabi",
    "tt": "Tatar",
    "gu": "Gujarati",
    "oc": "Occitan",
    "ha": "Hausa",
    "ba": "Bashkir",
    "my": "Burmese",
    "sd": "Sindhi",
    "am": "Amharic",
    "lb": "Luxembourgish",
    "bo": "Tibetan"
}

# As per https://platform.openai.com/docs/guides/speech-to-text#overview.
# TODO configurable
MAX_AUDIO_CLIP_FILESIZE_MB = 25
# TODO get from processor.feature_extractor.chunk_length
MAX_AUDIO_CLIP_DURATION_S = 30


class OpenAIServingTranscription(OpenAIServing):

    def __init__(
        self,
        engine_client: EngineClient,
        model_config: ModelConfig,
        models: OpenAIServingModels,
        *,
        request_logger: Optional[RequestLogger],
        return_tokens_as_token_ids: bool = False,
    ):
        super().__init__(engine_client=engine_client,
                         model_config=model_config,
                         models=models,
                         request_logger=request_logger,
                         return_tokens_as_token_ids=return_tokens_as_token_ids)

        diff_sampling_param = self.model_config.get_diff_sampling_param()
        if diff_sampling_param:
            logger.info(
                "Overwriting default completion sampling param with: %s",
                diff_sampling_param)

    async def _preprocess_transcription(
        self,
        request: TranscriptionRequest,
        audio_data: bytes,
    ) -> PromptType:
        # Validate request
        # TODO language should be optional and can be guessed.
        # For now we default to en. See
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/generation_whisper.py#L1520
        lang_token = f"<|{request.language}|>" if request.language else "<|en|>"
        if request.language:
            if request.language in ISO639_1_SUPPORTED_LANGS:
                pass
            elif request.language in ISO639_1_OTHER_LANGS:
                logger.warning(
                    "The selected language %s has limited accuracy with"
                    " reported WER>=0.5. Results may be less accurate "
                    "for this choice.", request.language)
            else:
                raise ValueError(
                    f"Unsupported language: {request.language}."
                    "Language should be one of:" +
                    f" {list(ISO639_1_SUPPORTED_LANGS.values())}" +
                    f"or {list(ISO639_1_OTHER_LANGS.values())}")

        if len(audio_data) / 1024**2 > MAX_AUDIO_CLIP_FILESIZE_MB:
            raise ValueError("Maximum file size exceeded.")

        with io.BytesIO(audio_data) as bytes_:
            y, sr = librosa.load(bytes_)
        if librosa.get_duration(y=y, sr=sr) > MAX_AUDIO_CLIP_DURATION_S:
            raise ValueError(
                f"Maximum clip duration ({MAX_AUDIO_CLIP_DURATION_S}s) "
                "exceeded.")

        prompt = {
            "encoder_prompt": {
                "prompt": "",
                "multi_modal_data": {
                    "audio": (y, sr),
                },
            },
            "decoder_prompt":
            f"<|startoftranscript|>{lang_token}<|transcribe|><|notimestamps|>{request.prompt}"
        }
        return cast(PromptType, prompt)

    # TODO (varun) : Make verbose response work !
    async def create_transcription(
        self, audio_data: bytes, request: TranscriptionRequest,
        raw_request: Request
    ) -> Union[TranscriptionResponse, TranscriptionResponseVerbose,
               ErrorResponse]:
        """Transcription API similar to OpenAI's API.

        See https://platform.openai.com/docs/api-reference/audio/createTranscription
        for the API specification. This API mimics the OpenAI transcription API.
        """
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        # If the engine is dead, raise the engine's DEAD_ERROR.
        # This is required for the streaming case, where we return a
        # success status before we actually start generating text :).
        if self.engine_client.errored:
            raise self.engine_client.dead_error

        if request.response_format not in ['text', 'json']:
            return self.create_error_response(
                "Currently only support response_format `text` or `json`")

        request_id = f"trsc-{self._base_request_id(raw_request)}"

        request_metadata = RequestResponseMetadata(request_id=request_id)
        if raw_request:
            raw_request.state.request_metadata = request_metadata

        try:
            (
                lora_request,
                prompt_adapter_request,
            ) = self._maybe_get_adapters(request)

            if lora_request:
                return self.create_error_response(
                    "Currently do not support LoRA for Transcription.")
            if prompt_adapter_request:
                return self.create_error_response(
                    "Currently do not support PromptAdapter for Transcription."
                )

            prompt = await self._preprocess_transcription(
                request=request,
                audio_data=audio_data,
            )

        except ValueError as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(str(e))

        result_generator: Optional[AsyncGenerator[RequestOutput, None]] = None
        try:
            # TODO(rob): subtract len of tokenized prompt.
            default_max_tokens = self.model_config.max_model_len
            default_params = self.model_config.get_diff_sampling_param()
            sampling_params = request.to_sampling_params(
                default_max_tokens, default_params)

            self._log_inputs(
                request_id,
                prompt['decoder_prompt'],  # type: ignore
                params=sampling_params,
                lora_request=None,
                prompt_adapter_request=None)

            result_generator = self.engine_client.generate(
                prompt,
                sampling_params,
                request_id,
            )
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

        # TODO(rob): figure out a way to pipe streaming in.
        print("GEN TYPE", type(self.engine_client))
        print("STREAM", request.stream)
        if request.stream:
            return self.transcription_stream_generator(request,
                                                       result_generator,
                                                       request_id,
                                                       request_metadata)

        # Non-streaming response.
        try:
            async for op in result_generator:
                result = op
            return TranscriptionResponse(text=result.outputs[0].text)
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

    async def transcription_stream_generator(
        self,
        request: TranscriptionRequest,
        result_generator: AsyncGenerator[RequestOutput, None],
        request_id: str,
        request_metadata: RequestResponseMetadata,
    ) -> AsyncGenerator[str, None]:
        created_time = int(time.time())
        model_name = request.model
        chunk_object_type: Final = "chat.completion.chunk"
        first_iteration = True

        previous_num_tokens = [0]
        finish_reason_sent = [False]
        num_prompt_tokens = 0
        num_cached_tokens = None

        # all_previous_token_ids: Optional[List[List[int]]]

        # stream_options = request.stream_options
        # if stream_options:
        #     include_usage = stream_options.include_usage
        #     include_continuous_usage = include_usage and \
        #                                stream_options.continuous_usage_stats
        # else:
        # include_usage, include_continuous_usage = False, False
        include_usage, include_continuous_usage = False, False

        try:
            async for res in result_generator:
                # print(res, "RES\n")
                if res.prompt_token_ids is not None:
                    # TODO hide prompt='<|startoftranscript|><|en|><|transcribe|><|notimestamps|>', prompt_token_ids=[50258, 50259, 50360, 50364]
                    num_prompt_tokens = len(res.prompt_token_ids)
                    # NOTE user can't pass encoder prompts directly -at least
                    # not to Whisper- as the audio log-mel spectogram is used.
                    # TODO Here we could return sr * len(audio) / frame_hop
                    # if res.encoder_prompt_token_ids is not None:
                    # num_prompt_tokens += len(res.encoder_prompt_token_ids)

                # We need to do it here, because if there are exceptions in
                # the result_generator, it needs to be sent as the FIRST
                # response (by the try...catch).
                if first_iteration:
                    num_cached_tokens = res.num_cached_tokens
                    # Fist delta message.
                    # TODO keep completion output or have a new type?
                    choice_data = ChatCompletionResponseStreamChoice(
                        index=0,
                        delta=DeltaMessage(content="", ),
                        logprobs=None,
                        finish_reason=None)
                    chunk = ChatCompletionStreamResponse(
                        id=request_id,
                        object=chunk_object_type,
                        created=created_time,
                        choices=[choice_data],
                        model=model_name)

                    # if continuous usage stats are requested, add it
                    if include_continuous_usage:
                        chunk.usage = UsageInfo(
                            prompt_tokens=num_prompt_tokens,
                            completion_tokens=0,
                            total_tokens=num_prompt_tokens)

                    data = chunk.model_dump_json(exclude_unset=True)
                    yield f"data: {data}\n\n"

                    first_iteration = False
                assert not type(res.outputs) == str
                for output in res.outputs:
                    # TODO I dont think you can have more than one here
                    i = output.index

                    if finish_reason_sent[i]:
                        continue
                    logprobs = None

                    # if not delta_text and not output.token_ids and \
                    #     not previous_num_tokens[i]:
                    #     # Chunked prefill case, don't return empty chunks
                    #     continue
                    delta_message = DeltaMessage(content=output.text)

                    # set the previous values for the next iteration
                    previous_num_tokens[i] += len(output.token_ids)

                    assert delta_message is not None

                    if output.finish_reason is None:
                        # Send token-by-token response for each request.n
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=delta_message,
                            logprobs=logprobs,
                            finish_reason=None)

                    # if the model is finished generating
                    else:
                        # check to make sure we haven't "forgotten" to stream
                        #   any tokens that were generated but previously
                        #   matched by partial json parsing

                        # Send the finish response for each request.n only once
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=delta_message,
                            logprobs=logprobs,
                            finish_reason=output.finish_reason,
                            stop_reason=output.stop_reason)

                        finish_reason_sent[i] = True

                    chunk = ChatCompletionStreamResponse(
                        id=request_id,
                        object=chunk_object_type,
                        created=created_time,
                        choices=[choice_data],
                        model=model_name)

                    # handle usage stats if requested & if continuous
                    # if include_continuous_usage:
                    #     completion_tokens = previous_num_tokens[i]
                    #     chunk.usage = UsageInfo(
                    #         prompt_tokens=num_prompt_tokens,
                    #         completion_tokens=completion_tokens,
                    #         total_tokens=num_prompt_tokens + completion_tokens,
                    #     )

                    data = chunk.model_dump_json(exclude_unset=True)
                    yield f"data: {data}\n\n"

            # once the final token is handled, if stream_options.include_usage
            # is sent, send the usage
            # if include_usage:
            #     completion_tokens = sum(previous_num_tokens)
            #     final_usage = UsageInfo(prompt_tokens=num_prompt_tokens,
            #                             completion_tokens=completion_tokens,
            #                             total_tokens=num_prompt_tokens +
            #                             completion_tokens)
            #     if self.enable_prompt_tokens_details and num_cached_tokens:
            #         final_usage.prompt_tokens_details = PromptTokenUsageInfo(
            #             cached_tokens=num_cached_tokens)

            #     final_usage_chunk = ChatCompletionStreamResponse(
            #         id=request_id,
            #         object=chunk_object_type,
            #         created=created_time,
            #         choices=[],
            #         model=model_name,
            #         usage=final_usage)
            #     final_usage_data = (final_usage_chunk.model_dump_json(
            #         exclude_unset=True, exclude_none=True))
            #     yield f"data: {final_usage_data}\n\n"

            # report to FastAPI middleware aggregate usage across all choices
            num_completion_tokens = sum(previous_num_tokens)
            request_metadata.final_usage_info = UsageInfo(
                prompt_tokens=num_prompt_tokens,
                completion_tokens=num_completion_tokens,
                total_tokens=num_prompt_tokens + num_completion_tokens)
            print("Usage", request_metadata.final_usage_info)

        except Exception as e:
            # TODO: Use a vllm-specific Validation Error
            logger.exception("Error in chat completion stream generator.")
            data = self.create_streaming_error_response(str(e))
            yield f"data: {data}\n\n"
        # Send the final done message after all response.n are finished
        yield "data: [DONE]\n\n"
