#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers import T5EncoderModel, T5Tokenizer, SpeechT5HifiGan

from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import is_torch_xla_available, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import AudioPipelineOutput, DiffusionPipeline, StableDiffusionMixin


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)

# Add tango-master to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tango-master', 'tango2'))

try:
    from tango import Tango
except ImportError:
    print("Using fallback Tango implementation...")
    from tango import Tango


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import scipy
        >>> import torch
        >>> from diffusers import AudioLDM2Pipeline

        >>> repo_id = "cvssp/audioldm2"
        >>> pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
        >>> pipe = pipe.to("cuda")

        >>> # define the prompts
        >>> prompt = "The sound of a hammer hitting a wooden surface."
        >>> negative_prompt = "Low quality."

        >>> # set the seed for generator
        >>> generator = torch.Generator("cuda").manual_seed(0)

        >>> # run the generation
        >>> audio = pipe(
        ...     prompt,
        ...     negative_prompt=negative_prompt,
        ...     num_inference_steps=200,
        ...     audio_length_in_s=10.0,
        ...     num_waveforms_per_prompt=3,
        ...     generator=generator,
        ... ).audios

        >>> # save the best audio sample (index 0) as a .wav file
        >>> scipy.io.wavfile.write("techno.wav", rate=16000, data=audio[0])
        ```
        ```
        #Using AudioLDM2 for Text To Speech
        >>> import scipy
        >>> import torch
        >>> from diffusers import AudioLDM2Pipeline

        >>> repo_id = "anhnct/audioldm2_gigaspeech"
        >>> pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
        >>> pipe = pipe.to("cuda")

        >>> # define the prompts
        >>> prompt = "A female reporter is speaking"
        >>> transcript = "wish you have a good day"

        >>> # set the seed for generator
        >>> generator = torch.Generator("cuda").manual_seed(0)

        >>> # run the generation
        >>> audio = pipe(
        ...     prompt,
        ...     transcription=transcript,
        ...     num_inference_steps=200,
        ...     audio_length_in_s=10.0,
        ...     num_waveforms_per_prompt=2,
        ...     generator=generator,
        ...     max_new_tokens=512,          #Must set max_new_tokens equa to 512 for TTS
        ... ).audios

        >>> # save the best audio sample (index 0) as a .wav file
        >>> scipy.io.wavfile.write("tts.wav", rate=16000, data=audio[0])
        ```
"""


class TangoPipeline(DiffusionPipeline, StableDiffusionMixin):
    r"""
    Pipeline for text-to-audio generation using Tango.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).
    """

    model_cpu_offload_seq = "text_encoder->unet->vae->vocoder"

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: T5EncoderModel,
        tokenizer: T5Tokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        vocoder: SpeechT5HifiGan,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            vocoder=vocoder,
        )
        
        vae_config = getattr(self.vae, "config", None)
        block_out_channels = getattr(vae_config, "block_out_channels", None)
        self.vae_scale_factor = 2 ** (len(block_out_channels) - 1) if block_out_channels else 8

    def _encode_prompt(
        self,
        prompt,
        device,
        num_waveforms_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds= None,
        negative_prompt_embeds = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.
        
        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device (`torch.device`):
                torch device
            num_waveforms_per_prompt (`int`):
                number of waveforms that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the audio generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
        """
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            attention_mask = text_inputs.attention_mask

            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask.to(device),
            )
            prompt_embeds = prompt_embeds[0]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_waveforms_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_waveforms_per_prompt, seq_len, -1)

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            uncond_input_ids = uncond_input.input_ids.to(device)
            attention_mask = uncond_input.attention_mask.to(device)

            negative_prompt_embeds = self.text_encoder(
                uncond_input_ids,
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_waveforms_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(bs_embed * num_waveforms_per_prompt, seq_len, -1)
            
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

    def decode_latents(self, latents):
        vae_config = getattr(self.vae, "config", None)
        scaling_factor = getattr(vae_config, "scaling_factor", getattr(self.vae, "scale_factor", 1.0))
        latents = 1 / scaling_factor * latents
        decoded = self.vae.decode(latents)
        mel_spectrogram = getattr(decoded, "sample", decoded)
        return mel_spectrogram

    def check_inputs(
        self,
        prompt,
        audio_length_in_s,
        vocoder_upsample_factor,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        min_audio_length_in_s = vocoder_upsample_factor * self.vae_scale_factor
        if audio_length_in_s < min_audio_length_in_s:
            raise ValueError(
                f"`audio_length_in_s` has to be a positive value greater than or equal to {min_audio_length_in_s}, but "
                f"is {audio_length_in_s}."
            )

        vocoder_config = getattr(self.vocoder, "config", None)
        model_in_dim = getattr(vocoder_config, "model_in_dim", getattr(vocoder_config, "num_mels", 64))
        if model_in_dim % self.vae_scale_factor != 0:
            raise ValueError(
                f"The number of frequency bins in the vocoder's log-mel spectrogram has to be divisible by the "
                f"VAE scale factor, but got {model_in_dim} bins and a scale factor of "
                f"{self.vae_scale_factor}."
            )

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )
    def prepare_extra_step_kwargs(self, generator, eta):
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs
    
    def mel_spectrogram_to_waveform(self, mel_spectrogram):
        if mel_spectrogram.dim() == 4:
            mel_spectrogram = mel_spectrogram.squeeze(1)
        if not hasattr(getattr(self.vocoder, "config", None), "model_in_dim"):
            mel_spectrogram = mel_spectrogram.transpose(1, 2)

        waveform = self.vocoder(mel_spectrogram)
        waveform = waveform.cpu().float()
        return waveform

    def prepare_latents(self, batch_size, num_channels_latents, height, dtype, device, generator, latents=None):
        vocoder_config = getattr(self.vocoder, "config", None)
        model_in_dim = getattr(vocoder_config, "model_in_dim", getattr(vocoder_config, "num_mels", 64))
        if not hasattr(vocoder_config, "model_in_dim"):
            latent_shape = (batch_size, num_channels_latents, 256, 16)
        else:
            latent_shape = (
                batch_size,
                num_channels_latents,
                int(height) // self.vae_scale_factor,
                int(model_in_dim) // self.vae_scale_factor,
            )
        shape = (
            *latent_shape,
        ) 
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @torch.no_grad()
    #@replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        audio_length_in_s: Optional[float] = None,
        num_inference_steps: int = 10,
        guidance_scale: float = 2.5,
        diversity_guidance_scale: float = 0.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_waveforms_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: Optional[int] = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        #cond_fn: Optional[Callable[[torch.Tensor, int, torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        output_type: Optional[str] = "np",
        # **** SPARKE / RKE CUSTOM KWARGS ****
        rke_guided_sampler: Optional[Any] = None,
        criteria_guidance_scale: float = 0.0,
        guidance_freq: int = 1,
        rff_dim: int = 3000,
        criteria: str = 'vscore_clap',
        clap_for_guidance: Optional[Any] = None,
        F_M: Optional[torch.Tensor] = None,
        F_T: Optional[torch.Tensor] = None,
        F_M_real: Optional[torch.Tensor] = None,
        F_T_real: Optional[torch.Tensor] = None,
        beta: float = 0.1,
        logger_: Optional[Any] = None,
        regularize: bool = False,
        regularize_weight: float = 0.0,
        return_kernels: bool = False,
        # ************************************
    ):
        r"""
            The call function to the pipeline for generation.

            Args:
                diversity_guidance_scale (`float`, *optional*, defaults to 0.0):
                    Applies diversity guidance among parallel generations to enforce variance. Active only if 
                    `num_waveforms_per_prompt` > 1.
                cond_fn (`Callable`, *optional*):
                    A custom condition function to adjust `noise_pred` before the scheduler step. Takes 
                    `(latents, timestep, prompt_embeds, noise_pred)` as arguments. 
        """
        
        vocoder_config = getattr(self.vocoder, "config", None)
        upsample_rates = getattr(vocoder_config, "upsample_rates", [5, 4, 2, 2, 2])
        sampling_rate = getattr(vocoder_config, "sampling_rate", 16000)
        vocoder_upsample_factor = np.prod(upsample_rates) / sampling_rate

        if audio_length_in_s is None:
            audio_length_in_s = 10

        height = int(audio_length_in_s / vocoder_upsample_factor)

        original_waveform_length = int(audio_length_in_s * sampling_rate)
        if height % self.vae_scale_factor != 0:
            height = int(np.ceil(height / self.vae_scale_factor)) * self.vae_scale_factor
            logger.info(
                f"Audio length in seconds {audio_length_in_s} is increased to {height * vocoder_upsample_factor} "
                f"so that it can be handled by the model. It will be cut to {audio_length_in_s} after the "
                f"denoising process."
            )
        original_waveform_length = int(audio_length_in_s * sampling_rate)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self.text_encoder.device
        do_classifier_free_guidance = guidance_scale > 1.0

        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_waveforms_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_waveforms_per_prompt,
            num_channels_latents,
            height,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        sig_x=0.6
        sig_y=0.3
        
        omegas_x = torch.randn((32000, rff_dim), device=device)* (1.0 / sig_x)
        omegas_y = torch.randn((512, rff_dim), device=device)* (1.0 / sig_y)
        
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform standard CFG and apply diversity guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    
                    # Base directional guidance
                    guidance = noise_pred_text - noise_pred_uncond

                    # Apply diversity guidance penalty if requested
                    if diversity_guidance_scale > 0.0 and num_waveforms_per_prompt > 1:
                        # Subtract the mean guidance of the batch to push predictions away from each other
                        mean_guidance = guidance.mean(dim=0, keepdim=True)
                        guidance = guidance - (diversity_guidance_scale * mean_guidance)

                    noise_pred = noise_pred_uncond + guidance_scale * guidance

                # Allow custom gradient injection / condition function (like dynamic thresholding, watermark, etc.)
                # **** SPARKE / RKE GUIDANCE BLOCK ****
                # Only apply SPARKE during the first 90% of inference
                if rke_guided_sampler is not None and criteria_guidance_scale != 0 and clap_for_guidance :
                    # Isolate the positive prompt embeddings (ignoring negative embeddings)
                    txt_embd_for_guidance = prompt_embeds.chunk(2)[1] if do_classifier_free_guidance else prompt_embeds
                
                    if i % guidance_freq == 0 and criteria == 'vscore_clap':
                        grads, F_M, F_T = rke_guided_sampler.cond_fn(
                            latents=latents,
                            timestep=t,
                            index=i,
                            noise_pred=noise_pred,
                            extra_step_kwargs=extra_step_kwargs,
                            criteria_guidance_scale=criteria_guidance_scale,
                            prompt=prompt,
                            clip_for_guidance=clap_for_guidance,
                            prompt_embeds=None,
                            regularize=regularize,
                            regularize_weight=regularize_weight,
                            F_M=F_M,
                            F_T=F_T,
                            F_M_real=F_M_real,
                            F_T_real=F_T_real,
                            beta=beta,
                            omegas_x=omegas_x,
                            omegas_y=omegas_y,
                        )
                
                        if torch.isnan(grads).any() or torch.isinf(grads).any():
                            if logger_ is not None:
                                logger_.info("Skipping gradient update due to NaN or Inf in grads.")
                        else:
                            # Update latents with the computed diversity/alignment gradient
                            if(i%20==0):
                                F_M_prev = F_M
                                F_T_prev = F_T
                                grads = grads
                                latents = latents + grads
                
                            F_M= F_M_prev
                            F_T= F_T_prev
                # **** END SPARKE / RKE GUIDANCE BLOCK ****
                
                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                
                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)
                
                if XLA_AVAILABLE:
                    xm.mark_step()
                
                self.maybe_free_model_hooks()
                    
        # 8. Post-processing
        mel_spectrogram = self.decode_latents(latents)
        audio = self.mel_spectrogram_to_waveform(mel_spectrogram)
        audio = audio[:, :original_waveform_length]

        if output_type == "np":
            audio = audio.numpy()

        if not return_dict:
            return (audio,)

        return AudioPipelineOutput(audios=audio)