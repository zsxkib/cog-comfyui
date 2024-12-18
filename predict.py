import os
import mimetypes
import json
import shutil
from typing import List
from cog import BasePredictor, Input, Path
from comfyui import ComfyUI
from cog_model_helpers import seed as seed_helper

OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
COMFYUI_TEMP_OUTPUT_DIR = "ComfyUI/temp"
ALL_DIRECTORIES = [OUTPUT_DIR, INPUT_DIR, COMFYUI_TEMP_OUTPUT_DIR]

# Add video MIME types for recognition
mimetypes.add_type("video/mp4", ".mp4")
mimetypes.add_type("video/quicktime", ".mov")

# Point this to our new JSON file containing the inverse/resampler workflow
api_json_file = "rf-inv.json"

# Force offline mode: no huggingface / telemetry
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


class Predictor(BasePredictor):
    def setup(self):
        """
        Setup method to start ComfyUI and handle required model weights.
        """
        # Start ComfyUI server
        self.comfyUI = ComfyUI("127.0.0.1:8188")
        self.comfyUI.start_server(OUTPUT_DIR, INPUT_DIR)

        # Load the workflow to figure out what weights we need
        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())

        # Handle all required weights for this model (adjust as needed for your environment)
        #
        # We want to download "clip-vit-large-patch14" but not "clip-vit-large-patch14/model.safetensors"
        # because the latter isn't recognized by the builtin downloader. We'll rename it later.
        self.comfyUI.handle_weights(
            workflow,
            weights_to_download=[
                "hunyuan_video_vae_bf16.safetensors",
                "hunyuan_video_720_fp8_e4m3fn.safetensors",
                "clip-vit-large-patch14",
                "llava-llama-3-8b-text-encoder-tokenizer",
            ],
        )

    def filename_with_extension(self, input_file, prefix):
        """
        Return a filename with the same extension as the input file.
        """
        extension = os.path.splitext(input_file.name)[1]
        return f"{prefix}{extension}"

    def handle_input_file(self, input_file: Path, filename: str = "input.mp4"):
        """
        Copy the user-provided video file to the INPUT_DIR with a chosen filename.
        """
        shutil.copy(input_file, os.path.join(INPUT_DIR, filename))

    def update_workflow(
        self,
        workflow,
        video_filename: str,
        # Basic video-loading parameters
        force_rate: int,
        force_size: str,
        custom_width: int,
        custom_height: int,
        frame_load_cap: int,
        skip_first_frames: int,
        select_every_nth: int,
        # Basic image-resizing parameters
        width: int,
        height: int,
        keep_proportion: bool,
        # Text prompt
        prompt: str,
        # HyVideoInverseSampler (node 84)
        invert_steps: int,
        invert_embedded_guidance_scale: float,
        invert_flow_shift: int,
        invert_seed: int,
        invert_gamma: float,
        invert_start_step: int,
        invert_end_step: int,
        invert_gamma_trend: str,
        # HyVideoReSampler (node 86)
        re_steps: int,
        re_embedded_guidance_scale: float,
        re_flow_shift: int,
        re_start_step: int,
        re_end_step: int,
        re_eta_base: float,
        re_eta_trend: str,
        # Final video combine (node 69)
        frame_rate: int,
        crf: int,
    ):
        """
        Overwrite the relevant node inputs in rf-inv.json workflow.
        NOTE: Do not change node 43's clip_name here to clip-vit-large-patch14/model.safetensors
        That happens after we 'load_workflow' in the main predict() method.
        """

        # 1. Load video node (60)
        workflow["60"]["inputs"]["video"] = video_filename
        workflow["60"]["inputs"]["force_rate"] = force_rate
        workflow["60"]["inputs"]["force_size"] = force_size
        workflow["60"]["inputs"]["custom_width"] = custom_width
        workflow["60"]["inputs"]["custom_height"] = custom_height
        workflow["60"]["inputs"]["frame_load_cap"] = frame_load_cap
        workflow["60"]["inputs"]["skip_first_frames"] = skip_first_frames
        workflow["60"]["inputs"]["select_every_nth"] = select_every_nth

        # 2. Image resize node (62)
        workflow["62"]["inputs"]["width"] = width
        workflow["62"]["inputs"]["height"] = height
        workflow["62"]["inputs"]["keep_proportion"] = keep_proportion

        # 3. Text encode node (65) -> prompt for HyVideoTextEncode
        workflow["65"]["inputs"]["prompt"] = prompt

        # 4. HyVideoInverseSampler (84)
        workflow["84"]["inputs"]["steps"] = invert_steps
        workflow["84"]["inputs"][
            "embedded_guidance_scale"
        ] = invert_embedded_guidance_scale
        workflow["84"]["inputs"]["flow_shift"] = invert_flow_shift
        workflow["84"]["inputs"]["seed"] = invert_seed
        workflow["84"]["inputs"]["gamma"] = invert_gamma
        workflow["84"]["inputs"]["start_step"] = invert_start_step
        workflow["84"]["inputs"]["end_step"] = invert_end_step
        workflow["84"]["inputs"]["gamma_trend"] = invert_gamma_trend

        # 5. HyVideoReSampler (86)
        workflow["86"]["inputs"]["steps"] = re_steps
        workflow["86"]["inputs"]["embedded_guidance_scale"] = re_embedded_guidance_scale
        workflow["86"]["inputs"]["flow_shift"] = re_flow_shift
        workflow["86"]["inputs"]["start_step"] = re_start_step
        workflow["86"]["inputs"]["end_step"] = re_end_step
        workflow["86"]["inputs"]["eta_base"] = re_eta_base
        workflow["86"]["inputs"]["eta_trend"] = re_eta_trend

        # 6. Final combine node (69)
        workflow["69"]["inputs"]["frame_rate"] = frame_rate
        workflow["69"]["inputs"]["crf"] = crf
        # If you want to automatically save the final output, set save_output to True
        workflow["69"]["inputs"]["save_output"] = True

    def predict(
        self,
        # Original input video
        video: Path = Input(
            description="Input video file for editing, restyling, or transformation."
        ),
        # Basic prompt
        prompt: str = Input(
            default="cat wearing red bajazzeled sunglasses",
            description="Text prompt describing the style, subject, or transformation.",
        ),
        # Basic resizing / loading
        width: int = Input(
            default=512,
            ge=64,
            le=2048,
            description="Target output width (px) for processing.",
        ),
        height: int = Input(
            default=512,
            ge=64,
            le=2048,
            description="Target output height (px) for processing.",
        ),
        keep_proportion: bool = Input(
            default=True, description="Whether to maintain the original aspect ratio."
        ),
        force_rate: int = Input(
            default=0,
            ge=0,
            le=240,
            description="Force a specific frame rate. 0 means disabled.",
        ),
        force_size: str = Input(
            default="Disabled",
            description="Force a specific size (e.g. '480p', '720p'). Default is 'Disabled'.",
        ),
        custom_width: int = Input(
            default=512,
            ge=64,
            le=2048,
            description="When force_size is a custom size, this is the width to force.",
        ),
        custom_height: int = Input(
            default=512,
            ge=64,
            le=2048,
            description="When force_size is a custom size, this is the height to force.",
        ),
        frame_load_cap: int = Input(
            default=49,
            ge=1,
            description="Maximum number of video frames to load from the input.",
        ),
        skip_first_frames: int = Input(
            default=0,
            ge=0,
            description="Number of frames to skip at the start of the video.",
        ),
        select_every_nth: int = Input(
            default=1,
            ge=1,
            description="Only select every nth frame from the original video.",
        ),
        # HyVideoInverseSampler (node 84)
        invert_steps: int = Input(
            default=30, description="Number of diffusion steps for the inverse sampler."
        ),
        invert_embedded_guidance_scale: float = Input(
            default=0.0,
            description="Guidance scale used in the inverse sampler process.",
        ),
        invert_flow_shift: int = Input(
            default=1, description="Flow shift value for the inverse sampler."
        ),
        invert_seed: int = Input(
            default=1, description="RNG seed used by the inverse sampler."
        ),
        invert_gamma: float = Input(
            default=0.6,
            description="Gamma parameter controlling darkness/brightness transitions in inverse sampler.",
        ),
        invert_start_step: int = Input(
            default=0, description="Which step to begin inverse sampling at."
        ),
        invert_end_step: int = Input(
            default=27, description="Which step to end inverse sampling at."
        ),
        invert_gamma_trend: str = Input(
            default="constant",
            description="Behavior trend of gamma values (e.g. constant, linear).",
        ),
        # HyVideoReSampler (node 86)
        re_steps: int = Input(
            default=30, description="Number of diffusion steps for the re-sampler."
        ),
        re_embedded_guidance_scale: float = Input(
            default=6.0, description="Guidance scale used in the re-sampler process."
        ),
        re_flow_shift: int = Input(
            default=1, description="Flow shift value for the re-sampler."
        ),
        re_start_step: int = Input(
            default=0, description="Step at which to begin re-sampling."
        ),
        re_end_step: int = Input(
            default=13, description="Step at which to end re-sampling."
        ),
        re_eta_base: float = Input(
            default=0.7,
            description="Eta base value controlling the noise schedule in re-sampling.",
        ),
        re_eta_trend: str = Input(
            default="constant",
            description="Behavior trend of eta values (e.g. constant, linear).",
        ),
        # Output video combine
        frame_rate: int = Input(
            default=24, ge=1, le=120, description="Frame rate of the output video."
        ),
        crf: int = Input(
            default=19,
            ge=0,
            le=51,
            description="Constant rate factor for H.264 encoding (lower is higher quality).",
        ),
        # Optional: seed helper
        seed: int = seed_helper.predict_seed(),
    ) -> Path:
        """
        The main prediction function for an inverse/resample approach to video editing using rf-inv.json.
        It loads the user’s video, modifies the relevant parameters in the JSON workflow, runs ComfyUI,
        and returns the output video path.
        """
        # 1. Clean up any previous run
        self.comfyUI.cleanup(ALL_DIRECTORIES)

        # 2. Copy user’s input file
        video_filename = self.filename_with_extension(video, "input")
        self.handle_input_file(video, video_filename)

        # 3. Load the rf-inv.json workflow
        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())

        # 4. Temporarily keep node 43’s clip_name = “clip-vit-large-patch14”
        #    So that handle_weights() sees “clip-vit-large-patch14”.
        workflow["43"]["inputs"]["clip_name"] = "clip-vit-large-patch14"

        # 5. Update other nodes with user inputs
        self.update_workflow(
            workflow=workflow,
            video_filename=video_filename,
            force_rate=force_rate,
            force_size=force_size,
            custom_width=custom_width,
            custom_height=custom_height,
            frame_load_cap=frame_load_cap,
            skip_first_frames=skip_first_frames,
            select_every_nth=select_every_nth,
            width=width,
            height=height,
            keep_proportion=keep_proportion,
            prompt=prompt,
            invert_steps=invert_steps,
            invert_embedded_guidance_scale=invert_embedded_guidance_scale,
            invert_flow_shift=invert_flow_shift,
            invert_seed=invert_seed,
            invert_gamma=invert_gamma,
            invert_start_step=invert_start_step,
            invert_end_step=invert_end_step,
            invert_gamma_trend=invert_gamma_trend,
            re_steps=re_steps,
            re_embedded_guidance_scale=re_embedded_guidance_scale,
            re_flow_shift=re_flow_shift,
            re_start_step=re_start_step,
            re_end_step=re_end_step,
            re_eta_base=re_eta_base,
            re_eta_trend=re_eta_trend,
            frame_rate=frame_rate,
            crf=crf,
        )

        # 6. Load the workflow (handle_weights sees “clip-vit-large-patch14”)
        wf = self.comfyUI.load_workflow(workflow)

        # 7. After load_workflow but BEFORE run_workflow,
        #    switch to the real “clip-vit-large-patch14/model.safetensors”
        wf["43"]["inputs"]["clip_name"] = "clip-vit-large-patch14/model.safetensors"

        # 8. Connect to ComfyUI and run
        self.comfyUI.connect()
        self.comfyUI.run_workflow(wf)

        # 9. Retrieve the output video
        output_files = self.comfyUI.get_files(OUTPUT_DIR)
        if not output_files:
            raise RuntimeError("No output video was generated.")
        return output_files[0]
