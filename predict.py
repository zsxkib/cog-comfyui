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

api_json_file = "v2v.json"

# Force Hugging Face offline mode to avoid network calls at runtime
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


class Predictor(BasePredictor):
    def setup(self):
        # Start ComfyUI server
        self.comfyUI = ComfyUI("127.0.0.1:8188")
        self.comfyUI.start_server(OUTPUT_DIR, INPUT_DIR)

        # Load the workflow to determine needed weights
        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())

        # Handle all required weights for this model
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
        extension = os.path.splitext(input_file.name)[1]
        return f"{prefix}{extension}"

    def handle_input_file(
        self,
        input_file: Path,
        filename: str = "input.mp4",
    ):
        shutil.copy(input_file, os.path.join(INPUT_DIR, filename))

    def update_workflow(
        self,
        workflow,
        video_filename: str,
        prompt: str,
        width: int,
        height: int,
        keep_proportion: bool,
        steps: int,
        guidance_scale: float,
        denoise_strength: float,
        flow_shift: int,
        seed: int,
        frame_rate: int,
        crf: int,
        force_rate: int,
        force_size: str,
        custom_width: int,
        custom_height: int,
        frame_load_cap: int,
        skip_first_frames: int,
        select_every_nth: int,
    ):
        # Update input video loading node (43)
        workflow["43"]["inputs"]["video"] = video_filename
        workflow["43"]["inputs"]["force_rate"] = force_rate
        workflow["43"]["inputs"]["force_size"] = force_size
        workflow["43"]["inputs"]["custom_width"] = custom_width
        workflow["43"]["inputs"]["custom_height"] = custom_height
        workflow["43"]["inputs"]["frame_load_cap"] = frame_load_cap
        workflow["43"]["inputs"]["skip_first_frames"] = skip_first_frames
        workflow["43"]["inputs"]["select_every_nth"] = select_every_nth

        # Update image resize node (42)
        workflow["42"]["inputs"]["width"] = width
        workflow["42"]["inputs"]["height"] = height
        workflow["42"]["inputs"]["keep_proportion"] = keep_proportion

        # Update text encoding node (30) with the user's prompt
        workflow["30"]["inputs"]["prompt"] = prompt

        # Update sampler node (3)
        workflow["3"]["inputs"]["steps"] = steps
        workflow["3"]["inputs"]["embedded_guidance_scale"] = guidance_scale
        workflow["3"]["inputs"]["flow_shift"] = flow_shift
        workflow["3"]["inputs"]["seed"] = seed if seed is not None else 13
        workflow["3"]["inputs"]["denoise_strength"] = denoise_strength
        # force_offload stays 1 as per default JSON

        # Update final video combine node (53)
        workflow["53"]["inputs"]["frame_rate"] = frame_rate
        workflow["53"]["inputs"]["crf"] = crf
        workflow["53"]["inputs"]["save_output"] = True

    def predict(
        self,
        video: Path = Input(
            description="Input video file.",
        ),
        prompt: str = Input(
            description="Text prompt describing the desired output video style. Be descriptive.",
            default="high quality nature video of a excited brown bear walking through the grass, masterpiece, best quality",
        ),
        width: int = Input(
            description="Output video width (divisible by 16 for best performance).",
            default=768,
            ge=64,
            le=2048,
        ),
        height: int = Input(
            description="Output video height (divisible by 16 for best performance).",
            default=768,
            ge=64,
            le=2048,
        ),
        keep_proportion: bool = Input(
            description="Keep aspect ratio when resizing. If true, will adjust dimensions proportionally.",
            default=True,
        ),
        steps: int = Input(
            description="Number of sampling (denoising) steps.",
            default=30,
            ge=1,
            le=150,
        ),
        guidance_scale: float = Input(
            description="Embedded guidance scale. Higher values follow the prompt more strictly.",
            default=6.0,
            ge=1.0,
            le=20.0,
        ),
        denoise_strength: float = Input(
            description="Denoise strength (0.0 to 1.0). Higher = more deviation from input content.",
            default=0.85,
            ge=0.0,
            le=1.0,
        ),
        flow_shift: int = Input(
            description="Flow shift for temporal consistency. Adjust to tweak video smoothness.",
            default=9,
            ge=1,
            le=20,
        ),
        seed: int = seed_helper.predict_seed(),
        frame_rate: int = Input(
            description="Frame rate of the output video.",
            default=24,
            ge=1,
            le=120,
        ),
        crf: int = Input(
            description="CRF value for output video quality (0-51). Lower values = better quality.",
            default=19,
            ge=0,
            le=51,
        ),
        force_rate: int = Input(
            description="Force a new frame rate on the input video. 0 means no change.",
            default=0,
            ge=0,
            le=240,
        ),
        force_size: str = Input(
            description="Force resize method. 'Disabled' means original size. Otherwise applies custom_width/height.",
            default="Disabled",
        ),
        custom_width: int = Input(
            description="Custom width if force_size is not 'Disabled'.",
            default=512,
            ge=64,
            le=2048,
        ),
        custom_height: int = Input(
            description="Custom height if force_size is not 'Disabled'.",
            default=512,
            ge=64,
            le=2048,
        ),
        frame_load_cap: int = Input(
            description="Max frames to load from input video.",
            default=101,
            ge=1,
        ),
        skip_first_frames: int = Input(
            description="Number of initial frames to skip from the input video.",
            default=0,
            ge=0,
        ),
        select_every_nth: int = Input(
            description="Use every nth frame (1 = every frame, 2 = every second frame, etc.).",
            default=1,
            ge=1,
        ),
    ) -> Path:
        """Run a single prediction on the model:
        This function takes an input video and generates a new video with the desired style and constraints defined
        by the prompt and additional parameters. It uses a video-to-video stable diffusion-like model under the hood.
        """
        self.comfyUI.cleanup(ALL_DIRECTORIES)

        # Prepare the input video
        video_filename = self.filename_with_extension(video, "input")
        self.handle_input_file(video, video_filename)

        # Load the workflow
        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())

        # Update workflow inputs based on user parameters
        self.update_workflow(
            workflow=workflow,
            video_filename=video_filename,
            prompt=prompt,
            width=width,
            height=height,
            keep_proportion=keep_proportion,
            steps=steps,
            guidance_scale=guidance_scale,
            denoise_strength=denoise_strength,
            flow_shift=flow_shift,
            seed=seed,
            frame_rate=frame_rate,
            crf=crf,
            force_rate=force_rate,
            force_size=force_size,
            custom_width=custom_width,
            custom_height=custom_height,
            frame_load_cap=frame_load_cap,
            skip_first_frames=skip_first_frames,
            select_every_nth=select_every_nth,
        )

        # Run the workflow
        wf = self.comfyUI.load_workflow(workflow)
        self.comfyUI.connect()
        self.comfyUI.run_workflow(wf)

        # Retrieve and return the output video
        output_files = self.comfyUI.get_files(OUTPUT_DIR)
        if not output_files:
            raise RuntimeError("No output video was generated.")
        return output_files[0]
