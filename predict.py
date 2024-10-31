# Simplified version with basic inputs only
import os
import json
import torch
from cog import BasePredictor, Input, Path
from comfyui import ComfyUI

OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
COMFYUI_TEMP_OUTPUT_DIR = "ComfyUI/temp"
ALL_DIRECTORIES = [OUTPUT_DIR, INPUT_DIR, COMFYUI_TEMP_OUTPUT_DIR]
# torch.set_default_dtype(torch.fp8_e4m3fn) # NOTE: There is no fp8_e4m3fn dtype in PyTorch see https://gist.github.com/malfet/7874d96b99670c3da83cbb779ab770c6

api_json_file = "workflow_api.json"


class Predictor(BasePredictor):
    def setup(self):
        """
        Available model files:

        CLIP Models (NOTE we don't need the `mochi/` prefix for CLIPs):
        - mochi_preview_clip-t5-xxl_encoderonly-bf16.safetensors
        - mochi_preview_clip-t5-xxl_encoderonly-fp8_e4m3fn.safetensors

        DIT Models:
        - mochi/mochi_preview_dit.safetensors (full precision)
        - mochi/mochi_preview_dit_GGUF_Q4_0_v1.safetensors (quantized)
        - mochi/mochi_preview_dit_GGUF_Q4_0_v2.safetensors (quantized)
        - mochi/mochi_preview_dit_GGUF_Q8_0.safetensors (quantized)
        - mochi/mochi_preview_dit_bf16.safetensors
        - mochi/mochi_preview_dit_fp8_e4m3fn.safetensors

        VAE Models:
        - mochi/mochi_preview_vae.safetensors (full precision)
        - mochi/mochi_preview_vae_bf16.safetensors

        NOTE: When selecting different model variants (bf16/fp8/etc), the precision setting
        in Node 4 needs to match. Currently using fp8_e4m3fn precision, but other options include:
        - fp8_e4m3fn
        - fp8_e4m3fn_fast
        - fp16
        - fp32
        - bf16
        """
        self.comfyUI = ComfyUI("127.0.0.1:8188")
        self.comfyUI.start_server(OUTPUT_DIR, INPUT_DIR)

        # Add required weights from the workflow
        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())

        self.dit_model = "mochi/mochi_preview_dit_fp8_e4m3fn.safetensors"
        self.vae_model = "mochi/mochi_preview_vae_bf16.safetensors"

        for weight in [
            "mochi_preview_clip-t5-xxl_encoderonly-fp8_e4m3fn.safetensors",
            self.dit_model,
            self.vae_model,
        ]:
            self.comfyUI.weights_downloader.download_weights(weight)

        self.model_precision = "fp8_e4m3fn"

    def update_workflow(self, workflow, **kwargs):
        # Update positive prompt (Node 1)
        positive_prompt = workflow["1"]["inputs"]
        positive_prompt["prompt"] = kwargs["prompt"]
        positive_prompt["strength"] = 1
        positive_prompt["force_offload"] = False

        # Update negative prompt (Node 8)
        negative_prompt = workflow["8"]["inputs"]
        negative_prompt["prompt"] = ""  # Hardcoded empty negative prompt
        negative_prompt["strength"] = 1
        negative_prompt["force_offload"] = False

        # Update sampler settings (Node 14)
        sampler = workflow["14"]["inputs"]
        sampler["width"] = 848
        sampler["height"] = 480
        sampler["num_frames"] = kwargs["num_frames"]
        sampler["steps"] = kwargs["steps"]
        sampler["cfg"] = 4.5  # Hardcoded CFG
        sampler["seed"] = kwargs["seed"]

        # Add VAE decode settings (Node 15)
        vae_decode = workflow["15"]["inputs"]
        vae_decode["enable_vae_tiling"] = kwargs["enable_vae_tiling"]

        # Update video settings (Node 9)
        video_settings = workflow["9"]["inputs"]
        video_settings["frame_rate"] = 30  # Hardcoded to match fal

        # Update model settings (Node 4)
        model_settings = workflow["4"]["inputs"]
        model_settings["precision"] = self.model_precision
        model_settings["attention_mode"] = "sdpa"

    def predict(
        self,
        prompt: str = Input(
            description="Text prompt for the video generation",
            default="nature video of a red panda eating bamboo in front of a waterfall",
        ),
        steps: int = Input(
            description="Number of sampling steps", default=30, ge=2, le=50
        ),
        num_frames: int = Input(
            description="Number of frames to generate (maximum 163)", default=163, ge=2, le=163
        ),
        enable_vae_tiling: bool = Input(
            description="Enable VAE tiling to reduce memory usage, may cause artifacts e.g. seams",
            default=False,
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize", default=None
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        self.comfyUI.cleanup(ALL_DIRECTORIES)

        # Generate seed if not provided
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())

        # NOTE: We need to do this because the Mochi ComfyUI nodes already look at `mochi/`
        # but when we download weights we need to include the full `mochi/` prefix
        # since they live in `ComfyUI/models/vae/mochi/` and `ComfyUI/models/diffusion_models/mochi/`
        workflow["4"]["inputs"]["model"] = self.dit_model.replace("mochi/", "")
        workflow["4"]["inputs"]["vae"] = self.vae_model.replace("mochi/", "")

        self.update_workflow(
            workflow,
            prompt=prompt,
            steps=steps,
            num_frames=num_frames,
            seed=seed,
            enable_vae_tiling=enable_vae_tiling,
        )

        wf = self.comfyUI.load_workflow(workflow)
        print("\n\n====================================")
        print(wf)
        print("====================================\n\n")
        self.comfyUI.connect()
        self.comfyUI.run_workflow(wf)

        # Get all MP4 files from the temp directory, sorted by creation time
        video_files = self.comfyUI.get_files(
            COMFYUI_TEMP_OUTPUT_DIR, file_extensions=["mp4"]
        )

        if not video_files:
            raise RuntimeError("No output video generated")

        # Get the most recently created file
        latest_video = max(video_files, key=lambda f: os.path.getctime(str(f)))
        return latest_video
