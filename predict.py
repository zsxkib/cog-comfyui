# Simplified version with basic inputs only
import os
import mimetypes
import json
import shutil
from typing import List
from cog import BasePredictor, Input, Path
from comfyui import ComfyUI
import random

OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
COMFYUI_TEMP_OUTPUT_DIR = "ComfyUI/temp"
ALL_DIRECTORIES = [OUTPUT_DIR, INPUT_DIR, COMFYUI_TEMP_OUTPUT_DIR]

api_json_file = "workflow_api.json"

class Predictor(BasePredictor):
    def setup(self):
        self.comfyUI = ComfyUI("127.0.0.1:8188")
        self.comfyUI.start_server(OUTPUT_DIR, INPUT_DIR)

        # Add required weights from the workflow
        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())
        self.comfyUI.handle_weights(
            workflow,
            weights_to_download=[
                "mochi_preview_dit_fp8_e4m3fn.safetensors",
                "mochi_preview_vae_bf16.safetensors",
                "mochi_preview_clip-t5-xxl_encoderonly-fp8_e4m3fn.safetensors"
            ],
        )

    def update_workflow(self, workflow, **kwargs):
        # Update positive prompt (Node 1)
        positive_prompt = workflow["1"]["inputs"]
        positive_prompt["prompt"] = kwargs["prompt"]
        positive_prompt["strength"] = kwargs["prompt_strength"]
        positive_prompt["force_offload"] = True  # Hardcoded

        # Update negative prompt (Node 8)
        negative_prompt = workflow["8"]["inputs"]
        negative_prompt["prompt"] = kwargs["negative_prompt"]
        negative_prompt["strength"] = kwargs["prompt_strength"]
        negative_prompt["force_offload"] = True  # Hardcoded

        # Update sampler settings (Node 5)
        sampler = workflow["5"]["inputs"]
        sampler["width"] = kwargs["width"]
        sampler["height"] = kwargs["height"]
        sampler["num_frames"] = kwargs["num_frames"]
        sampler["steps"] = kwargs["steps"]
        sampler["cfg"] = kwargs["cfg"]
        sampler["seed"] = kwargs["seed"]

        # Update video settings (Node 9)
        video_settings = workflow["9"]["inputs"]
        video_settings["frame_rate"] = kwargs["fps"]
        # Keep other video settings as default from workflow

        # Add this: Update model settings (Node 4)
        model_settings = workflow["4"]["inputs"]
        model_settings["precision"] = "fp8_e4m3fn"  # Changed from fp8_e4m3fn_fast
        model_settings["attention_mode"] = "sage_attn"  # Keep this the same

    def predict(
        self,
        prompt: str = Input(
            description="Text prompt for the video generation",
            default="nature video of a red panda eating bamboo in front of a waterfall"
        ),
        negative_prompt: str = Input(
            description="Things you do not want to see in your video",
            default=""
        ),
        width: int = Input(
            description="Width of the output video",
            default=848,
            ge=64,
            le=2048
        ),
        height: int = Input(
            description="Height of the output video",
            default=480,
            ge=64,
            le=2048
        ),
        num_frames: int = Input(
            description="Number of frames to generate",
            default=163,
            ge=1,
            le=200
        ),
        steps: int = Input(
            description="Number of sampling steps",
            default=50,
            ge=1,
            le=100
        ),
        cfg: float = Input(
            description="Classifier free guidance scale",
            default=4.5,
            ge=1.0,
            le=20.0
        ),
        prompt_strength: float = Input(
            description="Prompt strength",
            default=1.0,
            ge=0.0,
            le=1.0
        ),
        fps: int = Input(
            description="Frames per second",
            default=24,
            ge=1,
            le=60
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize",
            default=None
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        if width * height > 2048 * 2048:
            raise ValueError("Resolution too high. Width * Height must be less than 2048 * 2048")

        self.comfyUI.cleanup(ALL_DIRECTORIES)

        # Generate seed if not provided
        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())

        self.update_workflow(
            workflow,
            prompt=prompt,
            negative_prompt=negative_prompt,
            prompt_strength=prompt_strength,
            width=width,
            height=height,
            num_frames=num_frames,
            steps=steps,
            cfg=cfg,
            seed=seed,
            fps=fps,
        )

        wf = self.comfyUI.load_workflow(workflow)
        self.comfyUI.connect()
        self.comfyUI.run_workflow(wf)

        # Get the output video file
        output_files = self.comfyUI.get_files(OUTPUT_DIR)
        if not output_files:
            raise RuntimeError("No output video generated")
        
        # Return the first (and should be only) video file
        return output_files[0]
