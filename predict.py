# Simplified version with basic inputs only
import os
import json
from cog import BasePredictor, Input, Path
from comfyui import ComfyUI

OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
COMFYUI_TEMP_OUTPUT_DIR = "ComfyUI/temp"
ALL_DIRECTORIES = [OUTPUT_DIR, INPUT_DIR, COMFYUI_TEMP_OUTPUT_DIR]

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
        
        self.dit_model = "mochi/mochi_preview_dit_bf16.safetensors"
        self.vae_model = "mochi/mochi_preview_vae_bf16.safetensors"

        for weight in [
            "mochi_preview_clip-t5-xxl_encoderonly-bf16.safetensors", # CLIP
            self.dit_model, # DIT 
            self.vae_model, # VAE
        ]:
            self.comfyUI.weights_downloader.download_weights(weight)
        
        self.model_precision = "bf16"

    def update_workflow(self, workflow, **kwargs):
        # Update positive prompt (Node 1)
        positive_prompt = workflow["1"]["inputs"]
        positive_prompt["prompt"] = kwargs["prompt"]
        positive_prompt["strength"] = 1
        positive_prompt["force_offload"] = False

        # Update negative prompt (Node 8)
        negative_prompt = workflow["8"]["inputs"]
        negative_prompt["prompt"] = kwargs["negative_prompt"]
        negative_prompt["strength"] = 1
        negative_prompt["force_offload"] = False

        # Update sampler settings (Node 14)
        sampler = workflow["14"]["inputs"]
        sampler["width"] = kwargs["width"]
        sampler["height"] = kwargs["height"]
        sampler["num_frames"] = kwargs["num_frames"]
        sampler["steps"] = kwargs["steps"]
        sampler["cfg"] = kwargs["cfg"]
        sampler["seed"] = kwargs["seed"]

        # Update video settings (Node 9)
        video_settings = workflow["9"]["inputs"]
        video_settings["frame_rate"] = kwargs["fps"]

        # Update model settings (Node 4)
        model_settings = workflow["4"]["inputs"]
        model_settings["precision"] = self.model_precision
        model_settings["attention_mode"] = "sdpa"

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
            default=49,
            ge=7,
            le=200
        ),
        steps: int = Input(
            description="Number of sampling steps",
            default=30,
            ge=2,
            le=100
        ),
        cfg: float = Input(
            description="Classifier free guidance scale",
            default=4.5,
            ge=1.0,
            le=20.0
        ),
        fps: int = Input(
            description="Frames per second",
            default=24,
            ge=1,
            le=60
        ),
        enable_vae_tiling: bool = Input(
            description="Enable VAE tiling to reduce memory usage, may cause artifacts e.g. seams",
            default=True
        ),
        num_tiles_w: int = Input(
            description="Number of tiles horizontally for VAE decoding. Only used when enable_vae_tiling is True",
            default=4,
            ge=1,
            le=8
        ),
        num_tiles_h: int = Input(
            description="Number of tiles vertically for VAE decoding. Only used when enable_vae_tiling is True",
            default=4,
            ge=1,
            le=8
        ),
        tile_overlap: int = Input(
            description="Overlap between tiles in pixels. Only used when enable_vae_tiling is True. Higher values reduce seams but use more memory",
            default=16,
            ge=0,
            le=64
        ),
        min_block_size: int = Input(
            description="Minimum block size for tiling. Only used when enable_vae_tiling is True",
            default=1,
            ge=1,
            le=32
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize",
            default=None
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
            
        workflow["4"]["inputs"]["model"] = self.dit_model.replace("mochi/", "")
        workflow["4"]["inputs"]["vae"] = self.vae_model.replace("mochi/", "")

        self.update_workflow(
            workflow,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            steps=steps,
            cfg=cfg,
            seed=seed,
            fps=fps,
            enable_vae_tiling=enable_vae_tiling,
            num_tiles_w=num_tiles_w,
            num_tiles_h=num_tiles_h,
            tile_overlap=tile_overlap,
            min_block_size=min_block_size,
        )
        
        wf = self.comfyUI.load_workflow(workflow)
        self.comfyUI.connect()
        self.comfyUI.run_workflow(wf)
        
        # Get all MP4 files from the temp directory, sorted by creation time
        video_files = self.comfyUI.get_files(
            COMFYUI_TEMP_OUTPUT_DIR, 
            file_extensions=["mp4"]
        )
        
        if not video_files:
            raise RuntimeError("No output video generated")
        
        # Get the most recently created file
        latest_video = max(video_files, key=lambda f: os.path.getctime(str(f)))
        return latest_video
