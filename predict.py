# An example of how to convert a given API workflow into its own Replicate model
# Replace predict.py with this file when building your own workflow

import os
import mimetypes
import json
import shutil
from PIL import Image, ExifTags
from typing import List, Iterator
from cog import BasePredictor, Input, Path
from comfyui import ComfyUI
from cog_model_helpers import optimise_images
from cog_model_helpers import seed as seed_helper

OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
COMFYUI_TEMP_OUTPUT_DIR = "ComfyUI/temp"
ALL_DIRECTORIES = [OUTPUT_DIR, INPUT_DIR, COMFYUI_TEMP_OUTPUT_DIR]

mimetypes.add_type("image/webp", ".webp")

# Save your example JSON to the same directory as predict.py
api_json_file = "workflow_api.json"


class Predictor(BasePredictor):
    def setup(self):
        self.comfyUI = ComfyUI("127.0.0.1:8188")
        self.comfyUI.start_server(OUTPUT_DIR, INPUT_DIR)

        # Give a list of weights filenames to download during setup
        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())
        with open("instantid-weights.txt", "r") as weights_file:
            weights_to_download = [
                line.strip() for line in weights_file if line.strip()
            ]
        self.comfyUI.handle_weights(
            workflow,
            weights_to_download=weights_to_download,
        )

    def handle_input_file(
        self,
        input_file: Path,
        filename: str = "image.png",
        check_orientation: bool = True,
    ) -> str:
        image = Image.open(input_file)

        if check_orientation:
            try:
                for orientation in ExifTags.TAGS.keys():
                    if ExifTags.TAGS[orientation] == "Orientation":
                        break
                exif = dict(image._getexif().items())

                if exif[orientation] == 3:
                    image = image.rotate(180, expand=True)
                elif exif[orientation] == 6:
                    image = image.rotate(270, expand=True)
                elif exif[orientation] == 8:
                    image = image.rotate(90, expand=True)
            except (KeyError, AttributeError):
                # EXIF data does not have orientation
                # Do not rotate
                pass

        output_path = os.path.join(INPUT_DIR, filename)
        image.save(output_path)
        return output_path

    def update_workflow(self, workflow, **kwargs):
        # KSampler (node 3)
        sampler = workflow["3"]["inputs"]
        sampler["seed"] = kwargs.get("seed", sampler["seed"])
        sampler["steps"] = kwargs.get("steps", sampler["steps"])
        sampler["cfg"] = kwargs.get("cfg", sampler["cfg"])
        sampler["sampler_name"] = kwargs.get("sampler_name", sampler["sampler_name"])
        sampler["scheduler"] = kwargs.get("scheduler", sampler["scheduler"])
        sampler["denoise"] = kwargs.get("denoise", sampler["denoise"])

        # Update Empty Latent Image (node 5)
        latent_image = workflow["5"]["inputs"]
        latent_image["width"] = kwargs.get("width", latent_image["width"])
        latent_image["height"] = kwargs.get("height", latent_image["height"])
        latent_image["batch_size"] = kwargs.get(
            "batch_size", latent_image["batch_size"]
        )

        # Load Image (node 13)
        if "image" in kwargs:
            workflow["13"]["inputs"]["image"] = kwargs["image"]
        workflow["13"]["inputs"]["upload"] = "image"

        # CLIP Text Encode (Prompt) (node 39)
        workflow["39"]["inputs"]["text"] = kwargs.get(
            "prompt", workflow["39"]["inputs"]["text"]
        )

        # CLIP Text Encode (Negative Prompt) (node 40)
        workflow["40"]["inputs"]["text"] = kwargs.get(
            "negative_prompt", workflow["40"]["inputs"]["text"]
        )

        # Apply InstantID (node 60)
        instantid = workflow["60"]["inputs"]
        instantid["weight"] = kwargs.get("instantid_weight", instantid["weight"])
        instantid["start_at"] = kwargs.get("instantid_start_at", instantid["start_at"])
        instantid["end_at"] = kwargs.get("instantid_end_at", instantid["end_at"])

        # IPAdapter Advanced (node 72)
        ipadapter = workflow["72"]["inputs"]
        ipadapter["weight"] = kwargs.get("ipadapter_weight", ipadapter["weight"])
        ipadapter["weight_type"] = kwargs.get(
            "ipadapter_weight_type", ipadapter["weight_type"]
        )
        ipadapter["combine_embeds"] = kwargs.get(
            "ipadapter_combine_embeds", ipadapter["combine_embeds"]
        )
        ipadapter["start_at"] = kwargs.get("ipadapter_start_at", ipadapter["start_at"])
        ipadapter["end_at"] = kwargs.get("ipadapter_end_at", ipadapter["end_at"])
        ipadapter["embeds_scaling"] = kwargs.get(
            "ipadapter_embeds_scaling", ipadapter["embeds_scaling"]
        )

        return workflow

    def predict(
        self,
        image: Path = Input(description="Input image for face reference", default=None),
        prompt: str = Input(
            default="Cyberpunk character, neon lights, futuristic implants, urban dystopia, high contrast, young man"
        ),
        negative_prompt: str = Input(
            default="NSFW, nudity, painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured"
        ),
        seed: int = seed_helper.predict_seed(),
        steps: int = Input(
            default=30, description="Number of sampling steps", ge=1, le=50
        ),
        cfg: float = Input(
            default=4.5, description="Classifier-free guidance scale", ge=0.01, le=10
        ),
        sampler_name: str = Input(
            default="ddpm",
            description="Name of the sampler",
            choices=[
                "euler",
                "euler_ancestral",
                "heun",
                "heunpp2",
                "dpm_2",
                "dpm_2_ancestral",
                "lms",
                "dpm_fast",
                "dpm_adaptive",
                "dpmpp_2s_ancestral",
                "dpmpp_sde",
                "dpmpp_sde_gpu",
                "dpmpp_2m",
                "dpmpp_2m_sde",
                "dpmpp_2m_sde_gpu",
                "dpmpp_3m_sde",
                "dpmpp_3m_sde_gpu",
                "ddpm",
                "lcm",
                "ddim",
                "uni_pc",
                "uni_pc_bh2",
            ],
        ),
        scheduler: str = Input(
            default="karras",
            description="Name of the scheduler",
            choices=[
                "normal",
                "karras",
                "exponential",
                "sgm_uniform",
                "simple",
                "ddim_uniform",
            ],
        ),
        denoise: float = Input(
            default=1.0,
            description="Denoising strength (recommended to keep at 1.0)",
            ge=0,
            le=1,
        ),
        width: int = Input(default=1600, description="Width of the output image"),
        height: int = Input(default=1600, description="Height of the output image"),
        batch_size: int = Input(
            default=1,
            description="Batch size for generation (higher values may cause OOM errors with large width/height)",
            ge=1,
            le=8,
        ),
        instantid_weight: float = Input(
            default=0.6, description="Weight of the InstantID effect", ge=0.01, le=2
        ),
        instantid_start_at: float = Input(
            default=0.0, description="Start point of InstantID effect", ge=0, le=1
        ),
        instantid_end_at: float = Input(
            default=1.0, description="End point of InstantID effect", ge=0, le=1
        ),
        ipadapter_weight: float = Input(
            default=0.7, description="Weight of the IPAdapter effect", ge=0.01, le=2
        ),
        ipadapter_weight_type: str = Input(
            default="linear",
            description="Weight type for IPAdapter",
            choices=[
                "linear",
                "ease in",
                "ease out",
                "ease in-out",
                "reverse in-out",
                "weak input",
                "weak output",
                "weak middle",
                "strong middle",
                "style transfer",
                "composition",
                "strong style transfer",
                "style and composition",
                "style transfer precise",
            ],
        ),
        ipadapter_combine_embeds: str = Input(
            default="average",
            description="Method to combine embeddings in IPAdapter",
            choices=["concat", "add", "subtract", "average", "norm average"],
        ),
        ipadapter_start_at: float = Input(
            default=0.0, description="Start point of IPAdapter effect", ge=0, le=1
        ),
        ipadapter_end_at: float = Input(
            default=1.0, description="End point of IPAdapter effect", ge=0, le=1
        ),
        ipadapter_embeds_scaling: str = Input(
            default="V only",
            description="Embeds scaling method for IPAdapter",
            choices=["V only", "K+V", "K+V w/ C penalty", "K+mean(V) w/ C penalty"],
        ),
        output_format: str = optimise_images.predict_output_format(),
        output_quality: int = optimise_images.predict_output_quality(),
    ) -> Iterator[Path]:
        """Run a single prediction on the model"""
        # Hardcoded parameters
        hardcoded_params = {
            "checkpoint": "RealVisXL_V3.0_Turbo.safetensors",
            "instantid_file": "instantid-ip-adapter.bin",
            "control_net_name": "instantid-controlnet.safetensors",
            "face_analysis_provider": "CUDA",
            "ipadapter_file": "ip-adapter-plus-face_sdxl_vit-h.safetensors",
            "clip_vision_file": "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors",
        }

        self.comfyUI.cleanup(ALL_DIRECTORIES)

        seed = seed_helper.generate(seed)

        if image:
            image_path = self.handle_input_file(image)
        else:
            raise ValueError("An input image is required for this workflow.")

        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())

        self.update_workflow(
            workflow,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            denoise=denoise,
            width=width,
            height=height,
            batch_size=batch_size,
            image=image_path,
            instantid_weight=instantid_weight,
            instantid_start_at=instantid_start_at,
            instantid_end_at=instantid_end_at,
            ipadapter_weight=ipadapter_weight,
            ipadapter_weight_type=ipadapter_weight_type,
            ipadapter_combine_embeds=ipadapter_combine_embeds,
            ipadapter_start_at=ipadapter_start_at,
            ipadapter_end_at=ipadapter_end_at,
            ipadapter_embeds_scaling=ipadapter_embeds_scaling,
            **hardcoded_params,  # Include hardcoded parameters
        )

        wf = self.comfyUI.load_workflow(workflow)
        self.comfyUI.connect()
        self.comfyUI.run_workflow(wf)

        # Move files from temp to output directory
        for file in self.comfyUI.get_files(COMFYUI_TEMP_OUTPUT_DIR):
            shutil.move(file, OUTPUT_DIR)

        output_files = self.comfyUI.get_files(OUTPUT_DIR)

        if not output_files:
            raise ValueError(
                "No output files were generated. Check the ComfyUI logs for errors."
            )

        optimised_files = optimise_images.optimise_image_files(
            output_format, output_quality, output_files
        )

        result = [Path(str(file)) for file in optimised_files]

        for item in result:
            yield item
