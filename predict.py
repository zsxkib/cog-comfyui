# An example of how to convert a given API workflow into its own Replicate model
# Replace predict.py with this file when building your own workflow

import os
import mimetypes
import json
import shutil
from PIL import Image, ExifTags
from typing import List
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
        # Update KSampler settings (node 3)
        sampler = workflow["3"]["inputs"]
        sampler["seed"] = kwargs.get("seed", sampler["seed"])
        sampler["steps"] = kwargs.get("steps", sampler["steps"])
        sampler["cfg"] = kwargs.get("cfg", sampler["cfg"])
        sampler["sampler_name"] = kwargs.get("sampler_name", sampler["sampler_name"])
        sampler["scheduler"] = kwargs.get("scheduler", sampler["scheduler"])
        sampler["denoise"] = kwargs.get("denoise", sampler["denoise"])
        # Note: model, positive, negative, and latent_image are connections to other nodes,
        # so we don't update them directly here

        # Update Checkpoint Loader (node 4)
        workflow["4"]["inputs"]["ckpt_name"] = kwargs.get(
            "checkpoint", workflow["4"]["inputs"]["ckpt_name"]
        )

        # Update Empty Latent Image (node 5)
        latent_image = workflow["5"]["inputs"]
        latent_image["width"] = kwargs.get("width", latent_image["width"])
        latent_image["height"] = kwargs.get("height", latent_image["height"])
        latent_image["batch_size"] = kwargs.get(
            "batch_size", latent_image["batch_size"]
        )

        # Update VAEDecode (node 8)
        # Note: samples and vae are connections to other nodes, so we don't update them directly

        # Update InstantID Model Loader (node 11)
        workflow["11"]["inputs"]["instantid_file"] = kwargs.get(
            "instantid_file", workflow["11"]["inputs"]["instantid_file"]
        )

        # Update Load Image (node 13)
        if "image" in kwargs:
            workflow["13"]["inputs"]["image"] = kwargs["image"]
        workflow["13"]["inputs"]["upload"] = "image"  # Ensure this is set to "image"

        # Update PreviewImage (node 15)
        # Note: images is a connection to another node, so we don't update it directly

        # Update ControlNet Loader (node 16)
        workflow["16"]["inputs"]["control_net_name"] = kwargs.get(
            "control_net_name", workflow["16"]["inputs"]["control_net_name"]
        )

        # Update InstantID Face Analysis (node 38)
        workflow["38"]["inputs"]["provider"] = kwargs.get(
            "face_analysis_provider", workflow["38"]["inputs"]["provider"]
        )

        # Update CLIP Text Encode (Prompt) (node 39)
        workflow["39"]["inputs"]["text"] = kwargs.get(
            "prompt", workflow["39"]["inputs"]["text"]
        )
        # Note: clip is a connection to another node, so we don't update it directly

        # Update CLIP Text Encode (Negative Prompt) (node 40)
        workflow["40"]["inputs"]["text"] = kwargs.get(
            "negative_prompt", workflow["40"]["inputs"]["text"]
        )
        # Note: clip is a connection to another node, so we don't update it directly

        # Update Apply InstantID (node 60)
        instantid = workflow["60"]["inputs"]
        instantid["weight"] = kwargs.get("instantid_weight", instantid["weight"])
        instantid["start_at"] = kwargs.get("instantid_start_at", instantid["start_at"])
        instantid["end_at"] = kwargs.get("instantid_end_at", instantid["end_at"])
        instantid["noise_offset"] = kwargs.get(
            "noise_offset", instantid.get("noise_offset", 0.0)
        )
        instantid["ip_adapter_scale"] = kwargs.get(
            "ip_adapter_scale", instantid.get("ip_adapter_scale", 0.8)
        )
        instantid["reference_image_scale"] = kwargs.get(
            "reference_image_scale", instantid.get("reference_image_scale", 1.0)
        )
        instantid["identity_scale"] = kwargs.get(
            "identity_scale", instantid.get("identity_scale", 1.0)
        )
        # Note: instantid, insightface, control_net, image, model, positive, and negative
        # are connections to other nodes, so we don't update them directly

        return workflow

    def predict(
        self,
        image: Path = Input(description="An input image", default=None),
        prompt: str = Input(default=""),
        negative_prompt: str = Input(
            description="Things you do not want to see in your image", default=""
        ),
        seed: int = seed_helper.predict_seed(),
        steps: int = Input(default=30, description="Number of sampling steps"),
        cfg: float = Input(default=4.5, description="Classifier-free guidance scale"),
        sampler_name: str = Input(default="ddpm", description="Name of the sampler"),
        scheduler: str = Input(default="karras", description="Name of the scheduler"),
        denoise: float = Input(default=1.0, description="Denoising strength"),
        # checkpoint: str = Input(
        #     default="RealVisXL_V3.0_Turbo.safetensors",
        #     description="Name of the checkpoint file",
        # ),
        width: int = Input(default=1016, description="Width of the output image"),
        height: int = Input(default=1016, description="Height of the output image"),
        batch_size: int = Input(default=1, description="Batch size for generation"),
        # instantid_file: str = Input(
        #     default="instantid-ip-adapter.bin", description="InstantID model file"
        # ),
        # control_net_name: str = Input(
        #     default="instantid-controlnet.safetensors",
        #     description="Name of the ControlNet model",
        # ),
        # face_analysis_provider: str = Input(
        #     default="CUDA", description="Provider for face analysis"
        # ),
        instantid_weight: float = Input(
            default=0.8, description="Weight of the InstantID effect"
        ),
        instantid_start_at: float = Input(
            default=0.0, description="Start point of InstantID effect"
        ),
        instantid_end_at: float = Input(
            default=1.0, description="End point of InstantID effect"
        ),
        noise_offset: float = Input(
            default=0.0, description="Noise offset for InstantID"
        ),
        ip_adapter_scale: float = Input(
            default=0.8, description="Scale for IP-Adapter in InstantID"
        ),
        reference_image_scale: float = Input(
            default=1.0, description="Scale for the reference image in InstantID"
        ),
        identity_scale: float = Input(
            default=1.0, description="Scale for identity preservation in InstantID"
        ),
        output_format: str = optimise_images.predict_output_format(),
        output_quality: int = optimise_images.predict_output_quality(),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        self.comfyUI.cleanup(ALL_DIRECTORIES)

        # Hardcoded parameters
        checkpoint = "RealVisXL_V3.0_Turbo.safetensors"
        instantid_file = "instantid-ip-adapter.bin"
        control_net_name = "instantid-controlnet.safetensors"
        face_analysis_provider = "CUDA"

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
            checkpoint=checkpoint,
            width=width,
            height=height,
            batch_size=batch_size,
            instantid_file=instantid_file,
            image=image_path,
            control_net_name=control_net_name,
            face_analysis_provider=face_analysis_provider,
            instantid_weight=instantid_weight,
            instantid_start_at=instantid_start_at,
            instantid_end_at=instantid_end_at,
            noise_offset=noise_offset,
            ip_adapter_scale=ip_adapter_scale,
            reference_image_scale=reference_image_scale,
            identity_scale=identity_scale,
        )

        wf = self.comfyUI.load_workflow(workflow)
        self.comfyUI.connect()
        self.comfyUI.run_workflow(wf)
        print("DEBUG: Entering final section of predict function")
        print(f"DEBUG: OUTPUT_DIR = {OUTPUT_DIR}")
        print(f"DEBUG: Contents of OUTPUT_DIR: {os.listdir(OUTPUT_DIR)}")
        print(
            f"DEBUG: Contents of COMFYUI_TEMP_OUTPUT_DIR: {os.listdir(COMFYUI_TEMP_OUTPUT_DIR)}"
        )

        # Move files from temp to output directory
        for file in self.comfyUI.get_files(COMFYUI_TEMP_OUTPUT_DIR):
            shutil.move(file, OUTPUT_DIR)

        output_files = self.comfyUI.get_files(OUTPUT_DIR)
        print(f"DEBUG: output_files = {output_files}")
        print(f"DEBUG: Type of output_files = {type(output_files)}")

        if not output_files:
            raise ValueError(
                "No output files were generated. Check the ComfyUI logs for errors."
            )

        print(f"DEBUG: output_format = {output_format}")
        print(f"DEBUG: output_quality = {output_quality}")

        optimised_files = optimise_images.optimise_image_files(
            output_format, output_quality, output_files
        )
        print(f"DEBUG: optimised_files = {optimised_files}")
        print(f"DEBUG: Type of optimised_files = {type(optimised_files)}")

        result = [Path(str(file)) for file in optimised_files]
        print(f"DEBUG: result = {result}")
        print(f"DEBUG: Type of result = {type(result)}")

        for item in result:
            print(f"DEBUG: Item in result = {item}")
            print(f"DEBUG: Type of item = {type(item)}")
            print(f"DEBUG: Item exists? {item.exists()}")
            print(f"DEBUG: Item is file? {item.is_file()}")

        print("DEBUG: About to return from predict function")
        return result
