import json
import uuid
import os
import io
import tempfile
from PIL import Image
from datetime import datetime
from urllib import request, parse
import websocket
import ssl


## SDXL inference
class Lora:
    def __init__(self, path, model_weight=0.75, clip_weight=1.00):
        self.path = path
        self.model_weight = model_weight
        self.clip_weight = clip_weight


class Generation:
    def __init__(
        self,
        prompt,
        width,
        height,
        use_refiner,
        sampler,
        scheduler,
        steps,
        cfg,
        prompt_negative=None,
        seed=None,
        loras=None,
        workflow_path="workflows/sdxl-lora.json",
    ):
        self.workflow_path = workflow_path
        self.prompt = prompt
        self.prompt_negative = prompt_negative
        self.width = width
        self.height = height
        self.use_refiner = use_refiner
        self.sampler = sampler
        self.scheduler = scheduler
        self.steps = steps
        self.cfg = cfg
        self.seed = seed
        self.loras = loras if loras is not None else []

        # Inference call
        self.output = self.inference()

    def inference(self):
        workflow = read_workflow_from_file(self.workflow_path)
        self.set_values(workflow)
        return call_api(workflow, output_dir="outputs/comfy/")

    def set_values(self, workflow):
        # Override the workflow's default values before calling the API
        workflow["35"]["inputs"]["positive"] = self.prompt
        if self.prompt_negative:
            workflow["35"]["inputs"]["negative"] = self.prompt_negative
        workflow["35"]["inputs"]["empty_latent_width"] = self.width
        workflow["35"]["inputs"]["empty_latent_height"] = self.height

        refiner_switch_value = 2 if self.use_refiner else 1
        workflow["37"]["inputs"]["Input"] = refiner_switch_value
        workflow["38"]["inputs"]["Input"] = refiner_switch_value

        workflow["17"]["inputs"]["sampler_name"] = self.sampler
        workflow["17"]["inputs"]["scheduler"] = self.scheduler
        workflow["17"]["inputs"]["steps"] = self.steps
        workflow["17"]["inputs"]["cfg"] = self.cfg
        if self.seed:
            workflow["17"]["inputs"]["seed"] = self.seed
        else:
            seed = int.from_bytes(os.urandom(8), "big")
            print(f"Using random seed {seed}")
            workflow["17"]["inputs"]["seed"] = seed

        switches = ["switch_1", "switch_2", "switch_3"]
        for switch in switches:
            workflow["25"]["inputs"][switch] = "Off"

        for i, lora in enumerate(self.loras, start=1):
            workflow["25"]["inputs"][f"switch_{i}"] = "On"
            workflow["25"]["inputs"][f"lora_name_{i}"] = lora.path
            workflow["25"]["inputs"][f"model_weight_{i}"] = lora.model_weight
            workflow["25"]["inputs"][f"clip_weight_{i}"] = lora.clip_weight


def upscale(
    input_image, params, mode="lsdr", workflow_path="workflows/upscale_ldsr.json"
):
    if mode != "lsdr":
        raise ValueError(
            f"Unsupported mode: {mode}. Only 'lsdr' is currently supported."
        )

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp:
        input_image.save(temp.name, format="PNG")
        image_path = temp.name

    def set_values(workflow: dict, params):
        # TODO: validate input parameters
        workflow["3"]["inputs"]["image"] = os.path.abspath(image_path)
        workflow["2"]["inputs"]["steps"] = params["steps"]
        workflow["2"]["inputs"]["pre_downscale"] = params["pre_downscale"]
        workflow["2"]["inputs"]["post_downscale"] = params["post_downscale"]
        workflow["2"]["inputs"]["downsample_method"] = params["downsample_method"]
        return workflow

    workflow = set_values(read_workflow_from_file(workflow_path), params)
    result = call_api(workflow, output_dir="outputs/comfy/upscale")
    os.remove(image_path)
    return result


## Server configuration and helper functions
server_address = "127.0.0.1:8188"
client_id = str(uuid.uuid4())


def send_request(path, method="GET", data=None):
    url = f"http://{server_address}{path}"
    if data is not None:
        data = json.dumps(data).encode("utf-8")
    req = request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method=method
    )
    with request.urlopen(req) as response:
        return json.loads(response.read())


def queue_prompt(prompt):
    return send_request(
        "/prompt", method="POST", data={"prompt": prompt, "client_id": client_id}
    )


def get_image_data(filename, subfolder, folder_type):
    params = parse.urlencode(
        {"filename": filename, "subfolder": subfolder, "type": folder_type}
    )
    with request.urlopen(f"http://{server_address}/view?{params}") as response:
        return response.read()


def get_history(prompt_id):
    return send_request(f"/history/{prompt_id}")


def create_image(image_data, prompt_id, output_images, output_dir=None):
    # Save to the output directory
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        file_path = os.path.join(output_dir, f"{timestamp}_{prompt_id}.png")
        with open(file_path, "wb") as file:
            file.write(image_data)

        print(f"Saved image to {file_path}")

    # Store in memory
    image = Image.open(io.BytesIO(image_data))
    output_images.append(image)


# WebSocket for real-time updates
def check_prompt_status(prompt_id):
    """Check the status of the prompt to determine if it's completed, cached, or still executing."""
    history = get_history(prompt_id)
    status = history.get(prompt_id, {}).get("status", {})
    if status.get("completed", False) or "execution_cached" in status:
        print(f"Prompt {prompt_id} execution already completed or cached.")
        return True  # Indicates no need to wait for WebSocket updates
    return False


def wait_for_completion(prompt_id):
    """Wait for the completion of the prompt execution using WebSocket for real-time updates."""
    if check_prompt_status(prompt_id):
        return  # Exit early if already completed or cached

    ws = websocket.WebSocket()
    try:
        ws.connect(
            f"ws://{server_address}/ws?clientId={client_id}",
            sslopt={"cert_reqs": ssl.CERT_NONE},
        )
        print(f"Connected to WebSocket for real-time updates on prompt ID: {prompt_id}")
        while True:
            out = ws.recv()
            if out:
                message = json.loads(out)
                if (
                    message["type"] == "executing"
                    and "data" in message
                    and message["data"].get("prompt_id") == prompt_id
                ):
                    if "execution_cached" in message or (
                        message["data"].get("node") is None
                    ):
                        print(f"Execution for prompt ID {prompt_id} has finished.")
                        break
    finally:
        ws.close()


# Process images
def process_images_from_prompt(prompt_id, output_dir=None):
    history = get_history(prompt_id)
    output_images = []
    if "outputs" in history.get(prompt_id, {}):
        for node_id, node_output in history[prompt_id]["outputs"].items():
            if "images" in node_output:
                image_data = get_image_data(
                    node_output["images"][0]["filename"],
                    node_output["images"][0]["subfolder"],
                    node_output["images"][0]["type"],
                )
                create_image(image_data, prompt_id, output_images, output_dir)
                return output_images
    else:
        print(f"No output found for prompt ID {prompt_id}")


# Main function
def call_api(prompt, output_dir=None):
    prompt_response = queue_prompt(prompt)
    prompt_id = prompt_response["prompt_id"]
    print(f"Prompt queued with ID: {prompt_id}")

    if check_prompt_status(prompt_id):
        print(
            f"Skipping processing for already completed or cached prompt ID: {prompt_id}"
        )
        return

    wait_for_completion(prompt_id)
    return process_images_from_prompt(prompt_id, output_dir)


def read_workflow_from_file(file_path):
    with open(file_path, "r") as file:
        return json.load(file)
