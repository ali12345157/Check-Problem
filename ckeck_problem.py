from fastapi import FastAPI, UploadFile, File
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware  # Add this import

import torch
import io
import uvicorn

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_name = "google/vit-base-patch16-224"
model = ViTForImageClassification.from_pretrained(model_name)
processor = ViTImageProcessor.from_pretrained(model_name)
last_prediction = {"predicted_class": None, "issue_category": None}
CATEGORY_MAPPING = {
    "electricity": [
        "power", "cable", "wire", "plug", "socket", "electrical", "bulb", "lamp",
        "transformer", "breaker", "switch", "outlet", "voltage", "current", "battery",
        "fuse", "panel", "circuit", "wiring", "conduit", "relay", "generator", "inverter",
        "surge", "short circuit", "electrode", "grounding", "insulator", "power outage"
    ],
    "plumbing": [
        "pipe", "washbasin", "water", "sink", "tap", "leak", "drain", "toilet", "shower",
        "faucet", "sewage", "bathtub", "heater", "pump", "plumber", "clogged",
        "overflow", "gasket", "sealant", "hose", "nozzle", "septic", "hydrant",
        "backflow", "water pressure", "fixture", "pipe burst", "rust", "corrosion"
    ],
   "carpentry": [
    "wood", "door", "furniture", "table", "chair", "window", "cabinet", "shelf",
    "flooring", "wall panel", "bed", "drawer", "nail", "hammer", "carpenter",
    "hinge", "frame", "veneer", "laminate", "plank", "saw", "chisel",
    "wood glue", "sanding", "plywood", "molding", "varnish", "carving",
    "wooden door", "glass door", "metal door", "entrance", "exit"
]
,
    "air_conditioning": [
        "AC", "air conditioner", "vent", "cooling", "compressor", "refrigerant",
        "thermostat", "fan", "filter", "duct", "HVAC", "heater", "condenser", "coil",
        "evaporator", "insulation", "exhaust", "freon", "blower", "dehumidifier",
        "airflow", "temperature control", "heat pump", "split unit", "central air"
    ],
    "painting": [
        "paint", "brush", "roller", "wall", "ceiling", "color", "primer", "coating",
        "varnish", "stain", "plaster", "sandpaper", "graffiti", "artist", "decor",
        "spray", "lacquer", "pigment", "sealant", "epoxy", "putty", "enamel"
    ],
    "masonry": [
        "brick", "cement", "concrete", "tile", "stone", "wall", "pillar", "grout",
        "construction", "mortar", "pavement", "scaffold", "plaster", "floor",
        "slab", "rebar", "trowel", "stucco", "hardscape", "paving",
        "bricklaying", "foundation", "asphalt", "cinder block", "tiling", "grinding"
    ],
    "glass_work": [
        "glass", "mirror", "window", "frame", "shatter", "tempered", "laminated",
        "pane", "windshield", "stained glass", "glazier", "transparent",
        "fiberglass", "sealant", "silicone", "insulated glass", "double glazing",
        "crack", "frosted glass", "etched glass", "safety glass"
    ],
    "roofing": [
        "roof", "shingles", "tile", "gutter", "leak", "waterproof", "insulation",
        "chimney", "skylight", "drainage", "vent", "ceiling", "flashing", "membrane",
        "fascia", "downspout", "ridge", "eaves", "tar", "underlayment",
        "roofing nails", "roof deck", "soffit", "roof vent", "drip edge"
    ],
    "security_systems": [
        "camera", "CCTV", "sensor", "alarm", "lock", "keypad", "biometric", 
        "motion detector", "security", "surveillance", "monitoring", "gate",
        "intruder", "access control", "siren", "fire alarm", "smart lock",
        "video doorbell", "infrared", "fingerprint scanner", "keyless entry"
    ],
    "flooring": [
        "tile", "parquet", "vinyl", "hardwood", "laminate", "carpet", "rug",
        "floorboard", "grout", "marble", "stone", "epoxy", "linoleum", "subfloor",
        "underlayment", "adhesive", "sealant", "scratches", "polishing", "resurfacing"
    ],
    "welding": [
        "welding", "torch", "arc", "metal", "iron", "steel", "soldering", "fabrication",
        "gas welding", "fusion", "electric arc", "mig", "tig", "braze", "flux", "weld bead",
        "filler metal", "welding rod", "plasma cutter", "oxy-fuel", "slag", "penetration"
    ],
    "electronics": [
        "circuit board", "chip", "resistor", "capacitor", "diode", "transistor",
        "sensor", "fuse", "connector", "wireless", "PCB", "relay", "battery", "motherboard",
        "power supply", "oscillator", "microcontroller", "voltmeter", "amplifier",
        "IC", "semiconductor", "solder paste", "voltage regulator", "LED", "op-amp"
    ],
    "HVAC": [
        "heating", "ventilation", "air conditioning", "ductwork", "insulation",
        "boiler", "radiator", "air handler", "chiller", "thermostat", "exhaust fan",
        "heat exchanger", "pressure gauge", "humidity control", "ventilation fan"
    ],
    "solar_systems": [
        "solar panel", "inverter", "photovoltaic", "battery storage", "solar cell",
        "charge controller", "sunlight", "grid-tied", "off-grid", "net metering",
        "solar farm", "renewable energy", "solar tracker", "PV system"
    ],
    "appliances": [
        "refrigerator", "washing machine", "microwave", "oven", "dryer",
        "dishwasher", "water heater", "vacuum", "fan", "blender", "coffee maker",
        "stove", "freezer", "electric kettle", "toaster", "air fryer", "food processor"
    ]
}
class ImageURL(BaseModel):
    image_url: str

def classify_issue(predicted_label: str) -> str:
    predicted_label = predicted_label.lower()
    for category, keywords in CATEGORY_MAPPING.items():
        if any(keyword in predicted_label for keyword in keywords):
            return category
    return "other"

from fastapi import UploadFile, File

@app.post("/classify/")
async def classify_image(image: UploadFile = File(...)):
    try:
        # Ensure the uploaded file is an image
        if not image.content_type.startswith("image/"):
            return {"error": "Uploaded file is not an image. Please upload a valid image file."}

        image_bytes = await image.read()
        try:
            # Attempt to open the image
            image = Image.open(io.BytesIO(image_bytes))
            image.verify()  # Verify that the file is a valid image
        except Exception:
            return {"error": "The uploaded file is not a valid image or is corrupted."}

        # Preprocess the image and make predictions
        image = Image.open(io.BytesIO(image_bytes))  # Reopen the image after verification
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)

        predicted_class_idx = outputs.logits.argmax(-1).item()
        predicted_class = model.config.id2label[predicted_class_idx]

        issue_category = classify_issue(predicted_class)

        global last_prediction
        last_prediction = {"predicted_class": predicted_class, "issue_category": issue_category}

        return last_prediction

    except torch.cuda.CudaError as cuda_error:
        # Handle CUDA-specific errors
        return {"error": f"CUDA error occurred: {str(cuda_error)}. Ensure your GPU is properly configured."}

    except Exception as e:
        # Return a detailed error message for debugging
        return {"error": f"An error occurred while processing the image: {str(e)}"}

    
@app.get("/last_issue/")
async def get_last_issue():
    if last_prediction["predicted_class"] is None:
        return {"error": "No classification has been made yet"}
    return last_prediction

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
