from pathlib import Path

import numpy as np
import torch
import torchxrayvision as xrv
from PIL import Image


_model = None


def get_model():
    global _model

    if _model is None:
        model = xrv.models.DenseNet(
            weights="densenet121-res224-all"
        )
        model.eval()
        _model = model

    return _model



def predict_chest_xray(image_path: str) -> dict:
    """
    Return experimental chest X-ray model scores.
    Scores are not calibrated diagnostic probabilities.
    """
    path = Path(image_path)

    if not path.is_file():
        raise FileNotFoundError("Chest X-ray image was not found.")

    try:
        with Image.open(path) as image:
            pixels = np.asarray(
                image.convert("L"),
                dtype=np.float32
            )

        # Normalize and add the single-channel dimension.
        pixels = xrv.datasets.normalize(pixels, 255)
        pixels = pixels[None, ...]

        # Use the preprocessing recommended by TorchXRayVision.
        pixels = xrv.datasets.XRayCenterCrop()(pixels)
        pixels = xrv.datasets.XRayResizer(224)(pixels)

    except Exception as exc:
        raise ValueError(
            "Could not preprocess the uploaded chest X-ray image."
        ) from exc

    tensor = torch.from_numpy(
        np.asarray(pixels, dtype=np.float32)
    ).unsqueeze(0)

    model = get_model()

    with torch.no_grad():
        scores = model(tensor)[0].cpu().tolist()

    outputs = {
        label: round(float(score), 4)
        for label, score in zip(model.pathologies, scores)
    }

    top_outputs = sorted(
        outputs.items(),
        key=lambda item: item[1],
        reverse=True
    )[:5]

    
    return {
        "prediction": "Experimental chest X-ray model output",
        "model": "TorchXRayVision DenseNet121",
        "all_outputs": outputs,
        "top_outputs": [
            {"label": label, "score": score}
            for label, score in top_outputs
        ],
        "disclaimer": (
            "Research prototype only. Scores are not calibrated "
            "diagnostic probabilities and do not establish a diagnosis."
        ),
    }
