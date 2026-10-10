import os
import torch

from ecg_model import create_ecg_model
from preprocessing import preprocess_image


CLASS_NAMES = {
    0: "Myocardial Infarction",
    1: "History of Myocardial Infarction",
    2: "Abnormal Heartbeat",
    3: "Normal"
}


MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "ecg_model.pth"
)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# Load the trained model once when this module starts
model = create_ecg_model(
    num_classes=4
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)
model.eval()


def predict_ecg(image_path: str):

    image_tensor = preprocess_image(
        image_path
    )

    image_tensor = image_tensor.to(device)

    with torch.no_grad():

        output = model(
            image_tensor
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )

        predicted_class = torch.argmax(
            probabilities,
            dim=1
        ).item()

        confidence = probabilities[
            0
        ][predicted_class].item()

    return {
        "prediction": CLASS_NAMES[predicted_class],
        "confidence": round(
            confidence * 100,
            2
        )
    }