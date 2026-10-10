import os
import cv2
import torch

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from .ecg_model import create_ecg_model
from .preprocessing import preprocess_image


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


# ResNet18 final convolutional layer
target_layers = [
    model.layer4[-1]
]


def generate_gradcam(image_path: str, output_path: str):

    image_tensor = preprocess_image(image_path)
    image_tensor = image_tensor.to(device)

    # Read original image
    original_image = cv2.imread(image_path)

    if original_image is None:
        raise ValueError("Could not read ECG image.")

    original_image = cv2.cvtColor(
        original_image,
        cv2.COLOR_BGR2RGB
    )

    original_image = cv2.resize(
        original_image,
        (224, 224)
    )

    rgb_image = original_image.astype(
        "float32"
    ) / 255.0


    # Get model prediction
    with torch.no_grad():

        output = model(image_tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        predicted_class = torch.argmax(
            probabilities,
            dim=1
        ).item()


    # Grad-CAM
    target = [
        ClassifierOutputTarget(
            predicted_class
        )
    ]


    with GradCAM(
        model=model,
        target_layers=target_layers
    ) as cam:

        grayscale_cam = cam(
            input_tensor=image_tensor,
            targets=target
        )

        grayscale_cam = grayscale_cam[0]


    # Overlay heatmap
    visualization = show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True
    )


    # Save result
    cv2.imwrite(
        output_path,
        cv2.cvtColor(
            visualization,
            cv2.COLOR_RGB2BGR
        )
    )


    return {
        "prediction": CLASS_NAMES[predicted_class],
        "gradcam_path": output_path
    }