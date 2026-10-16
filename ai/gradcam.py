
import cv2
import torch

from inference import get_model, device, CLASS_NAMES
from preprocessing import preprocess_image


def generate_gradcam(image_path: str, output_path: str):
    # Reuse the ECG prediction model instead of loading a duplicate.
    model = get_model()

    # Import Grad-CAM utilities only when this feature is requested.
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image

    image_tensor = preprocess_image(image_path)
    image_tensor = image_tensor.to(device)

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

    rgb_image = original_image.astype("float32") / 255.0

    # Get model prediction.
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    target = [ClassifierOutputTarget(predicted_class)]

    # ResNet18 final convolutional layer.
    target_layers = [model.layer4[-1]]

    with GradCAM(
        model=model,
        target_layers=target_layers
    ) as cam:
        grayscale_cam = cam(
            input_tensor=image_tensor,
            targets=target
        )[0]

    visualization = show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True
    )

    if not cv2.imwrite(
        output_path,
        cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
    ):
        raise ValueError("Could not save Grad-CAM image.")

    return {
        "prediction": CLASS_NAMES[predicted_class],
        "gradcam_path": output_path
    }
