
import cv2
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

FACE_MODEL_DIR = BASE_DIR / "face_models"

DETECTOR_MODEL = FACE_MODEL_DIR / "face_detection_yunet_2023mar.onnx"
RECOGNIZER_MODEL = FACE_MODEL_DIR / "face_recognition_sface_2021dec.onnx"


def create_face_models():
    detector = cv2.FaceDetectorYN.create(
        str(DETECTOR_MODEL),
        "",
        (320, 320),
        0.9,
        0.3,
        5000
    )

    recognizer = cv2.FaceRecognizerSF.create(
        str(RECOGNIZER_MODEL),
        ""
    )

    return detector, recognizer


def extract_face_features(image_path, detector, recognizer):
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    height, width = image.shape[:2]
    detector.setInputSize((width, height))

    _, faces = detector.detect(image)

    if faces is None or len(faces) == 0:
        raise ValueError("No face detected in the image.")

    if len(faces) > 1:
        raise ValueError("Multiple faces detected. Use an image with one face.")

    aligned_face = recognizer.alignCrop(image, faces[0])
    features = recognizer.feature(aligned_face)

    return features


def compare_faces(image1_path, image2_path):
    detector, recognizer = create_face_models()

    features1 = extract_face_features(
        image1_path, detector, recognizer
    )

    features2 = extract_face_features(
        image2_path, detector, recognizer
    )

    score = recognizer.match(
        features1,
        features2,
        cv2.FaceRecognizerSF_FR_COSINE
    )

    print(f"Cosine similarity: {score:.4f}")
    print("This is a similarity score, not a probability of identity.")

    # A provisional threshold for experimentation only.
    if score >= 0.363:
        print("Result: Possible match")
    else:
        print("Result: No match")

    return score


if __name__ == "__main__":
    print("EMIAS Face Recognition Test")
    print("Models loaded successfully.")

    detector, recognizer = create_face_models()
    print("YuNet detector and SFace recognizer initialized.")

    image1 = input("Enter path to first test image: ").strip().strip('"')
    image2 = input("Enter path to second test image: ").strip().strip('"')

    try:
        compare_faces(image1, image2)
    except (ValueError, cv2.error) as error:
        print(f"Face recognition test failed: {error}")
