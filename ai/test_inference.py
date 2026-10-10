import os

from inference import predict_ecg


image_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "medical_reports",
    "EMIAS-52EBC1DB_f342399e4ef745b6bfac082602f8944a.png"
)


print("Testing ECG inference...")
print("Image:", image_path)

result = predict_ecg(
    image_path
)

print("\nPrediction Result:")
print(result)