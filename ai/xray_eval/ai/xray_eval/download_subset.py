
import csv
import time
import urllib.request
from pathlib import Path

csv_path = Path("ai/xray_eval/nih_sample.csv")
image_dir = Path("ai/xray_eval/images")
image_dir.mkdir(parents=True, exist_ok=True)

base_url = (
    "https://nih-chest-x-rays.s3.us-east-2.amazonaws.com/"
    "images_1024x1024/"
)

with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
    records = list(csv.DictReader(f))

downloaded = 0
skipped = 0
failed = []

print(f"Starting download for {len(records)} selected images...")

for i, row in enumerate(records, start=1):
    filename = row["Image Index"]
    destination = image_dir / filename

    if destination.exists() and destination.stat().st_size > 0:
        skipped += 1
        continue

    try:
        request = urllib.request.Request(
            base_url + filename,
            headers={"User-Agent": "EMIAS-research-evaluation/1.0"}
        )

        with urllib.request.urlopen(request, timeout=30) as response:
            content_type = response.headers.get("Content-Type", "")
            data = response.read()

        if not content_type.startswith("image/") or not data:
            raise ValueError(f"Unexpected response: {content_type}")

        destination.write_bytes(data)
        downloaded += 1

    except Exception as exc:
        failed.append((filename, str(exc)))

    if i % 10 == 0 or i == len(records):
        print(
            f"Processed {i}/{len(records)} | "
            f"Downloaded: {downloaded} | "
            f"Skipped: {skipped} | Failed: {len(failed)}"
        )

    time.sleep(0.1)

print("\nDownload summary")
print("New downloads:", downloaded)
print("Already present:", skipped)
print("Failed:", len(failed))

if failed:
    print("\nFirst failures:")
    for filename, error in failed[:10]:
        print(f"{filename}: {error}")
