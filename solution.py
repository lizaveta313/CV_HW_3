from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import cv2
import numpy as np


PARAMS = {
    "logs-1.jpg": {
        "blur": 5, "roi": (335, 0, 1020, 795),
        "hough": [dict(dp=1.0, minDist=48, param1=140, param2=31,
                       minRadius=16, maxRadius=80)],
    },
    "logs-2.jpg": {
        "blur": 7, "roi": (230, 50, 850, 780),
        "hough": [dict(dp=1.0, minDist=44, param1=140, param2=27,
                       minRadius=14, maxRadius=75)],
    },
    "logs-3.jpg": {
        "blur": 5, "roi": (350, 900, 805, 1415),
        "hough": [dict(dp=1.0, minDist=30, param1=180, param2=24,
                       minRadius=10, maxRadius=50)],
    },
    "logs-4.jpg": {
        "blur": 7,
        "hough": [dict(dp=1.0, minDist=20, param1=500, param2=14,
                       minRadius=6, maxRadius=25)],
    },
    "pipes-4.jpg": {
        "blur": 3,
        "hough": [dict(dp=1.0, minDist=19, param1=140, param2=24,
                       minRadius=8, maxRadius=24)],
    },
    "pipes-5.jpg": {
        "blur": 3,
        "hough": [dict(dp=1.0, minDist=17, param1=140, param2=23,
                       minRadius=5, maxRadius=30),
                  dict(dp=1.0, minDist=120, param1=140, param2=36,
                       minRadius=65, maxRadius=110)],
    },
    "pipes-6.jpg": {
        "blur": 5,
        "hough": [dict(dp=1.0, minDist=35, param1=200, param2=34,
                       minRadius=12, maxRadius=45),
                  dict(dp=1.0, minDist=180, param1=180, param2=75,
                       minRadius=85, maxRadius=165)],
    },
}


def detect_circles(image, params):
    height, width = image.shape[:2]
    x1, y1, x2, y2 = params.get("roi", (0, 0, width, height))
    gray = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, params["blur"])
    found = []
    for hough in params["hough"]:
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, **hough)
        if circles is not None:
            found.append(circles[0])
    if not found:
        return np.empty((0, 3), dtype=np.int32)
    circles = np.round(np.vstack(found)).astype(np.int32)
    circles[:, 0] += x1
    circles[:, 1] += y1
    return circles


def draw_circles(image, circles):
    result = image.copy()
    thickness = max(1, round(min(image.shape[:2]) / 700))
    for x, y, radius in circles:
        cv2.circle(result, (int(x), int(y)), int(radius),
                   (0, 255, 0), thickness, cv2.LINE_AA)
        cv2.circle(result, (int(x), int(y)), thickness + 1,
                   (0, 0, 255), -1, cv2.LINE_AA)
    banner = max(36, round(image.shape[0] * 0.035))
    result = cv2.copyMakeBorder(result, banner, 0, 0, 0,
                                cv2.BORDER_CONSTANT, value=(0, 0, 0))
    cv2.putText(result, f"Count: {len(circles)}", (10, banner - 10),
                cv2.FONT_HERSHEY_SIMPLEX, banner / 48,
                (255, 255, 255), thickness, cv2.LINE_AA)
    return result


def main():
    archive = Path.cwd() / "hw3.zip"
    if not archive.is_file():
        archive = next(iter(sorted(Path.cwd().glob("*.zip"))), None)
    if archive is None:
        raise FileNotFoundError("В текущей папке не найден архив hw3.zip")
    output = Path("results")
    output.mkdir(exist_ok=True)
    counts = []
    settings = [f"OpenCV = {cv2.__version__}", f"NumPy = {np.__version__}"]
    with TemporaryDirectory(prefix="hw3_") as temporary:
        with ZipFile(archive) as zipped:
            zipped.extractall(temporary)
        images = sorted(path for path in Path(temporary).rglob("*")
                        if path.suffix.lower() == ".jpg")
        print(f"Архив: {archive.name}. Изображений: {len(images)}")
        for path in images:
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Не удалось прочитать {path.name}")
            params = PARAMS[path.name]
            circles = detect_circles(image, params)
            result = draw_circles(image, circles)
            target = output / f"{path.stem}_result.jpg"
            if not cv2.imwrite(str(target), result):
                raise OSError(f"Не удалось сохранить {target}")
            line = f"{path.name}: {len(circles)}"
            counts.append(line)
            print(line)
            settings.extend(["", path.name, f"medianBlur = {params['blur']}",
                             f"roi = {params.get('roi', 'full image')}"])
            for number, hough in enumerate(params["hough"], start=1):
                settings.append(f"pass = {number}")
                settings.extend(f"{key} = {value}" for key, value in hough.items())
    (output / "results.txt").write_text("\n".join(counts) + "\n", encoding="utf-8")
    (output / "parameters.txt").write_text("\n".join(settings) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
