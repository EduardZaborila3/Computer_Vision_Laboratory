import argparse
import os
import random
import glob

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

random.seed(0)
np.random.seed(0)

DETECTORS = ["HARRIS", "SIFT", "SURF", "ORB", "BRISK", "AKAZE"]
RATIO = 0.75  # pragul Lowe pentru kNN

def _create(cls_name, **kwargs):
    """Daca detectorul nu exista, returneaza None."""
    for mod in (cv2, getattr(cv2, "xfeatures2d", None)):
        if mod is None:
            continue
        f = getattr(mod, f"{cls_name}_create", None)
        if f is None:
            cls = getattr(mod, cls_name, None)
            f = getattr(cls, "create", None) if cls is not None else None
        if f is not None:
            try:
                return f(**kwargs)
            except (cv2.error, TypeError, NotImplementedError):
                continue
    return None

class HarrisDetector:
    def __init__(self, level=0):
        self.thr = 0.01 / (5 ** level)
        self.desc = _create("SIFT")

    def detectAndCompute(self, gray, mask=None):
        r = cv2.cornerHarris(np.float32(gray), blockSize=2, ksize=3, k=0.04)
        r_dil = cv2.dilate(r, None)
        ys, xs = np.where((r == r_dil) & (r > self.thr * r.max()))
        kps = [cv2.KeyPoint(float(x), float(y), 8, response=float(r[y, x]))
               for x, y in zip(xs, ys)]
        if len(kps) == 0:
            return [], None
        return self.desc.compute(gray, kps)

def make_detector(name, level=0):
    if name == "HARRIS":
        return HarrisDetector(level)
    if name == "SIFT":
        return _create("SIFT", contrastThreshold=0.04 / (2 ** level))
    if name == "SURF":
        return _create("SURF", hessianThreshold=400 / (2 ** level))
    if name == "ORB":
        return _create("ORB", nfeatures=2000, fastThreshold=max(1, 20 - 5 * level))
    if name == "BRISK":
        return _create("BRISK", thresh=max(1, 30 - 7 * level))
    if name == "AKAZE":
        return _create("AKAZE", threshold=0.001 / (3 ** level))
    raise ValueError(name)

def norm_for(name):
    return cv2.NORM_HAMMING if name in ("ORB", "BRISK", "AKAZE") else cv2.NORM_L2


def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def compute(name, gray, max_level=4, verbose=False, max_kp=None):
    """Returneaza (keypoints, descriptors). Daca nu gaseste nimic, relaxeaza parametrii."""
    for level in range(max_level + 1):
        det = make_detector(name, level)
        if det is None:
            return None, None
        kps, des = det.detectAndCompute(gray, None)
        if kps is not None and len(kps) > 0 and des is not None:
            kps = list(kps)
            if max_kp is not None and len(kps) > max_kp:
                idx = np.argsort([-k.response for k in kps])[:max_kp]
                kps = [kps[i] for i in idx]
                des = des[idx]
            return kps, des
    if verbose:
        print(f"  [!] {name}: nu s-au gasit keypoints nici dupa modificarea parametrilor")
    return [], None

def match_bf(d1, d2, norm):
    """Brute force cu crossCheck. Returneaza lista de potriviri."""
    if d1 is None or d2 is None or len(d1) == 0 or len(d2) == 0:
        return []
    bf = cv2.BFMatcher(norm, crossCheck=True)
    return sorted(bf.match(d1, d2), key=lambda m: m.distance)


def match_knn(d1, d2, norm, ratio=RATIO):
    if d1 is None or d2 is None or len(d1) < 1 or len(d2) < 2:
        return []
    bf = cv2.BFMatcher(norm)
    pairs = bf.knnMatch(d1, d2, k=2)
    return [p[0] for p in pairs if len(p) == 2 and p[0].distance < ratio * p[1].distance]


MATCHERS = {"BF": match_bf, "kNN": match_knn}

# ex 1
def problem_1a(img, gray, out_dir):
    """Keypoints cu mai multe metode + plot."""
    results = {}
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, name in zip(axes.ravel(), DETECTORS):
        kps, des = compute(name, gray)
        if kps is None:
            ax.set_title(f"{name}: indisponibil")
            ax.axis("off")
            print(f"[!] {name} nu este disponibil in aceasta instalare OpenCV")
            continue
        results[name] = (kps, des)
        vis = cv2.drawKeypoints(img, kps, None, color=(0, 255, 0),
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{name}: {len(kps)} keypoints")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "1a_keypoints.png"), dpi=120)
    plt.show()
    return results

def problem_1b(gray, results, out_dir, grid=8):
    """Numar total + distributie pe 8x8 regiuni."""
    h, w = gray.shape
    print("\n=== 1b. Numar keypoints ===")
    for name, (kps, _) in results.items():
        print(f"{name:7s}: {len(kps)}")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, name in zip(axes.ravel(), results):
        kps = results[name][0]
        pts = np.array([k.pt for k in kps])
        # histograma 2D: randuri = y, coloane = x
        counts, _, _ = np.histogram2d(pts[:, 1], pts[:, 0], bins=grid,
                                      range=[[0, h], [0, w]])
        counts = counts.astype(int)
        print(f"\n{name} - keypoints pe fiecare regiune {grid}x{grid}:\n{counts}")
        print(f"  min={counts.min()}  max={counts.max()}  "
              f"regiuni goale={(counts == 0).sum()}  std={counts.std():.1f}")
        im = ax.imshow(counts, cmap="viridis")
        ax.set_title(name)
        for (i, j), v in np.ndenumerate(counts):
            ax.text(j, i, v, ha="center", va="center", color="w", fontsize=6)
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "1b_grid_counts.png"), dpi=120)
    plt.show()

def compare_with_transformed(gray, orig_res, transforms, title, out_dir):
    """Functie generica pentru 1c, 1d, 1e."""
    rows = []
    for tname, tf in transforms.items():
        g2 = tf(gray)
        for det in orig_res:
            kps2, des2 = compute(det, g2)
            kps1, des1 = orig_res[det]
            row = {"transformare": tname, "detector": det,
                   "kp_orig": len(kps1), "kp_transf": len(kps2)}
            for mname, mfun in MATCHERS.items():
                row[mname] = len(mfun(des1, des2, norm_for(det)))
            rows.append(row)
    df = pd.DataFrame(rows)
    print(f"\n=== {title} ===")
    print(df.to_string(index=False))
    df.to_csv(os.path.join(out_dir, f"{title.split('.')[0]}_matches.csv"), index=False)
    return df


def blur_transforms():
    t = {}
    for k in (3, 7, 15):
        t[f"mean {k}x{k}"] = lambda g, k=k: cv2.blur(g, (k, k))
    for k in (3, 7, 15):
        t[f"gauss {k}x{k}"] = lambda g, k=k: cv2.GaussianBlur(g, (k, k), 0)
    return t


def rotate(g, angle):
    h, w = g.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(g, M, (w, h))


def rotation_transforms():
    return {f"rot {a} deg": (lambda g, a=a: rotate(g, a)) for a in (10, 30, 45, 90, 180)}


def add_gauss_noise(g, std):
    noise = np.random.normal(0, std, g.shape)
    return np.clip(g.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def noise_transforms():
    return {f"noise std={s}": (lambda g, s=s: add_gauss_noise(g, s)) for s in (5, 15, 30, 60)}


def draw_example_matches(gray, orig_res, det, tf, out_dir, fname):
    g2 = tf(gray)
    kps1, des1 = orig_res[det]
    kps2, des2 = compute(det, g2)
    ms = match_knn(des1, des2, norm_for(det))
    vis = cv2.drawMatches(gray, kps1, g2, kps2, ms[:50], None, flags=2)
    cv2.imwrite(os.path.join(out_dir, fname), vis)

# ex 2
# Problema 2
def get_label(path):
    return os.path.basename(os.path.dirname(path))

def list_images(folder):
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    files = []
    for root, _, names in os.walk(folder):
        for n in names:
            if n.lower().endswith(exts):
                files.append(os.path.join(root, n))
    return sorted(files)

def load_features(files, det, max_side=800, max_kp=1000):
    feats = []
    for i, f in enumerate(files, 1):
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        h, w = img.shape
        if max(h, w) > max_side:
            scale = max_side / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        kps, des = compute(det, img, verbose=True, max_kp=max_kp)
        if kps is None:  # detector indisponibil
            return None
        feats.append((f, des))
        if i % 100 == 0 or i == len(files):
            print(f"  extrase features: {i}/{len(files)}", flush=True)
    return feats


def classify(train_feats, test_feats, det, matcher_name):
    norm = norm_for(det)
    train_ok = [(f, d) for f, d in train_feats if d is not None and len(d) > 0]
    train_labels = [get_label(f) for f, _ in train_ok]
    bf_cc = cv2.BFMatcher(norm, crossCheck=True)
    bf_knn = cv2.BFMatcher(norm)

    def count_matches(tdes, d):
        if matcher_name == "BF":
            return len(bf_cc.match(tdes, d))
        if len(tdes) < 1 or len(d) < 2:
            return 0
        pairs = bf_knn.knnMatch(tdes, d, k=2)
        return sum(1 for p in pairs if len(p) == 2 and p[0].distance < RATIO * p[1].distance)

    y_true, y_pred = [], []
    for idx, (tf_path, tdes) in enumerate(test_feats, 1):
        true = get_label(tf_path)
        if tdes is None or len(tdes) == 0:
            print(f"  [!] {os.path.basename(tf_path)} fara keypoints -> eticheta aleatoare")
            pred = random.choice(train_labels)
        else:
            counts = np.array([count_matches(tdes, d) for _, d in train_ok])
            best = np.where(counts == counts.max())[0]
            labels = [train_labels[i] for i in best]
            if len(best) == 1:
                pred = labels[0]
            else:
                votes = pd.Series(labels).value_counts()
                top = votes[votes == votes.max()].index.tolist()
                if len(top) == 1:
                    pred = top[0]
                else:
                    print(f"  [paritate] {os.path.basename(tf_path)}: {top}")
                    pred = random.choice(top)
        y_true.append(true)
        y_pred.append(pred)
        if idx % 20 == 0 or idx == len(test_feats):
            print(f"  clasificate: {idx}/{len(test_feats)}", flush=True)

    return y_true, y_pred


def problem_2(dataset_dir, out_dir):
    train_files = list_images(os.path.join(dataset_dir, "train"))
    test_files = list_images(os.path.join(dataset_dir, "test"))
    print(f"\nTrain: {len(train_files)} imagini, Test: {len(test_files)} imagini")
    summary = []
    for det in DETECTORS:
        print(f"\n##### Detector: {det} #####")
        train_feats = load_features(train_files, det)
        if train_feats is None:
            print(f"{det} indisponibil, trec mai departe.")
            continue
        test_feats = load_features(test_files, det)
        for mname in MATCHERS:
            print(f"--- {det} + {mname} ---")
            y_true, y_pred = classify(train_feats, test_feats, det, mname)
            acc = accuracy_score(y_true, y_pred)
            labels = sorted(set(y_true) | set(y_pred))
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            print(f"Acuratete: {acc:.3f}")
            print("Etichete:", labels)
            print("Matrice de confuzie:\n", cm)
            summary.append({"detector": det, "matcher": mname, "acc": acc})
    df = pd.DataFrame(summary)
    print("\n=== Rezumat acuratete ===")
    print(df.to_string(index=False))
    df.to_csv(os.path.join(out_dir, "2_accuracy.csv"), index=False)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    default_dataset = os.path.join(here, "IIT_DB_selectie")

    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None,
                    help="imaginea pentru problema 1 (implicit: prima imagine din train)")
    ap.add_argument("--dataset", default=default_dataset, help="folder cu train/ si test/")
    ap.add_argument("--out", default=os.path.join(here, "out"))
    ap.add_argument("--skip1", action="store_true", help="sare peste problema 1")
    ap.add_argument("--skip2", action="store_true", help="sare peste problema 2")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    train_dir = os.path.join(args.dataset, "train")
    test_dir = os.path.join(args.dataset, "test")
    for d in (train_dir, test_dir):
        if not os.path.isdir(d):
            raise SystemExit(f"Nu gasesc folderul: {d}")

    if not args.skip1:
        image_path = args.image or list_images(train_dir)[0]
        print("Problema 1 pe imaginea:", image_path)
        img = cv2.imread(image_path)
        gray = to_gray(img)

        res = problem_1a(img, gray, args.out)
        problem_1b(gray, res, args.out)
        compare_with_transformed(gray, res, blur_transforms(), "1c. Blur", args.out)
        compare_with_transformed(gray, res, rotation_transforms(), "1d. Rotatie", args.out)
        compare_with_transformed(gray, res, noise_transforms(), "1e. Zgomot gaussian", args.out)
        if "SIFT" in res:
            draw_example_matches(gray, res, "SIFT", lambda g: cv2.GaussianBlur(g, (7, 7), 0),
                                 args.out, "1c_example_matches.png")

    if not args.skip2:
        problem_2(args.dataset, args.out)


if __name__ == "__main__":
    main()