# download minecraft mob sound effects and convert them into a single fixed-length
# numpy dataset for the analog wave-network classifier (hostile / neutral / passive).
# clips are written flat into minecraft_mobs_audio/ with an index prefix so that
# on-disk order matches the row order of minecraft_mobs.npz. downloaded
# audio is for local research/training use only -- ship this script and your trained
# model, not the audio. decoding .ogg requires libsndfile with Ogg Vorbis support
# (the common case via the soundfile wheels); otherwise `apt install libsndfile1`.

import io
import time
from math import gcd
from pathlib import Path

import numpy as np
import requests
import soundfile as sf
from scipy.signal import resample_poly

BASE_DIR = Path(__file__).parent
AUDIO_DIR = BASE_DIR / "minecraft_mobs_audio"
DATASET_PATH = BASE_DIR / "minecraft_mobs.npz"

# -------------------------------------- settings -------------------------------------
API = "https://minecraft.wiki/api.php"
HEADERS = {
    "User-Agent": "mob-sound-research-script/0.1 (contact: leon.herrmann@uni-weimar.de)"
}
ROOT_CATEGORY = "Category:Mob sounds"
REQUEST_DELAY = 0.3  # seconds between requests, be polite to the wiki

TARGET_SR = 16_000  # Hz, resample target
CLIP_SECONDS = 1.5  # pad/truncate every clip to this length
CLIP_LEN = int(TARGET_SR * CLIP_SECONDS)

CLASS_TO_INT = {"hostile": 0, "neutral": 1, "passive": 2}
CLIPS_PER_MOB = 1  # -1 for every clip the wiki lists

# folder name (derived from the wiki category) -> behavior class. one representative mob
# per class is active; uncomment the rest to grow the dataset (villager is passive in the
# game, so a genuinely neutral mob -- enderman -- represents the neutral class here)
MOB_CLASS = {
    # hostile
    "creeper": "hostile",
    # "blaze": "hostile",
    # "bogged": "hostile",
    # "breeze": "hostile",
    # "creaking": "hostile",
    # "drowned": "hostile",
    # "ender_dragon": "hostile",
    # "evoker": "hostile",
    # "ghast": "hostile",
    # "guardian": "hostile",
    # "hoglin": "hostile",
    # "husk": "hostile",
    # "illusioner": "hostile",
    # "magma_cube": "hostile",
    # "mega_spud": "hostile",
    # "parched": "hostile",
    # "phantom": "hostile",
    # "piglin_brute": "hostile",
    # "pillager": "hostile",
    # "plaguewhale_slab": "hostile",
    # "poisonous_potato_zombie": "hostile",
    # "ravager": "hostile",
    # "shulker": "hostile",
    # "silverfish": "hostile",
    # "skeleton": "hostile",
    # "slime": "hostile",
    # "stray": "hostile",
    # "toxifin_slab": "hostile",
    # "vex": "hostile",
    # "vindicator": "hostile",
    # "warden": "hostile",
    # "witch": "hostile",
    # "wither_skeleton": "hostile",
    # "wither": "hostile",
    # "zoglin": "hostile",
    # "zombie": "hostile",
    # "zombie_villager": "hostile",
    # neutral
    "enderman": "neutral",
    # "bee": "neutral",
    # "dolphin": "neutral",
    # "fox": "neutral",
    # "goat": "neutral",
    # "iron_golem": "neutral",
    # "llama": "neutral",
    # "nautilus": "neutral",
    # "panda": "neutral",
    # "piglin": "neutral",
    # "polar_bear": "neutral",
    # "pufferfish": "neutral",
    # "spider": "neutral",
    # "wolf": "neutral",
    # "zombie_nautilus": "neutral",
    # "zombified_piglin": "neutral",
    # passive
    "cow": "passive",
    # "allay": "passive",
    # "armadillo": "passive",
    # "axolotl": "passive",
    # "bat": "passive",
    # "camel": "passive",
    # "camel_husk": "passive",
    # "cat": "passive",
    # "chicken": "passive",
    # "copper_golem": "passive",
    # "donkey": "passive",
    # "fish": "passive",
    # "frog": "passive",
    # "glow_squid": "passive",
    # "horse": "passive",
    # "mooshroom": "passive",
    # "ocelot": "passive",
    # "parrot": "passive",
    # "pig": "passive",
    # "rabbit": "passive",
    # "sheep": "passive",
    # "skeleton_horse": "passive",
    # "sniffer": "passive",
    # "snow_golem": "passive",
    # "squid": "passive",
    # "strider": "passive",
    # "sulfur_cube": "passive",
    # "tadpole": "passive",
    # "turtle": "passive",
    # "villager": "passive",
    # "wandering_trader": "passive",
    # "zombie_horse": "passive",
    # "generic_mob" deliberately excluded: shared hurt/death sounds,
    # not tied to one mob's behavior class.
}


# --------------------------------------- helper --------------------------------------
def api_get(params):
    r = requests.get(API, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


def get_category_members(category, cmtype):
    # titles of subcategories or files directly in category
    members = []
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": category,
        "cmtype": cmtype,
        "cmlimit": "500",
        "format": "json",
    }
    cont = {}
    while True:
        data = api_get({**params, **cont})
        members += [m["title"] for m in data["query"]["categorymembers"]]
        if "continue" in data:
            cont = data["continue"]
            time.sleep(REQUEST_DELAY)
        else:
            break
    return members


def collect_files_recursive(category, seen=None):
    # all File: titles under category, recursing into nested subcats
    if seen is None:
        seen = set()
    if category in seen:
        return []
    seen.add(category)
    files = get_category_members(category, "file")
    time.sleep(REQUEST_DELAY)
    for sub in get_category_members(category, "subcat"):
        time.sleep(REQUEST_DELAY)
        files += collect_files_recursive(sub, seen)
    return files


def resolve_urls(file_titles):
    # map File: page titles to direct download URLs, batched by 50
    urls = {}
    for i in range(0, len(file_titles), 50):
        batch = file_titles[i : i + 50]
        data = api_get(
            {
                "action": "query",
                "titles": "|".join(batch),
                "prop": "imageinfo",
                "iiprop": "url",
                "format": "json",
            }
        )
        for page in data["query"]["pages"].values():
            if "imageinfo" in page:
                urls[page["title"]] = page["imageinfo"][0]["url"]
        time.sleep(REQUEST_DELAY)
    return urls


def load_and_fit(raw_bytes):
    # decode audio bytes -> mono float32 waveform at TARGET_SR, length CLIP_LEN
    data, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=False)
    if data.ndim > 1:  # stereo -> mono
        data = data.mean(axis=1)
    if sr != TARGET_SR:
        g = gcd(TARGET_SR, sr)
        data = resample_poly(data, TARGET_SR // g, sr // g)
    if len(data) >= CLIP_LEN:
        data = data[:CLIP_LEN]
    else:
        data = np.pad(data, (0, CLIP_LEN - len(data)))
    return data.astype("float32")


# ------------------------------------ create data ------------------------------------
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
mob_categories = get_category_members(ROOT_CATEGORY, "subcat")
print(f"found {len(mob_categories)} mob sound categories")

waveforms, class_labels, mob_names, file_names = [], [], [], []
index = 0  # running row index; also the on-disk filename prefix, so order matches

for cat in mob_categories:
    mob_name = cat.replace("Category:", "").replace(" sounds", "").strip()
    folder_key = mob_name.replace(" ", "_").lower()
    mob_class = MOB_CLASS.get(folder_key)
    if mob_class is None:
        continue

    file_titles = collect_files_recursive(cat)
    if not file_titles:
        continue
    urls = resolve_urls(file_titles)
    if CLIPS_PER_MOB != -1:
        urls = dict(sorted(urls.items())[:CLIPS_PER_MOB])

    n_ok = 0
    for title, url in urls.items():
        origname = title.replace("File:", "").replace(" ", "_")
        fname = f"{index:04d}_{folder_key}_{origname}"
        dest = AUDIO_DIR / fname
        if not dest.exists():
            r = requests.get(url, headers=HEADERS, timeout=60)
            r.raise_for_status()
            dest.write_bytes(r.content)
            time.sleep(REQUEST_DELAY)

        try:
            wf = load_and_fit(dest.read_bytes())
        except Exception as e:
            print(f"  could not decode {fname}: {e}")
            dest.unlink(missing_ok=True)
            continue

        waveforms.append(wf)
        class_labels.append(CLASS_TO_INT[mob_class])
        mob_names.append(folder_key)
        file_names.append(fname)
        index += 1
        n_ok += 1

    print(f"  {mob_name} ({mob_class}): {n_ok} usable clips")

# --------------------------------------- export --------------------------------------
X = np.stack(waveforms)  # (N, CLIP_LEN)
y = np.array(class_labels, dtype="int64")  # (N,) 0=hostile 1=neutral 2=passive
mob = np.array(mob_names)  # (N,) mob name per clip
files = np.array(file_names)  # (N,) flat audio filename per clip, order matches X

np.savez_compressed(
    DATASET_PATH,
    X=X,
    y=y,
    mob=mob,
    files=files,
    sr=TARGET_SR,
    classes=np.array(["hostile", "neutral", "passive"]),
)
print(f"saved dataset: X{X.shape}, y{y.shape} -> {DATASET_PATH}")
