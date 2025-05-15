import os
from icrawler.builtin import GoogleImageCrawler, BingImageCrawler
from PIL import Image
import re
import pytesseract
import hashlib
import time
import shutil

def has_watermark_text(file_path):
    try:
        img = Image.open(file_path)
        # Convert to grayscale for OCR
        gray = img.convert('L')
        # Run OCR
        text = pytesseract.image_to_string(gray)
        # If text is detected and not just whitespace, likely a watermark
        if text.strip():
            return True
        return False
    except Exception:
        return False

def is_valid_image(file_path):
    # Filter out images with watermarks, text, or non-axolotl content (basic heuristics)
    try:
        img = Image.open(file_path)
        # Only keep images that are reasonably large and RGB
        if img.mode != 'RGB' or min(img.size) < 128:
            return False
        # Watermark detection
        if has_watermark_text(file_path):
            return False
        # Optionally, add more advanced checks here
        return True
    except Exception:
        return False

def remove_invalid_images(folder):
    for fname in os.listdir(folder):
        fpath = os.path.join(folder, fname)
        if not is_valid_image(fpath):
            os.remove(fpath)

def image_hash(file_path):
    """Return a hash of the image file for duplicate detection."""
    try:
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return None

def get_existing_hashes(folder):
    """Get a set of hashes for all images in the folder."""
    hashes = set()
    for fname in os.listdir(folder):
        fpath = os.path.join(folder, fname)
        h = image_hash(fpath)
        if h:
            hashes.add(h)
    return hashes

if __name__ == '__main__':
    OUT_DIR = os.path.join(os.path.dirname(__file__), 'data', 'axolotl_scraped')
    os.makedirs(OUT_DIR, exist_ok=True)

    bing_crawler = BingImageCrawler(storage={'root_dir': OUT_DIR})

    # Large, diverse list of search queries
    keywords = [
    "real axolotl photograph", "authentic axolotl photo", "actual axolotl close up", "real life axolotl macro shot",
    "photograph of real axolotl", "axolotl in real aquarium", "natural axolotl image", "axolotl in true habitat",
    "unfiltered axolotl photo", "unedited axolotl picture", "genuine axolotl underwater", "realistic axolotl in tank",
    "true axolotl head closeup", "axolotl real skin photo", "photoreal axolotl in water", "axolotl real eyes",
    "real axolotl pet photography", "real environment axolotl", "actual photo of axolotl face",
    "real axolotl dorsal view", "HD real axolotl in aquarium", "axolotl real gills macro", "natural light real axolotl",
    "documentary axolotl photo", "captured real axolotl moment", "real axolotl movement underwater",
    "real photo of axolotl in glass tank", "realistic texture axolotl photo", "real closeup of axolotl mouth",
    "axolotl in real water tank", "true color axolotl image", "real photo of axolotl breathing", 
    "close photo of real axolotl smile", "real-life axolotl swimming photo", "real pet axolotl on gravel",
    "realistic axolotl with plants", "axolotl in natural stream photo", "real axolotl eating worm photo",
    "raw image of axolotl in tank", "unedited shot of real axolotl", "real close image of axolotl body",
    "macro real-life axolotl shot", "actual axolotl in clean tank", "photograph of real albino axolotl",
    "high-res real axolotl face", "realistic axolotl captured on camera", "real photo of wild axolotl",
    "true pet axolotl in aquarium", "real closeup of axolotl foot", "photo of axolotl in clear water"
]


    used_keywords = set()
    total_downloaded = len(os.listdir(OUT_DIR))
    existing_hashes = get_existing_hashes(OUT_DIR)
    while total_downloaded < 5000 and len(used_keywords) < len(keywords):
        for kw in keywords:
            if kw in used_keywords:
                continue
            print(f"Searching for: {kw}")
            # Download directly to OUT_DIR
            before_files = set(os.listdir(OUT_DIR))
            bing_crawler.crawl(
                keyword=kw,
                filters={
                    'type': 'photo',
                    'size': 'large',
                    'color': 'color'
                },
                max_num=100000000
            )
            # Check for new files and remove duplicates
            after_files = set(os.listdir(OUT_DIR))
            new_files = after_files - before_files
            for fname in new_files:
                fpath = os.path.join(OUT_DIR, fname)
                h = image_hash(fpath)
                # Only remove the file if it is a duplicate or invalid, never touch pre-existing files
                if not h or h in existing_hashes or not is_valid_image(fpath):
                    try:
                        os.remove(fpath)
                    except Exception:
                        pass
                else:
                    existing_hashes.add(h)
            used_keywords.add(kw)
            total_downloaded = len(os.listdir(OUT_DIR))
            print(f"Total images so far: {total_downloaded}")
            if total_downloaded >= 5000:
                break
            time.sleep(2)
    print(f"Scraping complete. Images saved to {OUT_DIR}")
