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
        'axolotl close up photo', 'axolotl macro photography', 'axolotl aquarium closeup', 'axolotl face',
        'axolotl portrait', 'axolotl real photo', 'axolotl underwater photo', 'axolotl natural habitat',
        'axolotl hd photo', 'axolotl high resolution photo', 'axolotl in tank', 'axolotl pet',
        'axolotl in water', 'axolotl swimming', 'axolotl eating', 'axolotl mouth open',
        'axolotl gills closeup', 'axolotl albino', 'axolotl wild', 'axolotl black',
        'axolotl white', 'axolotl golden', 'axolotl blue', 'axolotl rare',
        'axolotl smiling', 'axolotl eyes', 'axolotl macro shot', 'axolotl underwater macro',
        'axolotl habitat', 'axolotl in aquarium', 'axolotl on gravel', 'axolotl with plants',
        'axolotl in nature', 'axolotl in river', 'axolotl close up head', 'axolotl close up body',
        'axolotl close up skin', 'axolotl close up tail', 'axolotl close up hand', 'axolotl close up foot',
        'axolotl close up eye', 'axolotl close up gills', 'axolotl close up mouth', 'axolotl close up nose',
        'axolotl close up teeth', 'axolotl close up smile', 'axolotl close up eating', 'axolotl close up swimming',
        'axolotl close up albino', 'axolotl close up wild', 'axolotl close up black', 'axolotl close up white',
        'axolotl close up golden', 'axolotl close up blue', 'axolotl close up rare', 'axolotl close up macro',
        'axolotl close up hd', 'axolotl close up high resolution', 'axolotl close up pet', 'axolotl close up tank',
        'axolotl close up aquarium', 'axolotl close up river', 'axolotl close up nature', 'axolotl close up plants',
        'axolotl close up gravel', 'axolotl close up water', 'axolotl close up habitat', 'axolotl close up portrait',
        'axolotl close up real photo', 'axolotl close up natural habitat', 'axolotl close up macro photography',
        'axolotl close up hd photo', 'axolotl close up high resolution photo', 'axolotl close up in tank',
        'axolotl close up in aquarium', 'axolotl close up in water', 'axolotl close up in river',
        'axolotl close up in nature', 'axolotl close up in plants', 'axolotl close up in gravel',
        'axolotl close up in habitat', 'axolotl close up in portrait', 'axolotl close up in real photo',
        'axolotl close up in natural habitat', 'axolotl close up in macro photography', 'axolotl close up in hd photo',
        'axolotl close up in high resolution photo', 'axolotl close up in pet', 'axolotl close up in wild',
        'axolotl close up in black', 'axolotl close up in white', 'axolotl close up in golden', 'axolotl close up in blue',
        'axolotl close up in rare', 'axolotl close up in macro', 'axolotl close up in hd', 'axolotl close up in high resolution'
    ]
    used_keywords = set()
    total_downloaded = len(os.listdir(OUT_DIR))
    existing_hashes = get_existing_hashes(OUT_DIR)
    while total_downloaded < 5000 and len(used_keywords) < len(keywords):
        for kw in keywords:
            if kw in used_keywords:
                continue
            print(f"Searching for: {kw}")
            # Download to a temp folder to check for duplicates before moving
            temp_dir = os.path.join(OUT_DIR, 'temp')
            os.makedirs(temp_dir, exist_ok=True)
            temp_crawler = BingImageCrawler(storage={'root_dir': temp_dir})
            temp_crawler.crawl(
                keyword=kw,
                filters={
                    'type': 'photo',
                    'size': 'large',
                    'color': 'color'
                },
                max_num=1000
            )
            # Only move non-duplicate, valid images to OUT_DIR
            for fname in os.listdir(temp_dir):
                fpath = os.path.join(temp_dir, fname)
                h = image_hash(fpath)
                if h and h not in existing_hashes and is_valid_image(fpath):
                    shutil.move(fpath, os.path.join(OUT_DIR, fname))
                    existing_hashes.add(h)
                else:
                    os.remove(fpath)
            os.rmdir(temp_dir)
            used_keywords.add(kw)
            total_downloaded = len(os.listdir(OUT_DIR))
            print(f"Total images so far: {total_downloaded}")
            if total_downloaded >= 5000:
                break
            time.sleep(2)
    print(f"Scraping complete. Images saved to {OUT_DIR}")
