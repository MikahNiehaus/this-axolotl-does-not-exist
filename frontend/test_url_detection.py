#!/usr/bin/env python
"""
Test script to verify GAN endpoint functionality with localhost detection
"""

import os
import sys
import requests
import base64
from PIL import Image
import io
import argparse
import time
import webbrowser

def test_endpoint(url="http://localhost:3000"):
    """Test the frontend's URL detection by opening it in a browser"""
    print(f"Opening frontend at {url}")
    webbrowser.open(url)
    print("Please check that:")
    print("1. The page loads correctly")
    print("2. It connects to the correct API (check browser console)")
    print("3. The image is generated and displayed")
    print("4. The model type is correctly shown in the title")

def main():
    parser = argparse.ArgumentParser(description="Test axolotl GAN endpoints")
    parser.add_argument("--url", default="http://localhost:3000", 
                        help="Frontend URL to test (default: http://localhost:3000)")
    args = parser.parse_args()
    
    # Start frontend test
    test_endpoint(args.url)
    
    print("\nTest complete!")
    print("You can verify that:")
    print("- When accessed via localhost, it uses the local API")
    print("- When accessed via a non-localhost URL, it uses the deployed API")

if __name__ == "__main__":
    main()
