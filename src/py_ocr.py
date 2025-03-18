# import the necessary packages
from PIL import Image
import pytesseract
import argparse
import cv2
import os

# Construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--folder", required=True,
    help="Path to the folder containing images")
ap.add_argument("-p", "--preprocess", type=str, default="thresh",
    help="Preprocessing method (thresh or blur)")
args = vars(ap.parse_args())

# Function to sort images by page number
def sort_key(file_name):
    # Extract the number from the filename
    page_number = int(file_name.split('_')[1].split('.')[0])
    return page_number

# List all files in the folder and sort them
image_files = [f for f in os.listdir(args["folder"]) if f.endswith(".png")]
image_files.sort(key=sort_key)

# Open the output text file
with open("pdf_env.txt", "w", encoding="utf-8") as output_file:
    # Loop over the image files
    for image_file in image_files:
        print("image_file", image_file)
        # Read the image path
        image_path = os.path.join(args["folder"], image_file)
        
        # Read the image and convert to grayscale
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply preprocessing
        if args["preprocess"] == "thresh":
            gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        elif args["preprocess"] == "blur":
            gray = cv2.medianBlur(gray, 3)
        
        # Save the preprocessed image temporarily
        temp_filename = "{}.png".format(os.getpid())
        cv2.imwrite(temp_filename, gray)
        
        # Perform OCR on the image
        # text = pytesseract.image_to_string(Image.open(temp_filename), lang='vie')
        text = pytesseract.image_to_string(Image.open(temp_filename), lang='eng')
        
        # Remove the temporary file
        os.remove(temp_filename)
        
        # Write the extracted text to the output file
        output_file.write(text)
        output_file.write("\n\n")

print("OCR complete. Text saved to pdf_env.txt")
