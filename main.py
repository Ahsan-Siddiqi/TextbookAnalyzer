import compare
import easyocr

min_score = 0.85    # This value needs to be adjusted

def get_text(img):
    reader = easyocr.Reader(['en'])   # Can add more languages
    result = reader.readtext(img)
    return result

def main():
    img1, img2 = compare.capture_image()
    score = compare.compare(img1, img2)

    if score is not None and score > min_score:
        for (bbox, text, prob) in get_text(img1):           # Print out all detected text
            print(f"Text: {text} | Probability: {prob}")
    else:
        print(f"Error: Score: {score} ")

if __name__ == "__main__":
    main()

