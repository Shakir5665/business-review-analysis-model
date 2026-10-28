import sys
from src.preprocessing.preprocessor import ReviewPreprocessor

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

def run_tests():
    preprocessor = ReviewPreprocessor()

    test_phrases = [
        "meka  narakai godak hodha item ekak..😡.awl..hodhata asurum karala ewanawa.......api gewana ganata godak hodha product ekak kiyala mama hithanawa",
        "The delivery was late 😡 but the product is good 😊",
        "verrrrryyyyy late deliveryyyyy",
        "delivari was very late and product was narakai",
        "meka  narakai godak  awl hodha item ekak..😡.awl ..hodhata asurum karala ewanawa.......api gewana ganata godak hodha product ekak kiyala mama hithanawa",
        None
    ]

    print("\n" + "="*50)
    print("PREPROCESSOR OUTPUT VERIFICATION")
    print("="*50)

    for i, text in enumerate(test_phrases, 1):
        output = preprocessor.preprocess(text)
        print(f"\nTest Case {i}:")
        print(f"  Input : {repr(text)}")
        print(f"  Output: {repr(output)}")

    print("\n" + "="*50)

if __name__ == "__main__":
    run_tests()
