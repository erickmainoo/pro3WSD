# run_on_test_data.py

from cs5322f25prog3 import (
    WSD_Test_director,
    WSD_Test_overtime,
    WSD_Test_rubbish,
)
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_NAME = "Erick_Mainoo"   # <--- CHANGE THIS TO EXACT FORMAT YOUR PROF WANTS

def run_for_word(word, func):
    input_file = os.path.join(BASE_DIR, f"{word}_test.txt")
    if not os.path.exists(input_file):
        print(f"Missing: {input_file}")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]

    results = func(sentences)

    output_file = os.path.join(BASE_DIR, f"result_{word}_{OUTPUT_NAME}.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(str(r) + "\n")

    print(f"Created: {output_file}")

def main():
    run_for_word("director", WSD_Test_director)
    run_for_word("overtime", WSD_Test_overtime)
    run_for_word("rubbish", WSD_Test_rubbish)

if __name__ == "__main__":
    main()
