# run_on_test_data.py

import os
from cs5322f25prog3 import (
    WSD_Test_director,
    WSD_Test_overtime,
    WSD_Test_rubbish,
)

# Folder where this script lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_NAME = "Erick_Mainooo"  # first + last name


def _read_test_file(path: str):
    """
    Read a test file where each line is a sentence.
    Blank lines are ignored.
    """
    sentences = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                sentences.append(line)
    return sentences


def run_for_word(word: str, func):
    """
    Generic runner for one word:
    - Reads <word>_test.txt
    - Calls the corresponding WSD_Test_* function
    - Writes result_<word>_<name>.txt with one label per line
    """
    input_file = os.path.join(BASE_DIR, f"{word}_test.txt")
    if not os.path.exists(input_file):
        print(f"[!] Missing test file: {input_file}")
        return

    # 1) Read sentences (no labels in test files)
    sentences = _read_test_file(input_file)
    if not sentences:
        print(f"[!] No sentences found in {input_file}")
        return

    # 2) Run classifier
    preds = func(sentences)

    if len(preds) != len(sentences):
        print(f"[!] Warning: got {len(preds)} predictions for {len(sentences)} sentences in {word}")
    
    # 3) Write results
    output_file = os.path.join(BASE_DIR, f"result_{word}_{OUTPUT_NAME}.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        for p in preds:
            f.write(str(int(p)) + "\n")

    print(f"[✓] Wrote {len(preds)} predictions to {output_file}")


def main():
    run_for_word("director", WSD_Test_director)
    run_for_word("overtime", WSD_Test_overtime)
    run_for_word("rubbish", WSD_Test_rubbish)


if __name__ == "__main__":
    main()


