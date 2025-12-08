# run_on_test_data.py

from cs5322f25prog3 import (
    WSD_Test_director,
    WSD_Test_overtime,
    WSD_Test_rubbish,
)


def load_test_sentences(path: str):
    sentences = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.strip() == "":
                continue  # skip completely empty lines
            sentences.append(line)
    return sentences


def save_results(path: str, labels):
    with open(path, "w", encoding="utf-8") as f:
        for lab in labels:
            f.write(str(lab) + "\n")


def main():
    # Adjust filenames if Canvas uses exact same names.
    director_sents = load_test_sentences("director_test.txt")
    overtime_sents = load_test_sentences("overtime_test.txt")
    rubbish_sents = load_test_sentences("rubbish_test.txt")

    director_pred = WSD_Test_director(director_sents)
    overtime_pred = WSD_Test_overtime(overtime_sents)
    rubbish_pred = WSD_Test_rubbish(rubbish_sents)

    # Replace 'ErickMainoo' with whichever group member's name you decide to use
    save_results("result_director_ErickMainoo.txt", director_pred)
    save_results("result_overtime_ErickMainoo.txt", overtime_pred)
    save_results("result_rubbish_ErickMainoo.txt", rubbish_pred)


if __name__ == "__main__":
    main()
