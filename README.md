# Program 3 – Word Sense Disambiguation (WSD)

**Course:** CS 5322 – Fall 2025  
**Group:** UG3  
**Members:** Erick Mainoo, Mollie Hamman, Alex Geer  

This project implements a small-scale Word Sense Disambiguation (WSD) system for three target words:

- `director`
- `overtime`
- `rubbish`

Each word has two senses defined in the provided data files. The goal is to decide, for each sentence containing the word, whether it uses Sense 1 or Sense 2.

The final system is a **hybrid model**:
- Supervised machine learning (TF–IDF + Logistic Regression)
- Rule-based overrides for high-confidence context clues (e.g., sports vs. workplace, film vs. business, literal trash vs. figurative “nonsense”)

---

## File Structure

```text
pro3WSD/
├── cs5322f25prog3.py       # Main module with WSD_Test_* functions (required by assignment)
├── train_wsd_models.py     # Script to train and save models
├── run_on_test_data.py     # Helper script to run on the official *_test.txt files
├── data/
│   ├── director.txt        # Training data for 'director'
│   ├── overtime.txt        # Training data for 'overtime'
│   └── rubbish.txt         # Training data for 'rubbish'
├── models/
│   ├── director_model.joblib
│   ├── director_vectorizer.joblib
│   ├── overtime_model.joblib
│   ├── overtime_vectorizer.joblib
│   ├── rubbish_model.joblib
│   └── rubbish_vectorizer.joblib
└── README.md
