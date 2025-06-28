## 
 Digit Classification – MLOps Lab 3

This project implements a basic digit classification model using scikit-learn's `load_digits` dataset.
It also demonstrates good software engineering practices like code modularization and Git workflows.

---

###  Project Structure

```
DigitClassification/
├── main.py       # Main script with only function calls
├── utils.py      # Contains all data processing and model functions
└── README.md     # This file
```

---

###  What’s Implemented

* Used logistic regression to classify handwritten digits.
* Refactored code into reusable functions.
* Moved functions to `utils.py`.
* Git workflow followed:

  * Created a new branch `branch1`
  * Committed all changes
  * Merged back to `main`
  * Deleted the branch after merge

---

### How to Run

Make sure you have Python and required libraries:

```bash
pip install scikit-learn numpy
```

Then run:

```bash
python main.py
```

---

### Sample Output

```
Accuracy: 0.97xxxx
```

---

### Notes

* All business logic is separated into `utils.py`
* `main.py` contains only function calls for clarity
* Project follows clean commit and branching strategy for MLOps lab submission
