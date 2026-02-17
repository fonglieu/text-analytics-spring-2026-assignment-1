# AI Usage Log

## Task: Calculate Text and Word Length
**Date:** February 12, 2026  
**Time spent:** 5 minutes

**What I was trying to do:**
Calculate the character length and word count for each review.

**AI Prompt:**
"How do I calculate the length of text in characters and count the number of words in a df column?"

**AI Response:**
```python
data['text_length'] = data['ReviewBody'].str.len()
data['word_count'] = data['ReviewBody'].str.split().str.len()
```

**What I Modified:**
```python
data['text_length'] = data['ReviewBody'].astype(str).apply(len)
data['word_count'] = data['ReviewBody'].astype(str).apply(lambda x: len(x.split()))
```

**Why I Modified:**
- Added `.astype(str)` to handle potential non-string values and NaN values safely
- Changed from `.str.len()` to `.apply(len)`
- Used lambda function for word counting 

**What I Learned:**
- Always handle missing/null values when working with text data
- `.astype(str)` is safer than assuming all values are already strings
- Lambda functions can make text processing more efficient

**AI Errors Found:**
- Original code would fail on NaN values 

---

## Task: Text Cleaning Pipeline
**Date:** February 12, 2026  
**Time spent:** 25 minutes

**What I was trying to do:**
Create a comprehensive text cleaning function to remove HTML entities, HTML tags, URLs, expand contractions, remove special characters, and normalize whitespace.

**AI Prompt:**
"How do I clean text data for sentiment analysis? I need to remove HTML tags, URLs, expand contractions, and remove special characters in Python."

**AI Response:**
```python
import re
from bs4 import BeautifulSoup

def clean_text(text):
    # Remove HTML tags
    text = BeautifulSoup(text, 'html.parser').get_text()
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove special characters
    text = re.sub(r'[^A-Za-z\s]', '', text)
    # Lowercase
    text = text.lower()
    return text
```

**What I Modified:**
```python
import html
import re
import contractions

def clean_review(t):
    t = "" if pd.isna(t) else str(t)    # Handle missing values
    t = html.unescape(t)    # Unescape HTML entities
    t = re.sub(r"<.*?>", " ", t)    # Remove HTML tags
    t = re.sub(r"http\S+|www\S+", " ", t)    # Remove URLs
    t = contractions.fix(t)    # Expand contractions
    t = re.sub(r"[^A-Za-z\s']", " ", t)    # Remove numbers and special characters
    t = re.sub(r"\s+", " ", t).strip()    # Normalize whitespace
    if t == "":
        return "emptyreview"
    return t.lower()
```

**Why I Modified:**
- Used built-in `html.unescape()` instead of BeautifulSoup 
- Added explicit handling for missing/NaN values at the start
- Improved URL regex to catch both "http" and "www" prefixes
- Added `contractions.fix()` to expand contractions ("don't" → "do not")
- Kept apostrophes in the character filter for better word preservation
- Added edge case handling for empty strings after cleaning
- Added `.lower()` at the end for consistent casing

**What I Learned:**
- `html.unescape()` is faster and lighter than BeautifulSoup
- Regex order matters - clean from broad to specific
- Replacing with spaces prevents "word1word2" concatenation issues
- Always handle NaN/null values before applying string operations
- Edge cases like empty strings after cleaning need different handling

**AI Errors Found:**
- AI didn't include contraction expansion, which is important for sentiment analysis
- AI's URL regex missed URLs without "http://" prefix
- AI didn't handle NaN values

---

## Task: Check for Edge Cases
**Date:** February 14, 2026  
**Time spent:** 10 minutes

**What I was trying to do:**
Identify potential edge cases in the cleaned text data.

**AI Prompt:**
"How do I check for empty strings and very short text entries after text cleaning?"

**AI Response:**
```python
# Check for empty strings
empty_count = (data['text'] == '').sum()
print(f"Empty strings: {empty_count}")

# Check for short texts
short_count = (data['text'].str.len() < 10).sum()
print(f"Short texts (<10 chars): {short_count}")
```

**What I Modified:**
```python
# Check for edge cases
print(f"Empty strings after cleaning: {(data['text'] == 'emptyreview').sum()}")
print(f"Very short texts (<10 chars): {(data['text'].str.len() < 10).sum()}")
```

**Why I Modified:**
- Changed empty string check to look for 'emptyreview' placeholder since that's what my cleaning function returns
- Added more descriptive output messages

**What I Learned:**
- Edge case detection is critical before running models
- Empty reviews need special handling in sentiment analysis
- Very short texts (1-2 words) may not provide enough context for accurate sentiment

**AI Errors Found:**
- AI assumed empty strings would be '' but didn't account for custom placeholder values

---

## Task: Understanding Model Differences
**Date:** February 14, 2026  
**Time spent:** 15 minutes

**What I was trying to do:**
Understand the key differences between VADER, TextBlob, and Transformer models in terms of tokenization, preprocessing, features, speed, and context handling.

**AI Prompt:**
"What are the main differences between VADER, TextBlob, and transformer-based sentiment analysis models? Please compare their tokenization, features, speed, and how they handle context."

**AI Response:**
AI provided a detailed comparison explaining that:
- VADER uses lexicon-based approach with rule-based features
- TextBlob uses pattern-based tokenization with polarity/subjectivity scores
- Transformers use neural networks with contextual embeddings
- Speed varies significantly: VADER fastest, Transformers slowest
- Context handling: VADER rule-based, TextBlob pattern-based, Transformers deep contextual

**What I Modified:**
I organized this information into a structured markdown table for clarity:

| Aspect | VADER | TextBlob | Transformer |
|--------|-------|----------|-------------|
| Tokenization | Lexicon-based (internal) | Word-based (internal) | WordPiece (subword) |
| Preprocessing | Ideally keep CAPS/!!! | Lowercase OK | Minimal |
| Features | 4 scores (neg/neu/pos/compound) | 2 scores (polarity/subjectivity) | 6 emotions |
| Speed | Very fast | Fast | Slower (neural network) |
| Context | Rule-based | Pattern-based | Deep contextual understanding |

**Why I Modified:**
- Converted Chat's explanation into a table so it is easier to read
- Added specific details about preprocessing requirements based on my testing
- Included the actual number of scores/features each model outputs

**What I Learned:**
- Each model has different strengths and optimal use cases
- Preprocessing requirements differ significantly between models

**AI Errors Found:**
- AI didn't explicitly mention that VADER benefits from keeping capitalization and punctuation
- AI was vague about specific preprocessing requirements
- Fixed by testing each model with different preprocessing approaches and documenting the results

---

## Task: Creating Markdown Table
**Date:** February 16, 2026  
**Time spent:** 5 minutes

**What I was trying to do:**
Create a markdown table to display the model comparison in my notebook and README.

**AI Prompt:**
"How do I create a markdown table with headers and multiple rows?"

**AI Response:**
```markdown
| Header 1 | Header 2 | Header 3 |
|----------|----------|----------|
| Row 1 Col 1 | Row 1 Col 2 | Row 1 Col 3 |
| Row 2 Col 1 | Row 2 Col 2 | Row 2 Col 3 |
```

**What I Modified:**
I used the template as-is but filled it with my actual data:
```markdown
| Criterion | VADER | TextBlob | DistilBERT | Winner | Justification |
|-----------|-------|----------|------------|--------|---------------|
| **Overall Accuracy** | 58% | 47% | 68% | DistilBERT | Transformer's contextual understanding gives it a 10-point edge over VADER |
```

**Why I Modified:**
- Added bolding to criterion names
- Added detailed justifications 

**What I Learned:**
- Markdown tables use pipes (|) and dashes (-)
- Bold text in tables uses ** syntax

**AI Errors Found:**
- None - the template was correct and worked perfectly

---

## Task: Export DataFrame to CSV
**Date:** February 16, 2026  
**Time spent:** 5 minutes

**What I was trying to do:**
Export my samples to a CSV file for manual encoding.

**AI Prompt:**
"How do I export my output to CSV in Google Colab?"

**AI Response:**
```python
data.to_csv('output.csv', index=False)
```

**What I Modified:**
```python
# Export to CSV
samples = data.sample(100, random_state=42).reset_index(drop=True)
samples.to_csv("BA_reviews_sample.csv", index=False)

sample_100 = pd.read_csv("BA_reviews_sample.csv")
```

**Why I Modified:**
- Changed my file name
- Reimported the csv after I modified it

**What I Learned:**
- `to_csv()` is straightforward for exporting dataframes

**AI Errors Found:**
- None - it was a good template for me to use

---

## Task: Create Classification Report
**Date:** February 16, 2026  
**Time spent:** 10 minutes

**What I was trying to do:**
Generate classification reports for each model.

**AI Prompt:**
"How do I create a classification report to evaluate sentiment analysis model performance?"

**AI Response:**
```python
from sklearn.metrics import classification_report

print(classification_report(y_true, y_pred))
```

**What I Modified:**
```python
from sklearn.metrics import classification_report

# VADER classification report
print("VADER Classification Report:")
print(classification_report(sample_100['Label'], sample_100['vader_label'], 
                           target_names=['negative', 'neutral', 'positive']))

# TextBlob classification report
print("\nTextBlob Classification Report:")
print(classification_report(sample_100['Label'], sample_100['tb_label'], 
                           target_names=['negative', 'neutral', 'positive']))

# Transformer classification report
print("\nTransformer Classification Report:")
print(classification_report(sample_100['Label'], sample_100['emotion_label'], 
                           target_names=['negative', 'neutral', 'positive']))
```

**Why I Modified:**
- Created separate reports for each of the three models
- Added `target_names` parameter 
- Referenced actual column names from my dataframe

**What I Learned:**
- `target_names` makes the output much more interpretable
- Comparing reports side-by-side helps identify which models perform best on which classes

**AI Errors Found:**
- AI used generic variable names (y_true, y_pred) without showing how to apply to real data

---

## Task: Create Confusion Matrix Visualizations
**Date:** February 16, 2026  
**Time spent:** 20 minutes

**What I was trying to do:**
Create visual confusion matrices for all three models.

**AI Prompt:**
"How do I create a confusion matrix heatmap in Python using seaborn for sentiment analysis results?"

**AI Response:**
```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.show()
```

**What I Modified:**
```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Create figure with subplots for all three models
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# VADER confusion matrix
cm_vader = confusion_matrix(sample_100['Label'], sample_100['vader_label'], 
                            labels=['neg', 'neu', 'pos'])
sns.heatmap(cm_vader, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Neutral', 'Positive'],
            yticklabels=['Negative', 'Neutral', 'Positive'],
            ax=axes[0])
axes[0].set_title('VADER Confusion Matrix')
axes[0].set_ylabel('True Label')
axes[0].set_xlabel('Predicted Label')

# TextBlob confusion matrix
cm_tb = confusion_matrix(sample_100['Label'], sample_100['tb_label'], 
                         labels=['neg', 'neu', 'pos'])
sns.heatmap(cm_tb, annot=True, fmt='d', cmap='Greens', 
            xticklabels=['Negative', 'Neutral', 'Positive'],
            yticklabels=['Negative', 'Neutral', 'Positive'],
            ax=axes[1])
axes[1].set_title('TextBlob Confusion Matrix')
axes[1].set_ylabel('True Label')
axes[1].set_xlabel('Predicted Label')

# Transformer confusion matrix
cm_emotion = confusion_matrix(sample_100['Label'], sample_100['emotion_label'], 
                              labels=['neg', 'neu', 'pos'])
sns.heatmap(cm_emotion, annot=True, fmt='d', cmap='Oranges', 
            xticklabels=['Negative', 'Neutral', 'Positive'],
            yticklabels=['Negative', 'Neutral', 'Positive'],
            ax=axes[2])
axes[2].set_title('Transformer Confusion Matrix')
axes[2].set_ylabel('True Label')
axes[2].set_xlabel('Predicted Label')

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.show()
```

**Why I Modified:**
- Created a 3-subplot figure to show all models side-by-side for easy comparison
- Added different color schemes for each model (Blues, Greens, Oranges)
- Added proper axis labels and titles 
- Added `xticklabels` and `yticklabels` 
- Used `tight_layout()` to prevent label overlap

**What I Learned:**
- Confusion matrices visually show where models make mistakes
- Side-by-side comparison reveals relative model strengths/weaknesses
- Color schemes help differentiate between models
- `annot=True` shows the actual counts in each cell

**AI Errors Found:**
- AI provided only a basic single-matrix example
- AI didn't show how to create multiple subplots for comparison
- AI didn't include labels for axes or customization options

---

## Task: Test Model Processing Speed
**Date:** February 16, 2026  
**Time spent:** 15 minutes

**What I was trying to do:**
Measure and compare the processing speed of VADER, TextBlob, and DistilBERT to understand the performance trade-offs.

**AI Prompt:**
"How do I measure the execution time of a function in Python to compare model performance?"

**AI Response:**
```python
import time

start_time = time.time()
# Your code here
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Execution time: {elapsed_time} seconds")
```

**What I Modified:**
```python
import time

# Test VADER speed
start = time.time()
for review in sample_100['text']:
    vader_analyzer.polarity_scores(review)
vader_time = (time.time() - start) * 1000 / len(sample_100)  # ms per review

# Test TextBlob speed
start = time.time()
for review in sample_100['text']:
    TextBlob(review).sentiment.polarity
tb_time = (time.time() - start) * 1000 / len(sample_100)  # ms per review

# Test Transformer speed
start = time.time()
for review in sample_100['text']:
    inputs = tokenizer(review, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
emotion_time = (time.time() - start) * 1000 / len(sample_100)  # ms per review

print(f"Speed Comparison (ms per review):")
print(f"VADER: {vader_time:.2f} ms")
print(f"TextBlob: {tb_time:.2f} ms")
print(f"Transformer: {emotion_time:.2f} ms")
```

**Why I Modified:**
- Tested each model separately with the same test set
- Converted time to milliseconds per review for easier comparison
- Calculated average time per review rather than total time

**What I Learned:**
- Speed differences become critical at scale (thousands of reviews)
- Converting to milliseconds per review makes comparisons more intuitive

**AI Errors Found:**
- AI showed basic timing but didn't explain how to get per-item averages

---

## Task: Rewording Analysis Responses
**Date:** February 17, 2026  
**Time spent:** 30 minutes

**What I was trying to do:**
I had written my analysis answers for the assignment but the wording wasn't the best. I wanted to make the explanations more clear to the general audience.

**AI Prompt:**
"I've written analysis answers for my sentiment analysis assignment but some of the wording is awkward and unclear. Can you help me rephrase these sections to make them clearer while keeping all the technical details and my insights? Here are my original responses: [pasted my original text]"

**AI Response:**
AI provided reworded versions of my analysis that:
- Made sentence structure clearer and more concise
- Maintained all my technical observations and findings
- Used better wording
- Fixed grammatical issues and run-on sentences

**What I Modified:**
- Reviewed each reworded section carefully to ensure it still reflected my actual analysis
- Adjusted some phrasings to sound more like my own voice rather than overly formal
- Made sure all the technical details from my work were kept
- Verified that all my key insights were still being highlighted

**What I Learned:**
- AI is excellent for identifying awkward phrasing and suggesting clearer alternatives
- It's important to review AI-reworded content to maintain your own voice
- Breaking long sentences into shorter ones makes it easier to read and interpret

**AI Errors Found:**
- AI sometimes made the tone too formal/academic
- AI occasionally removed specific technical details that were important
- Some rewordings changed the emphasis of my points slightly
