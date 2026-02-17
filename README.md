# Sentiment Analysis of British Airways Airline Reviews

## Project Overview

This project compares three sentiment analysis models (VADER, TextBlob, and DistilBERT) on British Airways customer reviews to determine which approach best captures customer sentiment. The analysis includes comprehensive text preprocessing, model implementation across 3,701 reviews, and detailed accuracy evaluation on 100 manually labeled reviews.

The goal is to provide actionable insights for British Airways to understand customer sentiment patterns and identify which model is most suitable for analyzing airline customer feedback at scale.

## Dataset Description

**Source:** British Airways Airline Reviews (BA_AirlineReviews.csv)  
**Size:** 3,701 customer reviews

**Key Features:**
- **ReviewBody:** Main text containing customer feedback (primary analysis target)
- **OverallRating:** Numeric rating from 1-10
- **Individual ratings:** SeatComfort, CabinStaffService, GroundService, ValueForMoney, Food&Beverages, InflightEntertainment, Wifi&Connectivity (1-5 scale)
- **Metadata:** TypeOfTraveller, SeatType, Route, DateFlown, Aircraft, Recommended (yes/no)

**Why This Dataset:**
Airline reviews are ideal for sentiment analysis because they contain complex, mixed emotions. Passengers might love the in-flight service but hate delays, or appreciate comfort but criticize price. This complexity provides an excellent test case for comparing different sentiment analysis approaches and has direct real-world applications for customer experience improvement in the airline industry.

## Key Findings Summary

### Model Performance

| Model | Accuracy | Speed (ms/review) | Best Use Case |
|-------|----------|-------------------|----------------|
| **VADER** | 58% | 2.19 | Real-time monitoring, interpretability |
| **TextBlob** | 47% | 1.66 | Not recommended for this dataset |
| **DistilBERT** | 68% | 589 | Primary model (best accuracy) |

### Detailed Insights

**DistilBERT (Winner - 68% accuracy):**
- Best overall performance with a 10-point advantage over VADER
- Excels at detecting specific emotions (anger, joy, sadness) beyond basic positive/negative classification
- Handles mixed sentiment reviews effectively, such as "good staff but terrible food"
- Can be fine-tuned on airline-specific data for even better results
- Recommended as the primary model for comprehensive sentiment analysis and reporting

**VADER (Strong Second - 58% accuracy):**
- Nearly 270x faster than DistilBERT, making it ideal for real-time applications
- Excellent interpretability - can explain classifications with clear lexicon-based rules
- Works well when reviews use clear positive/negative words without complex linguistic structures
- Struggles with modifier combinations like "Incredible rude" (interprets "Incredible" as positive)
- Recommended for real-time monitoring and when interpretability is required

**TextBlob (Poor Performance - 47% accuracy):**
- Essentially coin-flip accuracy, not recommended for this dataset
- Simple averaging approach misses context and produces moderate scores even for strong sentiment
- Cannot handle intensifiers properly ("Incredible rude" classified as positive)
- Only useful if subjectivity scores are specifically needed

### Common Challenges

**All three models struggle with:**
- Sarcasm and irony ("Great job losing my luggage!" classified as positive)
- Neutral/mixed reviews with balanced positive and negative elements
- Complex negation structures ("not bad at all" often interpreted as negative)
- Domain-specific languag
