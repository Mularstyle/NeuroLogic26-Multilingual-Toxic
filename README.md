# Multilingual Toxic Comment Classification

**Team**: TsukyNLP

**Members**:
- Dachatorn Chitnatharn
- Chakorn Silalai
- AlizionS

## Challenge Overview
This project tackles Challenge 3 of NeuroLogic '26: building a binary classifier to detect toxic comments in multilingual text (English and Hindi).

## Approach

### Model Selection
I used **XLM-RoBERTa-base**, a multilingual transformer model pre-trained on 100 languages. This choice makes sense for our task since:
- It handles both English and Hindi without needing separate models
- Pre-trained on large multilingual corpora, so it already understands cross-lingual patterns
- Good balance between performance and computational cost

### Data Preprocessing
Pretty straightforward here:
- Filled missing text values with empty strings
- Split training data 80/20 for train/validation
- Used stratified split to maintain label balance
- Tokenized with max length of 128 tokens (most comments are shorter than this)

### Training Configuration
- 3 epochs (more would likely overfit on this dataset size)
- Batch size: 16 for training, 32 for evaluation
- Learning rate: default from Hugging Face (5e-5)
- Warmup steps: 500
- FP16 training enabled for faster computation on GPU

### Evaluation Metric
Primary metric is **ROC-AUC** as specified in the challenge requirements. Also tracked accuracy, F1, precision, and recall during training to get a fuller picture of model performance.

## Results

### Validation Performance
- **ROC-AUC**: 0.9879
- **Accuracy**: 0.9561 (95.61%)
- **F1-Score**: 0.9566
- **Precision**: 0.9457
- **Recall**: 0.9677

The model shows strong performance across both classes. With an ROC-AUC of 0.9879, it demonstrates excellent ability to distinguish between toxic and non-toxic comments. The high recall (96.77%) means we're catching most toxic comments, while maintaining good precision (94.57%) to avoid false positives.

### Visual Results
Detailed performance visualizations including confusion matrix, ROC curve, and training metrics can be found in the `results_screenshots/` folder.

## How to Reproduce

### Requirements
Install all dependencies using:
```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install transformers datasets accelerate openpyxl scikit-learn pandas numpy matplotlib seaborn
```

### Running the Code
1. Upload `toxic_labeled.xlsx` and `toxic_no_label_evaluation.xlsx` to `/content/` in Google Colab
2. Open `toxic_classification.ipynb`
3. Run all cells in order
4. Output file `toxic_no_label_evaluation_predicted.xlsx` will be saved in `/content/`
5. Screenshots of results will be generated:
   - Cell 12: Validation scores summary
   - Cell 15: Confusion matrix, ROC curve, and predictions distribution
   - Save these to `results_screenshots/` folder

### Environment
- Python 3.8+
- Google Colab (with GPU runtime recommended)
- All dependencies listed in requirements above

## File Structure
```
.
├── toxic_classification.ipynb          # Main notebook with full pipeline
├── toxic_labeled.xlsx                  # Training data (not included in repo)
├── toxic_no_label_evaluation.xlsx      # Test data (not included in repo)
├── toxic_no_label_evaluation_predicted.xlsx  # Predictions output
├── requirements.txt                    # Python dependencies
├── results_screenshots/                # Screenshots of model performance
│   ├── validation_scores.png           # Validation metrics summary
│   ├── confusion_matrix.png            # Confusion matrix visualization
│   ├── roc_curve.png                   # ROC curve with AUC score
│   └── predictions_distribution.png    # Distribution of predictions
└── README.md                           # This file
```

## Notes
- Labels are binary: 0 (non-toxic) and 1 (toxic)
- Model predictions maintain this exact format as required
- Training takes approximately 15-20 minutes on Colab's T4 GPU
- Validation set used to prevent overfitting and select best checkpoint

## Potential Improvements
If I had more time, I'd try:
- Experimenting with mBERT or other multilingual models
- Data augmentation (back-translation, synonym replacement)
- Ensemble methods combining multiple models
- Fine-tuning hyperparameters more systematically
- Adding language-specific preprocessing for Hindi text
