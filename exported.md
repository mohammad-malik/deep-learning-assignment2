# Report: Legal Clause Similarity

## Dataset & Splits
- Train: 101,905 clauses across 394 labels (203,810 pairs)
- Validation: 21,846 clauses across 394 labels (43,692 pairs)
- Test: 22,061 clauses across 394 labels (44,122 pairs)
- Negative/positive ratio per split: 1.0:1

## Architectures & Training
1. BiLSTM Siamese: Embedding(30k,128) -> BiLSTM(+/-128) -> mean pool -> LayerNorm+MLP.
2. Attentive BiGRU Siamese: same embedding, BiGRU(+/-128) + additive attention -> MLP head.
3. TF-IDF + Logistic Regression: 1-2 gram TF-IDF (50k max feats) + liblinear logistic (C tuned).

Shared hyperparams: max_len=150, train_batch=64, eval_batch=128, Adam lr=0.001, weight_decay=0.0001, epochs=50, patience=3, grad clip=2.0.
## Test Metrics
              accuracy  precision    recall        f1   roc_auc    pr_auc  threshold  train_time_sec
BiLSTM        0.868614   0.857923  0.883550  0.870548  0.946642  0.949848   0.681313     9490.024658
AttentiveGRU  0.895993   0.902432  0.887992  0.895154  0.961654  0.964579   0.645903    17230.376958
TFIDF         0.507411   0.503776  0.988668  0.667452  0.586827  0.573302   0.217144       38.753078

ROC curve image: roc_curves.png | PR curve image: pr_curves.png

## Error Patterns
- False positives: clauses sharing boilerplate but diverging in scope or carve-outs.
- False negatives: paraphrases with modality shifts ("may" vs "shall") or rearranged conditions.