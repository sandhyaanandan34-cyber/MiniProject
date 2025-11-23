# Mini Project : Education & Career Success

Objective:

The goal is to analyse the dataset and predict the career success based on the features "Field of Study, Work-life Balance, Internship & Projects Completed, Academics, Career Satisfaction". The output will be a binary categorisation of Yes/No to indicate whether the chosen field_of_study has career success or not.

Source : https://drive.google.com/file/d/1lFXvn_E9dQrl2Frvx1Pz7DXxpJu6DZn3/view?usp=drive_link

Models Used : Logistic Regression 
Reason: 
- Logistic regression is used for binary classification by modeling the probability of an outcome. It uses a sigmoid function to map the output to a value between 0 and 1.  
- It provides coefficients that directly show how each feature influences the probability of success
- Unlike plain classification models, logistic regression gives probability scores
- It’s simple, fast to train, and gives you a benchmark before trying more complex models

Key Results:
- Evaluation Metrics Used:
- Accuracy – proportion of correct predictions across all samples.
- F1 Score – harmonic mean of precision and recall, useful for imbalanced outcomes.
- AUC (Area Under ROC Curve) – measures the ability of the model to distinguish between classes.
- Precision & Recall – to understand trade‑offs between false positives and false negatives.

Evaluation Metrics Result:
- Accuracy: ~82% (the model correctly classified most student career outcomes)
- F1 Score: ~0.80 (balanced performance across precision and recall, showing robustness even with mixed data)
- AUC (ROC Curve): ~0.88 (strong ability to distinguish between successful vs. less successful career outcomes)
- Precision & Recall: Both around 0.80, indicating the model is reliable in identifying true positives without too many false alarms.

Key Insights:
- Academic Strength (GPA, SAT, University GPA) had the strongest positive influence on career success probability.
- Practical Experience (internships, projects, certifications) contributed significantly, especially for higher salary and faster promotions.
- Social Capital (networking, soft skills) consistently improved job offers and satisfaction scores.
- Career Outcomes (work‑life balance, entrepreneurship, gender, field category) showed high and meaningful effects.




