# Amazon Product Reviews Prediction Model
Machine learning model I created for my cs74 class final project


# Introduction

This report presents a comprehensive analysis of classification models for predicting the overall ratings of Amazon product reviews. The goal of the study was to develop accurate binary classification model, multiclass classification models, and clustering models to classify products as "good" or "bad" based on their ratings. We utilized a dataset consisting of Amazon product reviews, including various features such as review text, reviewer information, and product category. The study involved several key steps, including data preprocessing, feature selection/engineering, label generation, and model training/evaluation.

The dataset used in this study consists of two .csv files: 'training.csv' and 'test.csv'. The dataset includes various fields such as overall rating, review verification, review text, reviewer information, product ID, reviewer name, and other relevant features. The training dataset was utilized for model development and evaluation, while the test dataset was used to assess the model's performance on unseen data.

The initial step in the analysis involved data preprocessing to ensure the dataset's quality and suitability for modeling. This included handling missing values, encoding categorical variables, and vectorizing text columns using the TFIDF vectorizer. Missing values were removed from the dataset to ensure data integrity. Categorical variables, such as the 'verified' field, were encoded using one-hot encoding to transform them into numerical representations suitable for modeling. The 'reviewText' and 'summary' columns were vectorized using the TFIDF vectorization technique, capturing the importance of words in the reviews.

Feature selection is a crucial step in model development, as it helps identify the most relevant features for predicting the overall rating. In this study, we analyzed the available features and selected a subset that showed potential significance in determining the product rating. Additional feature engineering techniques, such as dimensionality reduction or feature transformation, were not employed in this analysis, as the selected features provided adequate discriminatory power for the classification task.

A function called findBest was utilized all throughout the different classifiers and models to determine which parameters gave the best results. This meant the classification model would run continuously over a wide range of parameters and this resulted in long waits as the function would take a long time to finish, but it delivered favorable parameters to optimize the model. The “best” score was found by finding the highest f1 macro score amongst all the different parameters’ results.

# Binary Classification

To facilitate binary classification, the dataset's overall ratings were converted into binary labels based on specified cutoff values. For each cutoff value (1, 2, 3, and 4), samples with a rating less than or equal to the cutoff were labeled as 0 (indicating a "bad" product), while samples with a rating above the cutoff were labeled as 1 (indicating a "good" product). This allowed the models to learn to distinguish between positive and negative sentiments based on the product ratings.

Columns 7 and 8 (indexed 6 and 7) of the data csv files contained the relevant information, which were customer reviews and review summaries. To convert the text documents into numerical feature vectors, Term Frequency - Inverse Document Frequency (TFIDF) was employed, providing a convenient approach. This method assigns TFIDF scores to the terms in the vocabulary, enabling the representation of each document as a vector in the transformed numerical space.

The function binary_results served to measure the performance of the binary classification model and was run for each iteration and condition. This function returned the f1 macro score, roc score, accuracy, precision, recall, true negatives, and the confusion matrix. 

The method I used to tune the hyperparameters was by using the GridSearchCV method, which is an exhaustive function that involves specifying a grid of any number of different hyperparameters, and the computer will build a model for each combination of hyperparameters to see which one performs the best. Through this function, I determined which hyperparameters were optimal for my model.

## Linear Regression

I chose to tune the hyperparameters ‘C’, ‘penalty’, and ‘solver’ because ‘C’ controls the inverse of the regularization strength. By tuning 'C', I can adjust the balance between fitting the training data well and prevent overfitting. Tuning ‘penalty’ and ‘solver’ allows me to choose which regularization technique is used and that helps me choose the appropriate regularization technique based on the dataset. I created a function called linear_regression that ran the linear regression and printed the best hyperparameters, best score, ran the binary_results function, plotted the roc curve, and created the csv file for the kaggle competitions.


### Cutoff = 1

I found that ngram=(1,3), min_df=0.0, and max_df=0.3 yielded the best results for my linear regression model with a cutoff of 1. The best hyperparameters were a C of 1.0, a penalty of 'l2', and a solver of ‘'liblinear'. The best score I got was  0.8686833230006709. My scores were: 
Accuracy : 0.8730729701952723

Precision : 0.8296149633499031

Recall : 0.7528968652928685

True Negatives : 659.0000

F1 Score (Macro) : 0.7815327530074511 (Above Baseline)

AUC-ROC : 0.7528968652928683

Confusion Matrix : 
[[ 659  542]
 [ 199 4438]]



### Cutoff = 2

I found that the parameters of ngram=(1,3), min_df=0.001, and max_df=0.99 yielded the best results for my linear regression model with a cutoff of 2. The best hyperparameters were a C of 1.0, a penalty of 'l2', and a solver of ‘'liblinear'. The best score I got was  0.8071534891326273. My scores were: 

Accuracy : 0.8105515587529976

Precision : 0.8066070236823801

Recall : 0.7966325702833147

True Negatives : 1711.0000

F1 Score (Macro) : 0.8005067270293303 (Above Baseline)

AUC-ROC : 0.7966325702833148

Confusion Matrix : [[1711  655]
 [ 451 3021]]




### Cutoff = 3

I found that the parameters of ngram=(1,3), min_df=0.0001, and max_df=0.999 yielded the best results for my linear regression model with a cutoff of 3. The best hyperparameters were a C of 1.0, a penalty of 'l2', and a solver of ‘'liblinear'. The best score I got was  0.8188012902534382.
My scores were: 

Accuracy : 0.817060637204522

Precision : 0.8098721815681353

Recall : 0.7994990126567505

True Negatives : 3144.0000

F1 Score (Macro) : 0.8037950879100961 (Above Baseline)

AUC-ROC : 0.7994990126567506

Confusion Matrix : 
[[3144  445]
 [ 623 1626]]


### Cutoff = 4

I found that the parameters of ngram=(1,3), min_df=0.001, and max_df=0.999 yielded the best results for my linear regression model with a cutoff of 4. The best hyperparameters were a C of 1.0, a penalty of 'l2', and a solver of ‘'liblinear'. The best score I got was  0.8541574308041758.
My scores were: 

Accuracy : 0.8532031517643028

Precision : 0.7762315951018731

Recall : 0.6965918825857622

True Negatives : 4488.0000

F1 Score (Macro) : 0.7239208951177556 (Above Baseline)

AUC-ROC : 0.6965918825857623

Confusion Matrix : 
[[4488  239]
 [ 618  493]]

​​

## Naive Bayes

I used the Naive Bayes classifier because it was a model suitable for classification with discrete features, such as word counts for text classification. Thus, I thought it would be best especially since one of the points of analysis in the amazon product csv files were the summary, which are a collection of texts.

I chose to tune the hyperparameters ‘alpha’ and ‘fit_prior’. 'alpha' represents the smoothing parameter or additive smoothing and by tuning this I can handle the issue of zero probabilities in the training data which lets me control the smoothing strength. ‘fit_prior’ controls whether class prior probabilities are learned from the training data or if they are assumed to be uniform. By tuning ‘fit_prior’, I can choose whether to estimate class prior probabilities from the training data or use uniform class priors.

I created a function called naiveBayes that ran the classification model and printed the best hyperparameters, best score, ran the binary_results function, plotted the roc curve, and created the csv file for the kaggle competitions.

### Cutoff = 1
The best hyperparameters I found that yielded the best model were an ‘alpha’ value of 9.7, a ‘fit_prior’ value of False'. The best score I got was  0.8556304793410432.
My scores were: 

Accuracy : 0.855258650222679

Precision : 0.7963493326261822

Recall : 0.7178001023191437

True Negatives : 579.0000

F1 Score (Macro) : 0.7453881206516495 (Above Baseline)

AUC-ROC : 0.7178001023191437

Confusion Matrix : 
[[ 579  614]
 [ 231 4414]]


### Cutoff = 2
The best hyperparameters I found that yielded the best model were an ‘alpha’ value of 7.0, a ‘fit_prior’ value of False'. The best score I got was  0.8034192220858033.
. My scores were: 

Accuracy : 0.8004453579993148

Precision : 0.7930536343430044

Recall : 0.7903436342718422

True Negatives : 1735.0000

F1 Score (Macro) : 0.7915960562015637 (Above Baseline)

AUC-ROC : 0.7903436342718422

Confusion Matrix : 
[[1735  614]
 [ 551 2938]]



### Cutoff = 3
The best hyperparameters I found that yielded the best model were an ‘alpha’ value of 6.5, a ‘fit_prior’ value of False'. The best score I got was  0.8072559529898781. 
My scores were: 

Accuracy : 0.8016443987667009

Precision : 0.7924014565502362

Recall : 0.7933506645026643

True Negatives : 2941.0000

F1 Score (Macro) : 0.7928635334143854

AUC-ROC : 0.7933506645026644

Confusion Matrix : 
[[2941  590]
 [ 568 1739]]

### Cutoff = 4
The best hyperparameters I found that yielded the best model were an ‘alpha’ value of 9.0, a ‘fit_prior’ value of False'. The best score I got was  0.832710902669724. 
My scores were: 

Accuracy : 0.8256252141144228

Precision : 0.7214330406835604

Recall : 0.6610589450778528

True Negatives : 4374.0000

F1 Score (Macro) : 0.681388254795728

AUC-ROC : 0.6610589450778529

Confusion Matrix : 
[[4374  323]
 [ 695  446]]


## Ridge Classifier

I used the Ridge Classifier because it reduces the impact of highly correlated features which makes it more robust and less prone to overfitting. And with its ability to handle datasets with more features than samples it can provide more stable and reliable results compared to standard logistic regression.

I chose to tune the hyperparameters ‘alpha’ and ‘solver’.'alpha' represents the regularization strength and controls the amount of regularization applied to the model's coefficients, helping to prevent overfitting. Tuning this hyperparameter allows me to prevent overfitting. ‘solver’ allows me to choose which regularization technique is used and that helps me choose the appropriate regularization technique based on the dataset. I created a ridgeClassifier function that ran the classification model and printed the best hyperparameters, best score, ran the binary_results function, plotted the roc curve, and created the csv file for the kaggle competitions.

### Cutoff = 1
The best hyperparameters I found that yielded the best model were an ‘alpha’ value of 1.0, a ‘solver’ value of ‘auto’. The best score I got was  0.863921236294696.
My scores were: 

Accuracy : 0.8609112709832134

Precision : 0.8213084752777454

Recall : 0.7139940765965266

True Negatives : 558.0000

F1 Score (Macro) : 0.7477695179945413 (Above Baseline)

AUC-ROC : 0.7139940765965267

Confusion Matrix : 
[[ 558  644]
 [ 168 4468]]


### Cutoff = 2
The best hyperparameters I found that yielded the best model were an ‘alpha’ value of 5.0, a ‘solver’ value of ‘lsqr’. The best score I got was  0.8074961250314953.
My scores were: 

Accuracy : 0.816032887975334

Precision : 0.8133439859416176

Recall : 0.7998902040025743

True Negatives : 1684.0000

F1 Score (Macro) : 0.8048757200756795 (Above Baseline)

AUC-ROC : 0.7998902040025744

Confusion Matrix : 
[[1684  663]
 [ 411 3080]]


### Cutoff = 3
The best hyperparameters I found that yielded the best model were an ‘alpha’ value of 5.0, a ‘solver’ value of ‘lsqr’. The best score I got was  0.8147586514845493.
My scores were: 

Accuracy : 0.81551901336074

Precision : 0.8129970655895251

Recall : 0.7945270565841431

True Negatives : 3166.0000

F1 Score (Macro) : 0.8011170485620024 (Above Baseline)

AUC-ROC : 0.7945270565841431

Confusion Matrix : 
[[3166  386]
 [ 691 1595]]


### Cutoff = 4
The best hyperparameters I found that yielded the best model were an ‘alpha’ value of 1.0, a ‘solver’ value of ‘lsqr’. The best score I got was  0.8491556122438499.
My scores were: 

Accuracy : 0.8528605686879068

Precision : 0.7779220779220779

Recall : 0.6890778959098159

True Negatives : 4508.0000

F1 Score (Macro) : 0.7180277061266948 (Above Baseline)

AUC-ROC : 0.689077895909816

Confusion Matrix : 
[[4508  222]
 [ 637  471]]


## Kaggle

Kaggle username: jusungprk

Binary Cutoff = 1

Best Score: 0.77443

Binary Cutoff = 2

Best Score: 0.79866

Binary Cutoff = 3

Best Score: 0.80753

Binary Cutoff = 4

Best Score: 0.70997

Multiclass

Best Score: 0.48043


# Multiclass Classification

For multiclass classification, since I didn’t need to run the function 4 times for each cutoff like in the binary classification, I didn’t create a function for each individual classifier and instead ran all the code directly. However, I still created a function to analyze the performance of the multiclass classifiers and also created functions for creating the csv files for the kaggle competition and for creating the 6-line ROC curve graph. The multiclass_results function differed from the binary_results function in that a histogram was created rather than a bar graph and labels were generated for each classification rather than 1s and 0s that we used for the binary classification. Again, GridSearchCV was used to find the best hyperparameters for each classifier. 
 
## Logistic Regression
I chose to tune the hyperparameters ‘C’, ‘penalty’, and ‘multi_class’ because ‘C’ controls the inverse of the regularization strength. By tuning 'C', I can adjust the balance between fitting the training data well and prevent overfitting. Tuning ‘penalty’ allows me to choose which regularization technique is used and that helps me choose the appropriate regularization technique based on the dataset. Similarly, I chose to tune the ‘multi_class’ hyperparameter because it allows me to choose how the multiclass classification is performed. For the logistic regression classifier, the best hyperparameters I found that yielded the best model were an ‘C’ value of 1.0, a ‘multi_class’ value of ‘ovr’, and a ‘penalty’ value of ‘l2’. The best score I got was 0.4933022338095162.
My scores were:

Accuracy: 0.4950325453922576

Precision: 0.48575881170716706

Recall: 0.4948490091375651

F1 Score (Macro): 0.48832687334296915 (Above Baseline)

Confusion Matrix: 
      [[829, 247,  64,  35,  27],
       [309, 473, 231, 105,  61],
       [122, 283, 383, 259, 115],
       [ 69, 110, 175, 474, 337],
       [ 39,  58,  73, 229, 731]])

ROC Scores:

fit_time: [1.87732387, 1.72800708, 1.91928506, 1.84042406, 1.83134985]

score_time: [0.02954888, 0.02482891, 0.02539086, 0.02401996, 0.02216721]

test_accuracy: [0.47207948, 0.45854745, 0.50650908, 0.49537513, 0.49563132]

test_f1_macro: [0.46255165, 0.45285428, 0.50014639, 0.49205322, 0.48379923]

test_recall_macro: [0.47192421, 0.45959645, 0.50700626, 0.49538974, 0.49627959]

test_precision_macro: [0.46073661, 0.4493891 , 0.49812876, 0.48976733, 0.48187737]

test_roc_auc_ovr: [0.7920571 , 0.77719137, 0.81238482, 0.80701711, 0.80867718]



 

## Naive Bayes
I chose to tune the hyperparameters ‘alpha’ and ‘fit_prior’. 'alpha' represents the smoothing parameter or additive smoothing and by tuning this I can handle the issue of zero probabilities in the training data which lets me control the smoothing strength. ‘fit_prior’ controls whether class prior probabilities are learned from the training data or if they are assumed to be uniform. By tuning ‘fit_prior’, I can choose whether to estimate class prior probabilities from the training data or use uniform class priors. For the Naive Bayes classifier, the best hyperparameters I found that yielded the best model were an ‘alpha’ value of 10.0, and a ‘fit_prior’ value of False. The best score I got was  0.4929595450881763.
My scores were:

Accuracy: 0.47670435080507023

Precision: 0.47407011261843907

Recall: 0.47634883691060637

F1 Score (Macro): 0.4745282745863488 (Above Baseline)

Confusion Matrix: 
      [[781, 266,  68,  41,  33],
       [305, 434, 244, 119,  65],
       [ 87, 282, 435, 255, 112],
       [ 59, 131, 192, 482, 299],
       [ 42,  64,  65, 326, 651]]

Roc scores:

fit_time: [0.01952696, 0.01277709, 0.01543331, 0.01567626, 0.01986003]

score_time: [0.039644  , 0.02953601, 0.02421784, 0.03057098, 0.0321691 ]

test_accuracy: [0.47550531, 0.45700582, 0.5080507 , 0.49691675, 0.50505397]

test_f1_macro: [0.46641153, 0.45493305, 0.50228132, 0.49513601, 0.49288253]

test_recall_macro: [0.47530013, 0.45814955, 0.50814352, 0.4966959 , 0.50561107]

test_precision_macro: [0.46477137, 0.45606656, 0.50243506, 0.49489709, 0.49043344]

test_roc_auc_ovr: [0.79161599, 0.77785463, 0.80594318, 0.80404124, 0.81051717]





## Random Forest Classifier
I used a Random Forest Classifier for its ability to aggregate predictions from multiple decision trees and reduce the impact of individual noisy or biased trees which improves overall prediction performance. It also captures non-linear relationships, interactions, and feature dependencies effectively which makes it suitable for a wide range of datasets. I chose to tune 'n_estimators' as it determines the number of decision trees so I could balance between model accuracy and performing and computational efficiency. ‘Max_depth’ controls the depth of each decision tree which helps model more complex interactions within the model. By tuning ‘max_depth’, I could also balance between computational efficiency and model performance. For the RF classifier, the best hyperparameters I found that yielded the best model were an ‘n_estimators’ value of 100, and a ‘max_depth’ value of None. The best score I got was  0.459933603326595.
My scores were:

Accuracy: 0.47173689619732784

Precision: 0.46003684182321986

Recall: 0.46982237552964295

F1 Score (Macro): 0.4619079958711111

Confusion Matrix: 
      [[838, 176,  79,  48,  40],
       [351, 381, 220, 135,  71],
       [145, 235, 356, 279, 145],
       [ 54, 100, 195, 481, 306],
       [ 61,  69,  79, 296, 698]]

ROC Scores:

fit_time: [63.88108492, 61.55098915, 72.62881112, 82.92368627, 75.63247085]

score_time: [0.9511199 , 1.38416696, 1.11479902, 0.96590376, 1.32639122]

test_accuracy: [0.44587187, 0.42463172, 0.47242206, 0.46060295, 0.46256639]

test_f1_macro: [0.43394979, 0.41799814, 0.4640719 , 0.45167259, 0.44914464]

test_recall_macro: [0.44587317, 0.42534008, 0.47300914, 0.46100327, 0.46292459]

test_precision_macro: [0.43231532, 0.41496043, 0.46164666, 0.45041352, 0.44527164]

test_roc_auc_ovr: [0.76841915, 0.74443996, 0.77852589, 0.77942477, 0.77745422]





# Clustering

K-Means Clustering
I used K-means clustering to cluster the product reviews in the test.csv dataset. I clustered by product types, of which there were six of, so I set k as 6. To check the quality of my clustering model, I used the metrics Silhouette score and Rand index. 


Silhouette score: 0.6741702591579974 (Above Baseline)

Rand index: 0.18287114907563293

# Model Evaluation

All my f1 macro scores for each part (binary, multiclass, clustering) were above the baseline shared for the project.

To evaluate the performance of the binary and multiclass classification models, we employed several performance metrics, including confusion matrix, ROC curve, AUC score, macro F1 score, and accuracy. For the clustering model, we employed the Silhouette score and the Rand index. These metrics provide insights into different aspects of model performance, such as the ability to correctly classify positive and negative instances, the discriminative power of the model, and overall accuracy.

Confusion Matrix: The confusion matrix provides a tabular representation of the model's performance, showing the counts of true positives, true negatives, false positives, and false negatives. It helps assess the model's ability to classify positive and negative instances accurately.
ROC Curve and AUC: The roc curve illustrates the performance of a binary classifier at various classification thresholds. It plots the true positive rate against the false positive rate. The AUC summarizes the overall performance of the classifier, with higher values indicating better discrimination between positive and negative instances.
Macro F1 Score: The macro F1 score is the harmonic mean of precision and recall, computed separately for each class and then averaged. It provides an overall measure of the model's ability to balance precision and recall across both positive and negative instances.
Accuracy: Accuracy represents the proportion of correctly classified instances out of the total instances. It provides a general measure of the model's performance but may not be sufficient in cases of imbalanced datasets.
Silhouette score: ​​The Silhouette score is a metric used to evaluate the quality of clustering algorithms and assess the compactness and separation of clusters in the data. It provides a quantitative measure of how well each data point fits within its assigned cluster by assessing cohesion and separation.
Rand Index: The Rand index is a metric used to evaluate the quality of clustering algorithms by comparing the similarity between the true class labels and the cluster assignments. It measures the percentage of pairwise data point relationships that are classified consistently, and it does this by counting the number of true positives, true negatives, false positives, and false negatives.

