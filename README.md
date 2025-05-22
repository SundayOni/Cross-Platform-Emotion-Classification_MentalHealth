# 🧠 Mental Health Prediction via Emotion Classification
## Cross-Platform Emotion Analysis Using Twitter and Reddit Data


## Introduction
Globally, mental health concerns have become a significant public health issue, impacting millions of people with rising rates of anxiety, depression, and other illnesses. The World Health Organisation (WHO) estimated that 970 million individuals worldwide, or 1 in 8 people, suffered from a mental illness in 2019, with anxiety and depressive disorders being the most prevalent types of mental illness. Due to the COVID-19 epidemic, the number of individuals suffering from anxiety and depression illnesses increased dramatically in 2020, while one in four people worldwide may experience neurological and mental health issues at some point in their lives (E and Juliet 2023). In the UK alone, 72 million working days are missed annually due to mental illness, making it the second-largest cause of disease burden in England, costing approximately £34.9 billion (MHFA England 2019).

## Problem Statement
The study by Benrouba and Boudour (2023) suggests that although mental health conditions like anxiety and depression are becoming more common, traditional methods of assessment like 
self-report questionnaires and clinical interviews, are limited in that they are retrospective in nature and prone to bias. While many research have emphasised the enormous potential of social media data, existing methods of predicting mental health conditions using social media data tend to focus on single-platform analysis, ignoring the diverse emotional expressions as well as the contextual and linguistic differences across different platforms. This project addresses this gap by investigating cross-platform emotion classification to enhance the accuracy and reliability of mental health predictions. 

## Aim of the project
This project aims to investigate how the prediction of mental illness can be improved through 
the integration of emotion data across two social media platforms, Reddit and Twitter. The goal is to develop and evaluate models that can classify emotions accurately across these platforms. In addition, the study will elucidate the challenges of effective cross-platform emotion classification for improved prediction of mental health disorders. Specific objectives include:
- To apply Natural Language Processing (NLP) techniques for emotion detection in diverse 
text formats.   
- To examine how linguistic patterns and emotional expressions differ across Twitter and 
Reddit in the context of mental health discussions.  
- To develop machine learning models for early prediction of mental disorders. 
- To improve patient outcomes by reducing mental health-related complications, thereby lowering healthcare costs associated with mental illnesses.

## Methodology
### 1. *Data Sources and Data Collection Technique*
This study utilised two primary datasets ethically obtained from Twitter and Reddit via their official APIs. These platforms were selected due to their widespread use, rich user-generated content related to emotions and mental health, and relatively permissive data access policies, which made them suitable for mental health (MH) research. Although expanding to more platforms could have enhanced the study's reliability, restrictions on data access posed limitations. The data collection involved selecting relevant keywords, such as “depression,” “anxiety,” “suicide,” and various MH-related hashtags on Twitter and targeting fifteen MH-focused subreddits on Reddit, identified in prior studies as highly active. Filters were applied to collect only English-language posts and tweets within the allowed time frames, noting that Reddit allowed retrieval of both old and recent content while Twitter was limited to more recent data due to API access level. Data scraping was performed using developer accounts and API endpoints accessed via Python libraries: Tweepy for Twitter and PRAW for Reddit.

### 2. *Tools and Libraries used*
This study was conducted using Google Colab (Python 3), where all data processing, NLP tasks, and machine learning modelling were carried out. The analysis leveraged various Python libraries, including:

- Pandas: Used for data cleaning, manipulation, and exploratory data analysis.
- Scikit-learn: Utilised for developing ML algorithms, performing data preprocessing (e.g., TF-IDF vectorisation), and evaluating model performance.
- Hugging Face Transformers: Applied for automatic text annotation and emotion classification using pre-trained transformer models.
- Matplotlib and Seaborn: Employed for creating visualisations to support data analysis and communicate insights effectively.

### 3. *Initial Preprocessing (Data Cleaning)*
- Duplicates, empty values (nulls), and irrelevant columns that do not support the analysis were removed to ensure data quality and focus.
- All text was standardised by converting to lowercase and cleaning out punctuations, special characters, URLs, hyperlinks, mentions, and contractions.
- Posts and tweets made by moderators or admins were excluded to ensure the analysis focuses solely on content generated by individuals potentially experiencing mental health issues.
- Tokenisation and stop word removal were carried out to break down the text into individual words and eliminate common, non-informative words, ensuring only meaningful terms were retained for analysis.
- Lemmatisation was performed to reduce words to their root forms, improving consistency and accuracy in text analysis.
- Annotation was done using pre-trained transformer models, which automatically labeled the dataset with six core emotions: happiness, anger, sadness, fear, surprise, and joy.

### 4. *Exploratory Data Analysis (EDA)*

The Twitter dataset contains mostly short texts ranging from 9 to 765 characters, with most tweets falling between 50 and 230 characters, showing a right-skewed distribution due to a few longer tweets. In contrast, Reddit posts are generally longer and more detailed, ranging from 0 to 4,363 characters, with most falling between 150 and 850 characters and a notable presence of lengthy outliers.

![Text_length_both](https://github.com/user-attachments/assets/c5b64e4f-a363-4f90-a565-0bfa28a6d7bd)

The figure below illustrates the word count distribution on both platforms. Twitter posts range from 3 to 84 words, with most tweets containing 5 to 30 words, showing a right-skewed distribution due to a few lengthier entries. Reddit posts, however, show a broader range from 1 to 780 words, also with the majority between 5 and 30 words, but with a longer tail suggesting that some users tend to write much more detailed and extended content.
![Word_count_both](https://github.com/user-attachments/assets/cfa8b75f-6e39-4fa3-a365-5d3454265ca8)


The figure below presents word clouds for the emotions "joy" and "sadness" across Twitter and Reddit. On Twitter, the "joy" class includes terms like "mental health", "hope", "recovery", and "treatment", suggesting that joyful conversations often involve themes of overcoming challenges, hope, and emotional support. In contrast, "sadness" on Twitter features words like "depression", "suicide", and "disorder", indicating a strong focus on mental health struggles. On Reddit, "joy" is associated with words like "life", "feel", and "have", pointing to personal growth and reflection, while "sadness" includes terms such as "trauma", "hate", "sleep", and "parent", hinting at emotional pain and difficulties in relationships or daily life. Overall, the word clouds highlight how both platforms express these emotions through discussions of mental health, self-awareness, and interpersonal connections.

![Wordclouds_both](https://github.com/user-attachments/assets/4c9230fb-c864-4616-8978-cf2c5c850ea0)


After preprocessing and emotion annotation, the majority of tweets were classified under "joy," followed by smaller proportions labeled as sadness, fear, anger, love, and very few as surprise. The dominance of the "joy" category in the Twitter dataset may point to challenges in filtering out posts from users who do not experience mental health issues but use similar hashtags, making accurate targeting difficult. In contrast, Reddit showed a more diverse emotional spread, with sadness making up the largest portion, followed by joy, fear, and anger. Only a small fraction of posts fell into the categories of love and surprise, likely due to Reddit’s structured mental health communities which foster more relevant emotional discourse.
![Emotion_distribution_both](https://github.com/user-attachments/assets/1e141992-ef03-4a44-961c-80e3b4533ac7)


### 5. *Final Preprocessing*
Final preprocessing steps included TF-IDF vectorisation to convert the text into numerical features, label encoding to transform emotion labels into numeric format, and addressing class imbalance by applying random oversampling to increase the representation of minority emotion classes. The dataset was then split into training and testing sets using an 80:20 ratio to prepare for modelling.

![final_prepro](https://github.com/user-attachments/assets/8b3a73dc-fe63-4353-8abc-12e2e94e923f)

![final_prepro2](https://github.com/user-attachments/assets/8a9d17c9-fad4-47cf-ae19-5f0f2e306e76)



### 6. *Model Training*
Two machine learning models, Support Vector Machine (SVM) and Naive Bayes, were initially trained and evaluated separately on the Twitter and Reddit datasets to compare their individual performances. Following this, a cross-platform evaluation was conducted by training the models on Twitter data and testing on Reddit data, and vice versa, to assess whether the emotion patterns learned from one platform could generalise effectively to the other.
![Modelling1](https://github.com/user-attachments/assets/d39e04e7-e57c-4733-894f-08afdb259186)

![modelling2](https://github.com/user-attachments/assets/8a6291ff-15cc-440f-bec6-f216b2520165)



### 7. *Model Evaluation*
The trained models were evaluated using key performance metrics such as accuracy, precision, recall, and F1 score. On single-platform evaluation, the SVM model achieved around 39% accuracy on Twitter and 38% on Reddit, while Naive Bayes reached approximately 24.5% and 24.3% respectively. In the cross-platform setup, when SVM was trained on Twitter and tested on Reddit, it achieved 38% accuracy, with the "sadness" class showing strong performance (85% recall, 55% F1 score). Similarly, training on Reddit and testing on Twitter yielded a lower overall accuracy of 32%, but "sadness" still performed relatively well, with a recall of 68% and F1 score of 44%.

![Eval1](https://github.com/user-attachments/assets/3cbfdebf-d1f6-4f39-adb8-9d3bc3651a9c)

![eval2](https://github.com/user-attachments/assets/89afa436-4c24-43c6-82d6-74e20432b1e5)

![confmat](https://github.com/user-attachments/assets/feec0448-ff84-409c-bc97-f9ceb9ed0256)




## Key Insights
- Emotion Distribution Varies by Platform: Reddit posts exhibited a higher proportion of genuine mental health-related content, with "sadness" being the most dominant emotion, while Twitter data showed an unexpectedly high presence of "joy," likely due to hashtag misuse and less targeted data collection.

- Content Depth and Length Differ: Reddit users tend to write longer, more detailed posts, providing richer emotional context, while Twitter content is generally shorter and more concise, influencing the quality and variety of emotional expressions captured.

- Cross-Platform Challenges and Generalisation: Models trained on one platform showed limited generalisation when tested on the other, highlighting platform-specific linguistic and emotional patterns. However, the emotion "sadness" demonstrated relatively strong cross-platform detectability.

- Modest Model Performance with Potential for Improvement: While SVM slightly outperformed Naive Bayes across evaluations, overall accuracies remained modest, suggesting a need for more advanced modelling techniques, better handling of class imbalance, or more refined data filtering methods for improved emotion prediction.


## Conclusion and Recommendation
In this project, I explored the potential of cross-platform emotion classification for mental health prediction by analysing user-generated content from Twitter and Reddit. Through a comprehensive pipeline of preprocessing, annotation, TF-IDF vectorisation, and supervised machine learning using SVM and Naive Bayes, the study highlighted notable differences in emotion distribution, content length, and expression styles between the two platforms. While both models demonstrated moderate performance, with SVM slightly outperforming Naive Bayes, the emotion class "sadness" consistently emerged as the most detectable across platforms. These findings emphasise the complexity of cross-platform generalisation and the importance of platform-specific nuances in emotion detection.

The implications of this work are significant for various stakeholders: mental health researchers, professionals, social media platforms, and policymakers. Emotion classification models like those developed here can be used for real-time monitoring and early detection of mental health concerns by identifying emotional shifts on social media. Mental health organisations can leverage such tools to issue early warnings or interventions and integrate them into digital health applications that support users experiencing emotional distress. These tools could offer real-time feedback, suggest coping strategies, or connect users with professional help.

## References

E, N.P. and Juliet, S. (2023) Comparative Analysis of Machine Learning Techniques for Mental Health Prediction. 2023 8th International Conference on Communication and Electronics Systems (ICCES) IEEE.  

MHFA England (2019): Mental health statistics. https://mhfaengland.org/mhfa-centre/impact-report-2019.pdf 
 Accessed 28 May 2024. Mental illness: estimated cases, World, 2021 (https://ourworldindata.org/mental-health) 

WHO (2017) Depression and other common mental disorders: Global health estimates. 
Tech. Rep.,World Health Organization. https://www.who.int/news-room/fact-sheets/detail/mental-disorders Accessed 16 July 2024.  


