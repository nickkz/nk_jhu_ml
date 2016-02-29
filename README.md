# nk_jhu_ml
John Hopkins Practical Machine Learning

View the latset .docx (currently ml_course_project_v5_20160228.docx) for complete code and output of the predictive model.

The following steps were performed in developing the predictive model: 
 - initial workspace setup, load libraries, model tuning
 - data loading and cleaning
 - exploratory analysis to build intuition of important covariates
 - reduce covariates through correlation analysis
 - use 3 separate common machine learning algorithms (random forest, gradient boosting, linear discriminant analysis) to process data
 - for each algo train model and predict test data. 
 - combine all 3 predictions into a common ensemble (stacked) model
 - use this to determine final prediction model, apply to testing data set

Specific section comments and notes are inline below.  Note: According to quiz results, this method resulted in 16 of 20 correct predictions, which was not a perfect result but good enough for passing. 

You can also view the raw HTML output using the htmlpreview tool e.g.  
http://htmlpreview.github.io/?https://github.com/nickkz/nk_jhu_ml/blob/master/ml_course_project_v3.html
