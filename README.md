# **Travel Insurance Classification** #


**Created by: Novta Winkey Pradana**

---

# **Business Problem Understanding** #

## **Context :** ##

The global travel insurance industry is experiencing significant growth. In 2021, the global market value was USD 17.8 billion, and analysts forecast that the matter will increase at a compound annual growth rate (CAGR) of 15.4% from [2022 to 2030](https://www.grandviewresearch.com/industry-analysis/travel-insurance-market-report#:~:text=Report%20Overview%20The%20global%20travel,for%20the%20travel%20insurance%20market)​. Tourism demand and technological developments drive this growth, raising awareness of the importance of travel [insurance](https://www.gminsights.com/industry-analysis/travel-insurance-market#:~:text=Travel%20Insurance%20Market%20size%20was,the%20importance%20of%20travel%20insurance). In response to this, travel insurance companies can increase customer satisfaction and company profitability by using customer data to predict and manage [claims](https://www.mckinsey.com/industries/financial-services/our-insights/insurance/elevating-customer-experience-a-win-win-for-insurers-and-customers). 

In the digital era, adapting to the latest technology is the key to exploiting market potential that continues to grow. Innovations like machine learning and AI have changed how travel insurance companies operate, from improving customer service to speeding up the [claims process](http://www.itij.com/latest/long-read/tech-transforming-travel-insurance). In addition, technology also enhances differentiated customer experience, which is very important in attracting and retaining customers in a [competitive market](http://www.itij.com/latest/long-read/tech-transforming-travel-insurance).


Targets:

0: Did not make a claim

1: Make a claim

## **Problem Statement :** ##

<span style="color:yellow">The stakeholders</span> in this matter are <span style="color:yellow">the manager of the risk and claims departement and the marketing manager at the travel insurance company</span>. <span style="color:yellow">The problem is that not all policyholders submit claims</span>, and those who do submit claims may experience varying levels of claim acceptance. The challenge lies in identifying patterns and factors that predict the likelihood of filing a claim. If the company can identify these factors, <span style="color:yellow">it can help insurance companies simplify their risk management processes, set premiums more accurately, and improve targeting and customer service</span> .

## **Goals :** ##

Based on the previously explained problems,<span style="color:yellow"> a model is needed to predict which customers are most likely to submit a claim</span>.
This model allows travel insurance companies to:
1. Adjust their risk assessment and pricing strategy.
2. Identify potential high-risk customers and offer customized travel insurance packages.
3. Improve preventive customer service for tourists who may need assistance.

## **Analytic Approach :** ##

To solve stakeholder problems in the travel insurance industry, I will analyze data to find patterns from existing features and <span style="color:yellow">develop a claims prediction model using machine learning techniques</span>. This model will process and analyze historical customer data to identify ways to influence the likelihood of a claim filing. Once the model has been developed and tested, <span style="color:yellow">risk and claims management departments can use it to adjust risk assessment and premium pricing. Furthermore, marketing and sales teams can leverage the results of this analysis to target high-risk customers with tailored products. In contrast, customer service teams can proactively offer support and preventative services to customers who need help</span>. This approach aims to enhance risk management efficiency and accuracy and improve customer satisfaction and retention.

## **Metrix Evaluation :** ##

<p align="center">
  <img src="https://drive.google.com/uc?id=1y6a7WzH5c5C6uSn2diVAMQ0iiC8EayDV" alt="Gambar dari Google Drive" style="display: block; margin: 0 auto;">
</p>

- TP: The model predicts prospective travelers who make claims to make claims.
- TN: The model predicts that prospective travelers who do not make a claim will not make a claim.
- FP: The model predicts that prospective travelers who do not make a claim will make a claim.
- FN: The model predicts that prospective travelers who make claims will not make claims.

There are two types of errors:
- Error type 1 (FP):
Consequences: Waste marketing costs and company resources.
- Error type 2 (FN):
Consequences: Loss of potential candidates.

However, further analysis is needed by assessing the costs of the errors caused to determine priority treatment for these two types of errors. Using the annual report at PT Asuransi Harta Aman Pratama Tbk for [2022](https://asuransi-harta.co.id/wp-content/uploads/2023/05/AnnualReport2022.pdf), the company incurred claims expenses of 153.63 billion rupiah and operating expenses of 126.07 billion. Assuming equal value for each insurance product the company offers, we will divide these amounts into seven equal parts. Since travel insurance at the company forms a part of a bundle that includes seven other insurances, we will distribute it equally among these seven parts. Below is the complete calculation:

1. Calculate the Value of Business Expenses and Travel Insurance Claim Value:
- Total claims expenses in 2022: 153.63 billion rupiah
- Total operating expenses in 2022: 126.07 billion rupiah
- Value per insurance for claims: 153.63 billion/ 7 = 21.947 billion rupiah
- Value per insurance for operating expenses: 126.07 billion/ 7 = 18.009 billion rupiah
- For travel insurance: (21.947 billion/7) = <span style="color:yellow">3.135 billion rupiah</span>
- For operating expenses: (18.009 billion/7) = <span style="color:yellow">2.573 billion rupiah</span>

2. Estimated cost of false positives (FP):
- Direct Line Travel Insurance in the UK reports that claimants completely fabricate around [5%](http://www.itij.com/latest/long-read/how-travel-insurers-are-tackling-fraud) of travel insurance claims. Analysts use this figure to estimate the proportion of FP.
So, the estimated FP costs are as follows:
  - <span style="color:yellow">Total costs FP</span> = 5% of the value of travel insurance business expenses = 5% * 2.573 billion = <span style="color:yellow">128.65 million rupiah</span>.

3. Estimated cost of false negatives (FN):
- There needs to be detailed information regarding the percentage of travel insurance companies making mistakes in estimates for potential consumer assessments. The investigation reveals an [11%](https://www.finder.com.au/do-travel-travel-insurance-companies-pay-claims#:~:text=Nearly%2090,every%20other%20category%20of%20insurance) rejection rate for travel insurance claims. This rate serves as a basis to assume the proportion of FN, indicating the general percentage of rejected travel insurance claims.
So, the estimated FN costs are as follows:
     - <span style="color:yellow"> Total cost of FN</span> = 11% of travel insurance value = 11% * 3.135 billion rupiah =<span style="color:yellow"> 344.85 million rupiah</span>.

Based on an assessment of the error costs incurred, <span style="color:yellow">the result is that the FN value is greater than the FP</span>, so as much as possible, I will create a model that can reduce the FN costs so as not to lose potential candidates. According to [(Murphy, 2012)](http://noiselab.ucsd.edu/ECE228/Murphy_Machine_Learning.pdf) recall (sensitivity) becomes a very relevant metric because recall will measure the fraction of actual positive cases identified by the model, which is especially important for larger FNs than FPs.<span style="color:yellow"> So, based on considerations, the recall metric will be chosen as the primary metric</span>.

## **Project Limitation :** ##


One of the limitations of this project arises from the inherent challenge of handling an imbalanced dataset, which complicates the task of optimizing the trade-off between recall and precision in our classification model.

# **Data Understanding** #

The problems explained in understanding business problems will be analyzed using the Travel Insurance dataset. The dataset contains historical data on travel insurance users who submitted claims and who didn't.

Dataset source : [Link](https://drive.google.com/drive/folders/1iVx5k6tWglqfHb05o0DElg8JHg7VVG_J)

Note:
- Unbalanced dataset
- There are categorical and numerical features
- Each row of data represents customer travel insurance information

### **Attributes Information**

| **Attribute**                | **Data Type** | **Description**                                           |
|----------------------------|---------------|---------------------------------------------------------|
| Agency              | Object        | Name of agency     |
| Agency Type         | Object        | Type of travel insurance agencies |
| Distribution Channel          | Object        | Channel of travel insurance agencies |
| Product Name    | Object        | Name of the travel insurance products                  |
| Gender             | Object        | Gender of insured                           |
| Duration           | Int64        | Duration of travel                            |
| Destination      | Object         | Destination of travel            |
| Net Sales      | Float64       | Amount of sales of travel insurance policies            |
| Commission (in value) | Float64         | Commission received for travel insurance agency                 |
| Age      | Int64         | Age of insured                      |
| **Claim**  | **Object**         | **Claim status (Target)**                    |


Based on the research conducted by [Leggat and Leggat in 2022](https://pubmed.ncbi.nlm.nih.gov/12044271/), as detailed in their study, ten attributes significantly influence travel insurance claims:

1. Medical and dental conditions
2. Age
3. Type of claim
4. Etiology of the claim
5. Utilization of emergency services
6. Location
7. Sex and occupation
8. Duration and purpose of the trip
9. Level of insurance coverage
10. A general category of the claim

However, the study identifies 'sex and occupation' as having less impact on insurance claims. My dataset encompasses several of these influential attributes identified by Leggat and Leggat. I will set limitations for the aspects not present in my dataset. In line with the study's findings, I have decided not to use the gender attribute in my analysis, considering its minimal influence on claims. I will continue using other features outside Leggat and Leggat's research in my analysis.

## Load Library

## Load Dataset

### **Convert target into numeric**

Before start data skimming, I will convert the target column into numeric.

## **Skimming Data**

In the first phase of data skimming, we conducted a preliminary examination of the travel insurance dataset. This process included reviewing the first five rows and assessing the attribute information. Our primary objectives were to understand the scope of the data, decipher the descriptions of each attribute, understand what each row represents, and discern the relevance of these attributes in the business context.

Insight from data skimming:

- **Dataset Preview**:
  - The data includes a variety of columns such as 'Agency', 'Agency Type', 'Distribution Channel', 'Product Name', 'Gender', 'Duration', 'Destination', 'Net Sales', 'Commision (in value)', 'Age', and 'Claim'.
  - There are missing values in the 'Gender' column, with approximately 71% of data unavailable.

- **Dataset Size**:
  - The dataset contains 35,462 rows and 11 columns, indicating its size and complexity.

- **Data Information**:
  - Columns 'Agency', 'Agency Type', 'Distribution Channel', 'Product Name', 'Gender', 'Destination', and 'Claim' are categorical data.
  - 'Duration', 'Net Sales', 'Commision (in value)', and 'Age' are numerical data.
  - The highest proportion of missing values is in the 'Gender' column.
  - There is considerable variation in the number of unique values for each feature.
  - No significant negative values are found in numerical data, but some numerical columns have a substantial percentage of zero values.
  - Examples of unique values for each feature provide an initial overview of the variation in the data.

- **Descriptive Statistics**:
  - Statistics for numerical variables show the range of data, mean, standard deviation, and minimum and maximum values.
  - 'Duration' has a wide range with a maximum value of 4,881 days, while 'Age' has a maximum value of 118 years, which may indicate the presence of outliers or data entry errors.
  - Categorical columns show the number of unique categories, the frequency of the most common type, and the total count of each kind.


After thoroughly skimming through the data, I've prepared a cohesive summary that bridges the dataset's attributes with their relevance in the business context and an overview of the diverse values encountered within the dataset. 

**Business Context and Attribute Relevance:**

1. **Sales Mechanisms**: The attributes 'Agency', 'Agency Type', and 'Distribution Channel' shed light on the various strategies and channels used for selling insurance policies.
2. **Product Details and Financials**: 'Product Name', 'Duration', 'Destination', 'Net Sales', and 'Commission (in value)' provide a deep dive into the specifics of the insurance products offered and their financial aspects.
3. **Customer Demographics**: Understanding the demographics of customers is facilitated by the 'Gender' and 'Age' attributes.
4. **Claim Outcomes**: The 'Claim' attribute indicates whether a policyholder has made an insurance claim.

**Dataset Values Analysis:**

1. **Variety in Categories**: Attributes like 'Agency', 'Product Name', and 'Destination' are rich with unique categories, showcasing a diverse range of options and choices.
2. **Numerical Ranges**: 'Duration', 'Net Sales', 'Commission', and 'Age' span various numerical ranges, indicating different customer choices and policy details.
3. **Gender Insights**: The 'Gender' category, despite having missing values, adds a layer of demographic insight.
4. **Claim Decisions**: The 'Claim' category is binary, consisting of 'Yes' or 'No', directly reflecting the outcome of insurance policies.

In summary, this dataset is comprehensive and relevant for delving into queries related to travel insurance claims. Its completeness and depth of information make it a viable candidate for further detailed analysis.


# **Exploratory Data Analysis**

Before we delve into building predictive models, it is crucial to understand the distribution of our numeric features. This step is part of our exploratory data analysis (EDA), which will help us uncover patterns, detect outliers, and test assumptions about the data. We'll start by visualizing the distribution of numeric variables using histograms and box plots, essential tools in [EDA](https://www.codecademy.com/article/eda-data-visualization#) that provide insights into the data's minimum and maximum values, central location, spread, and unusual patterns like skewness or multimodality​​. Moreover, we will perform the Anderson-Darling test to assess the normality of our distributions statistically. This test helps determine if the data is close enough to a normal distribution for applying statistical tools, a crucial step in validating many of the [assumptions underlying predictive modeling](https://www.isixsigma.com/dictionary/anderson-darling-normality-test/).

### **Numerical Feature** ###

#### **Data Distribution**

##### **Distribution of Numerical Data With Boxplot**

Insight : 
| Feature              | Skewness | Skewness Description                                   | Box Plot Findings                                       |
|----------------------|----------|--------------------------------------------------------|---------------------------------------------------------|
| Duration             | 24.63    | Highly right-skewed with several extremely high values. | Many high outliers, low median and quartiles.           |
| Net Sales            | 3.32     | Right-skewed with some high values.                     | Many high outliers, low median.                         |
| Commission (in value)| 4.04     | Right-skewed, similar to Net Sales.                     | Many high outliers, low median.                         |
| Age                  | 2.98     | Slightly right-skewed.                                  | Some high outliers, more evenly distributed.            |


##### **Distribution of Numerical Data With Histogram**

Insight :

| Feature              | Description of Distribution                         | Insights from Histogram                               |
|----------------------|-----------------------------------------------------|-------------------------------------------------------|
| Duration             | Concentrated at lower values with extreme highs.    | Majority of policies have short durations. Significant presence of outliers with very long durations. |
| Net Sales            | Concentrated at lower values with some extreme highs.| Most sales are moderate, but there are outliers indicating very high sales in a few cases. |
| Commission (in value)| Mostly lower values with some extreme highs.        | Majority of transactions yield low commissions, but there are outliers with exceptionally high commissions. |
| Age                  | More evenly distributed with some high outliers.    | Age distribution is relatively even, but includes some outliers at higher ages. |


##### **Distribution of Numerical Data With The Anderson-Darling Test**

| Feature              | Insight From Anderson-Darling Test                                                                                          |
|----------------------|---------------------------------------------------------------------------------------------------|
| Duration             | The duration is heavily skewed, likely due to extreme outliers. Care should be taken in handling these for analysis. |
| Net Sales            | Sales figures are varied, with some transactions significantly higher than most, indicating potential outliers or heavy-tailed distribution. |
| Commission (in value)| The commission has a high variation with some extreme values, suggesting a need for robust statistical methods that do not assume normality. |
| Age                  | Age distribution is not normal, possibly due to high-age outliers. This could affect modeling if age is a predictor. |


#### **Data Correlation**

##### **Point-Biserial Correlation**

After establishing a foundational understanding of our data distributions, we progressed to exploring the relationships between the numeric features and the binary 'Claim' target variable using point-biserial correlation.The point-biserial correlation assesses relationships between continuous variables (like 'Age', 'Duration', and 'Net Sales') and the binary 'Claim' variable. This method is chosen for its effectiveness in quantifying the association between continuous and dichotomous variables, offering a clear range (-1 to 1) for interpreting the correlation's strength and direction. Its application is ideal for datasets with naturally dichotomous outcomes, making it a relevant and insightful tool in our exploratory data analysis ([Laerd Statistics](https://statistics.laerd.com/spss-tutorials/point-biserial-correlation-using-spss-statistics.php), [Statistics Solutions](https://www.statisticssolutions.com/point-biserial-correlation-coefficient/)).


| Feature              | Correlation | Interpretation |
|----------------------|-------------|----------------|
| Net Sales            | 0.135       | There is a weak positive relationship between 'Net Sales' and the likelihood of a claim. As the net sales increase, there is a slightly higher probability of a claim being made, suggesting that higher-value policies might be more prone to claims. |
| Commission (in value)| 0.102      | Shows a weak positive correlation. Policies with higher commissions might be associated with a somewhat higher probability of a claim, indicating that policies providing more incentive to agents might be more prone to claims. |
| Duration             | 0.065      | A very weak positive relationship. Longer travel duration has a marginally higher association with the likelihood of a claim, possibly due to increased exposure or risk on longer trips. |
| Age                  | -0.012     | A very weak negative correlation, almost negligible. It suggests that as age increases, the likelihood of a claim slightly decreases, but this relationship might not be practically significant. |

**Comprehensive Insights Corellation between Claim and Numerical Features :**
- **Net Sales** and **Commission** show the most notable positive correlations with claims, though still weak. These variables could be significant in risk management strategies.
- **Duration** shows a fragile positive relationship, indicating it may not be a substantial claim predictor.
- **Age** has a negligible negative correlation, suggesting it might not be a significant factor in predicting claims.

Employing point-biserial correlation, we quantitatively assessed how each numeric feature correlates with the likelihood of a claim. This analysis offered preliminary insights into which features might be more influential in predicting claims. The next step is to study bivariate analysis based on the insights gained from correlation analysis.

#### **Bivariate Correlation**

Insight: 

1. From Boxplot: 
    -   Duration: If the median duration is higher in the 'Claim' group, it suggests longer trips are more likely to have claims.
    -   Net Sales: Higher median net sales in the 'Claim' group could indicate higher-value policies are more prone to claims.
    -   Commission (in value): A higher median commission in the 'Claim' group might imply that policies with higher commissions see more claims.
    -   Age: A notable difference in median age between groups could suggest an age-related trend in filing claims

2. From Mann-Whitney Test:
    -   Duration: Insurance coverage duration shows a significant but less pronounced difference between the groups compared to Commission and Net Sales.
    -   Net Sales: Net Sales also exhibits a significant difference in distribution between the groups, suggesting it's closely related to claim occurrences.
    -   Commission (in value): The distribution of 'Commission (in value)' shows the most significant difference between the Claim and No Claim groups, indicating a strong association with the likelihood of a claim.
    -   Age: Age shows the least significant difference among the tested features but still indicates some association with claim status.

Overall Insight from Bivariate Correlation: 

The boxplots and Mann-Whitney U test results together suggest that features like duration, net sales, and commission have a significant association with the likelihood of a claim being made. These insights can be instrumental for risk assessment and policy pricing strategies in travel insurance.

After completing our bivariate correlation analysis, where we examined the relationships between numeric features and the 'Claim' target variable, we now explore our dataset's categorical dimensions.

### **Categorical Feature** ###

First, start by evaluating how often each category appears in the dataset. This evaluation provides an idea of the frequency and distribution of classes. Next, test the relationship between categorical features and targets with the Chi-square test, which is the right choice for categorical data. Finally, bivariate analysis with cross-tabulation provides a deeper understanding of the relationship between categorical and target feature pairs.

##### **Distribution of Categorical Data With Barplot**

**Travel and Insurance Market Insights**

1. **Top 10 Agencies and Destinations:**
    - The agency with the code "EPX" dominates in transaction frequency, indicating a solid market presence or preferred status among consumers.
    - Singapore emerges as the leading destination, which could reflect a higher demand for travel services or insurance products in that region, suggesting potential market expansion opportunities.

2. **Top 10 Product Names and Agency Type Distribution:**
    - The "Cancellation Plan" stands out as the most frequently offered insurance product, which may signal a consumer preference for flexible travel options and the potential for developing similar products.
    - Travel agencies account for a larger share of transactions than airlines, highlighting the importance of partnering with these agencies to capitalize on their market reach.

3. **Distribution Channel and Gender Demographics:**
    - Online distribution channels are the primary source of transactions, underscoring the necessity for a robust digital platform and online marketing strategy to capture this significant market segment.
    - The gender distribution leans towards female customers, which might influence marketing strategies and product design to cater to this demographic's preferences and needs.

**Summary Insight**
The dominance of a single agency and the prominence of Singapore as a travel destination suggest concentrated areas of consumer interest. The preference for cancellation insurance products indicates a market demand for flexible travel solutions. The significant transaction volume through travel agencies means these entities play a critical role in customer acquisition and service distribution. The predominance of online bookings suggests a shift toward digital consumption patterns, and the slight female majority of customers provides a demographic focus for targeted marketing efforts. These insights can guide strategic decisions in product development, marketing, and partnerships within the industry.

#### **Data Correlation**

##### **Chi Square Test**

The [Chi-square test](https://pubmed.ncbi.nlm.nih.gov/28295394/#:~:text=URL%3A%20https%3A%2F%2Fpubmed) is a robust statistical method for determining if there is a significant association between two categorical variables. It is particularly relevant for our dataset because it can test whether the distribution of claims across different categories, like insurance agencies or plan types, is due to chance or signifies a natural correlation. This test is appropriate when the data do not require a normal distribution assumption and when variables are nominal or ordinal, as with our categorical data.​

Insight from chi-square test results:

| Feature               | Insight |
|----------------------|---------|
| Agency               | There is a highly significant relationship between the agency selling the insurance policy and the likelihood of a claim. Specific institutional settings correlate with a propensity for higher or lower claims. |
| Agency Type          | There is a significant relationship between the type of agency and claims, indicating that the likelihood of filing a claim differs between travel agencies and airlines. |
| Distribution Channel | There is no significant relationship between the distribution channel and claims. The method of purchasing insurance (online or offline) does not significantly influence the possibility of a claim. |
| Product Name         | There is a highly significant relationship between the insurance product name and claims, suggesting that certain insurance products are significantly more or less likely to be associated with claims. |
| Gender               | There is a statistically significant, though weak, relationship between the policyholder's gender and claims, which may suggest that gender could influence claim behavior. |
| Destination          | The significant relationship between travel destination and claims indicates that the destination of travel strongly influences the likelihood of a claim occurring. |



Based on the chi-square test results, we can draw some key conclusions and identify areas for further analysis. The test reveals strong correlations between factors like agency, agency type, product name, and destination with the incidence of travel insurance claims. In contrast, the distribution channel does not correlate significantly with claims. Gender, while showing a statistically significant correlation, presents a weaker relationship, suggesting nuances that merit deeper investigation.

The next logical step is to conduct a bivariate analysis using cross-tabulation to bridge this analysis and gain a more comprehensive understanding. This approach will allow us to examine the interactions between pairs of categorical features and the target variable (claims) in more detail.

#### **Bivariate Correlation**

##### **Cross Tabulation**

[Cross tabulation](https://www.alchemer.com/resources/blog/cross-tabulation/#) is a valuable technique for bivariate correlation analysis as it helps to clarify the relationship between two categorical variables, making complex data sets more manageable and insights more discernible. This approach is particularly useful in market research or survey analysis, where it can reveal patterns and connections that inform strategic decisions.

| Feature | Insight |
| ------ | ------- |
| Insurance Agency | Some agencies like C2B have higher claims, indicating possibly broader coverage or riskier customer profiles. |
| Agency Type | Airlines see more claims than Travel Agency, suggesting riskier travel or a higher propensity to claim. |
| Distribution Channel | More claims are filed Online, possibly due to convenience or increased customer awareness. |
| Product Name | Bronze Plan and 2 way Comprehensive Plan have more claims, indicating they may offer more relevant coverage for riskier travels. |
| Gender | Claims frequency is similar across genders, showing no significant difference in travel insurance claims. |
| Travel Destination | Destinations like Australia and Viet Nam have higher claims, hinting at higher risks or a greater likelihood of travelers filing claims. |


Having completed the cross-tabulation as part of this project's exploratory data analysis (EDA), we've gained valuable insights into the relationships between categorical variables and our target variable. The next phase of our project will involve data preprocessing. In this phase, we'll focus on preparing the data for modeling.

# **Data Preparation**

## **Handling Missing Value** ##

The percentage of missing values in the Gender column is 71%. However, to handle these missing values, I dropped the Gender column altogether because gender doesn't significantly impact the claims; as explained earlier, this step will be carried out in the feature engineering phase. Next, I proceeded to check for any duplicated values in the dataset.

## **Handling Data Duplicated**

There are 4667 duplicate rows. I decided to drop the duplicate data and reset the index afterward.

After I dropped 4667 duplicate rows, I reset the index. 39661 rows remaining and no more exact data. After this, I will handle outliers.

## **Handling Outlier**

A two-pronged approach is employed to handle outliers in the dataset: first, conducting descriptive statistical analysis to gain an overview of the central tendencies and dispersion, and second, utilizing boxplot examination to identify and understand the extent of the outliers visually. This comprehensive method ensures a thorough understanding of the data's distribution, aiding in making informed decisions about effectively addressing the outliers.

Based on the results of the boxplot analysis, it is evident that the entire dataset contains outliers. This observation highlights the presence of data points that significantly deviate from the typical range, indicating variability in the dataset that may warrant further investigation or consideration in the context of data analysis and interpretation. After this process, we will proceed to handle outliers in duration.

### **Handling Outliers in Duration**

To handle outliers in the duration column, I will bin the duration column based on the author's preferences by dividing the data based on the time range as follows:

1. 1-4 months = 1-120 days
2. 5-8 months = 121-241 days
3. 9-12 months = 242- 361 days
4. '>12 months = > 361 days

I have done binning in the duration column to handle outliers in duration and drop NaN values that arise due to binning. After this, I will handle outliers in net sales.

### **Handling Outlier in Net Sales**

After checking more deeply using IQR on Net Sales, there were 9.15 outliers. Outliers in business data, like 'Net Sales', often reflect crucial real-world phenomena. Eliminating these can obscure vital business insights, as [Boldosova and Luoto's research](https://www.emerald.com/insight/content/doi/10.1108/MRR-03-2019-0106/full/html) on data interpretation in business highlights. Moreover, removing outliers without careful consideration risks overfitting in predictive models, reducing their effectiveness on new data. So, I decided not to deal with those outliers. After this process, we will proceed to handle outliers in commision.

### **Handling Outlier in Commision (In Value)**

After checking more deeply using IQR on Commision, there were 10.40% outliers. We opting not to handle outliers in 'Commision (in value)' because the outliers in 'Commission (in value)' data can be reasonably justified by prioritizing the accurate representation of business phenomena. These outliers may reflect significant or unusual transactions essential for [understanding business activities](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5548942/#:~:text=Outliers%20significantly%20affect%20the%20process,values%20and%20outliers%20are%20processed). Omitting them could lead to losing crucial insights, distorting the business landscape. After this process, we will proceed to handle outliers in age.

### **Handling Outliers in Age**

I will bin the age column based on the Republic of Indonesia Health Ministerial Decree [number 25 of 2016](http://hukor.kemkes.go.id/uploads/produk_hukum/PMK_No._25_ttg_Rencana_Aksi_Nasional_Kesehatan_Lanjut_Usia_Tahun_2016-2019_.pdf). Here are the criteria:
- Babies 0-1 years old
- Children 2-10 years
- Teenager 11-19 years old
- Adult 20-44 years
- Pre-Senior Age 45-59 years
- Senior aged ≥ 60 years

The 'Age' column in the dataset has been categorized (binned) by the guidelines set out in the Health Regulation of the Republic of Indonesia. This binning process aligns the age data with standardized age groups, facilitating a more structured and relevant analysis that adheres to recognized national health standards. This alignment not only enhances the accuracy of the analysis but also ensures consistency with regulatory frameworks, making the findings more applicable and informative in a health policy context. Apart from that, there has been a drop in NaN caused by binning. After this process, we will proceed to handle data type.

## **Handling Data Type**

**The data types within the dataset have been verified and are appropriately aligned**, ensuring that each variable is correctly formatted for effective analysis. This alignment of data types is crucial as it lays the foundation for accurate and efficient data processing, analysis, and subsequent interpretation of results. It guarantees that the computational operations and analytical methods applied to the dataset will function as intended, thus enhancing the reliability of the analysis. After this process, we will proceed to handle rare label.

## **Handling Rare Label**

Rare labels in a dataset refer to categories within a categorical variable that occur infrequently. These labels represent a small percentage of the data and are distinct from the more commonly occurring categories.

Insight :

Rare labels, such as less common agencies or destinations with very low frequencies, suggest a wide diversity in the data. These rare labels could offer unique insights into specific, less common scenarios or customer behaviors. However, their rarity also means they might not significantly impact broad trends or patterns in the data. This diversity in our dataset can be crucial for detailed analyses, especially when looking for niche trends or unusual ways.

In anomaly detection and data analysis, retaining rare labels in a dataset is supported by research that underscores the complex nature and potential significance of these infrequent occurrences. Studies in anomaly detection reveal that anomalies, although temporally rare, can exhibit a substantial spatial footprint, suggesting varied frequencies among different types of rare events. Further, the importance of comprehensive ground truth and labeling is highlighted, where biases can arise from incomplete or selective labeling. This complexity and variability of rare labels reinforce the rationale for not handling them, as they might [represent critical patterns or insights within the dataset](https://ar5iv.labs.arxiv.org/html/2211.10129), enhancing the depth and accuracy of the analysis​​. After this process, we will proceed to handle cardinality.

## **Handling Cardinality**

Insight :
1. Agency: 15 unique values suggest a moderate variety of agencies involved.
2. Agency Type & Distribution Channel: Both have two unique values, indicating binary classifications within the dataset.
3. Product Name: 26 unique values denote a more comprehensive range of products.
4. Gender: 2 unique values likely refer to male and female categories.
5. Destination: 135 unique values show a high geographical diversity.

These cardinalities imply that simple encoding techniques may suffice for some features. In contrast, others with higher uniqueness, like destinations, agencies, and product names, may need more complex methods to manage dimensionality in machine learning models. After this process, we will proceed to handle collinearity.

## **Handling Collinearity**

Insight:

1. Duration and Net Sales show a moderate positive correlation (approximately 0.376).
2. Net Sales and Commision (in value) have a stronger positive correlation (approximately 0.635).
3. Commision (in value) and Duration also have a positive correlation, though it's weaker (approximately 0.302).
4. Correlations involving Age are weak.

These correlations, especially between Net Sales and Commision (in value), suggest that these variables are somewhat collinear. Therefore, it can be essential to consider when building predictive models, as collinearity can affect the performance and interpretation of specific algorithms. Depending on the modeling approach, we may need to address this through feature engineering or by choosing algorithms less sensitive to collinearity. After this process, we will continue to carry out feature engineering.

## **Feature Engineering**

Feature engineering streamlines the predictive modeling process by refining raw data into a robust dataset. It begins with data splitting to delineate training and testing sets, ensuring an unbiased model evaluation. Encoding then translates categorical variables into a machine-readable numeric format, while scaling adjusts feature ranges for uniformity. New features are created to capture additional insights, and feature selection is applied to remove redundant or irrelevant data, honing the model's focus on the most impactful variables. Together, these steps are foundational in enhancing model accuracy and predictive power.

I will be dropping the Gender column, as previously explained, based on the rationale that gender does not influence claims significantly. In addition, I will also drop the Age and Duration columns since these have been subjected to binning, rendering the original columns unnecessary and redundant. Following this process, I will move on to encoding.

### **Encoding**

By employing these encoding techniques, we can maintain the categorical data's richness while transforming it into a format amenable to machine learning models.

1. Agency will be binary encoded due to its non-ordinal nature and high unique value count.
2. Agency Type is to be one-hot encoded, fitting its few unique, non-ordinal values.
3. Distribution Channel will also receive one-hot encoding, mirroring Agency Type.
4. Product Name, like Agency, will be binary encoded to manage its high cardinality.
5. Destination is set for binary encoding, avoiding the feature bloat of one-hot encoding.
6. Age Group will be ordinally encoded into integers 0-5, reflecting ordered age categories.
7. Duration Category' will be ordinally encoded into integers 0-3, corresponding to travel duration ranges.

Summary Table:

| Feature Name           | Encoding Method   |
|------------------------|-------------------|
| Agency                 | Binary Encoding   |
| Agency Type            | One-Hot Encoding  |
| Distribution Channel   | One-Hot Encoding  |
| Product Name           | Binary Encoding   |
| Destination            | Binary Encoding   |
| Age Group              | Ordinal Encoding  |
| Duration Category      | Ordinal Encoding  |


This encoding approach streamlines categorical features for practical model training. After knowing the encoding method, we will continue to set the pipeline.

### **Setup Pipeline**

Setting up a data processing pipeline is essential in machine learning workflows, ensuring consistent and efficient dataset handling. This structured approach in the pipeline prepares the data effectively for subsequent modeling.

After meticulously establishing the data processing pipeline harmonizing and refining the dataset, the next logical phase is to embark on the modeling experiment. This transition from data preparation to modeling is crucial. It's where the cleaned and structured data can be leveraged to build, test, and evaluate various machine learning models, exploring their capabilities in predicting outcomes with precision and reliability.

## **Modelling Experiment**

## Data Splitting

To enhance the model's predictive accuracy for future datasets with a similar format and to prevent data leakage, it is essential to perform data splitting again before training the model. This step ensures that the model is tested on a fresh subset of data, mirroring real-world scenarios where it encounters unseen data, reinforcing its robustness and reliability.

I have finished splitting the data. After that, I will set up the modeling experiment.

#### **Setup Pycaret**

The first stage in the modeling experiment is to set up py caret, which uses the pipeline that we created previously.

With the PyCaret setup complete, the stage is set for robust experimentation. The data, now primed within PyCaret's environment, awaits the application of various algorithms. After that, determine the best model from benchmark modeling.

### **Benchmarking Models**

[Benchmarking in machine learning](https://www.nature.com/articles/s42254-022-00441-7) is a compass for navigating the vast landscape of algorithms and architectural choices, guiding researchers toward the most effective solutions for their scientific inquiries. In essence, benchmarking crystallizes the capabilities of various models, elucidating their strengths and weaknesses in specific contexts​​. It is the bedrock upon which the reliability and generalizability of machine learning technologies are evaluated, especially as datasets grow in size and complexity.

#### **Benchmark With ADASYN (Adaptive Synthetic Sampling) Method**

[Adaptive synthetic (ADASYN) sampling](https://ieeexplore.ieee.org/document/4633969) helps approximate imbalanced data. The essential idea of ADASYN is to use a weighted distribution for different minority class examples according to their level of difficulty in learning, where more synthetic data is generated for minority class examples that are harder to learn compared to those that are easier to understand. Benchmark models using ADASYN provide valuable insight into the effectiveness of such techniques in creating good classification.

Create a model from the three best ADASYN compare results based on reacall value.

Having identified the three top-performing models based on recall value; Logistic Regression, Ada Boost Classifier and Linear Discriminant Analysis. The next logical step is benchmarking with SMOTE (Synthetic Minority Over-sampling Technique).

#### **Benchmark With Smote Method**

[The Synthetic Minority Over-sampling Technique (SMOTE)](https://arxiv.org/abs/1106.1813) is a widely recognized method for addressing class imbalance in datasets. SMOTE creates synthetic samples from the minority class to promote equal representation. It achieves this by interpolating between existing minority class instances to generate new, synthetic instances. By employing SMOTE during benchmarking, we can assess the impact of a balanced class distribution on the performance of models like Naive Bayes, Logistic Regression, and Ridge Classifier. This insight is pivotal for understanding how each model handles a more balanced dataset and maintains performance metrics, especially for minority class predictions.

Create a model from the three best SMOTE compare results based on recall value.

Having identified the three top-performing models— Quadratic Discriminant Analysis, Logistic Regression, and Ridge Classifier—the next logical step is benchmarking with penalized method.

#### **Benchmark With Penalized and Resampling Method**

[Penalized models]((https://www.sciencedirect.com/science/article/pii/S1053811923004044)) solve class imbalance by adjusting the learning algorithm to prioritize the minority class. This approach modifies the loss function, assigning higher penalties for misclassifying minority class instances. Therefore, the third benchmark was carried out using the penalized method.


Because the recall value is still small due to data imbalance, I will use more weighting on smaller class values. Utilizing a class weight ratio of [0:1 to 1:10](https://gking.harvard.edu/files/0s.pdf) is a strategic approach to mitigate the bias towards the majority class in imbalanced datasets. This particular ratio amplifies the cost of misclassifying the minority class, effectively 'penalizing' the model more for errors made on the less represented class. Based on the research of [Hastie, Tibshirani, and Friedman](https://link.springer.com/book/10.1007/978-0-387-84858-7), I will perform these weights on **Logistic Regression and Random Forest classifiers** because they both allow adjusting the weights during model training to address class imbalance.

Create a model with adjusting class weights.

Having identified the top-performing models based on recall value; Logistic Regression and Random Forest Classifier. The next logical step is comparing the top eight models from the previous benchmark.

#### **Compare the Top Eight Models**

| Model                            | Technique   | Accuracy | AUC    | Recall | Precision | F1    | Kappa  | MCC   | TT (Sec) |
|----------------------------------|-------------|----------|--------|--------|-----------|-------|--------|-------|----------|
| Logistic Regression              | Penalized   | 0.2941   | 0.7775 | 0.9042 | 0.0216    | 0.0422| 0.0089 | 0.0544|    -     |
| Logistic Regression              | ADASYN      | 0.7766   | 0.7838 | 0.7097 | 0.0530    | 0.0986| 0.0688 | 0.1506| 0.2650   |
| Ada Boost Classifier             | ADASYN      | 0.7692   | 0.7881 | 0.7073 | 0.0512    | 0.0955| 0.0655 | 0.1459| 0.3720   |
| Linear Discriminant Analysis     | ADASYN      | 0.7864   | 0.7857 | 0.7072 | 0.0553    | 0.1026| 0.0730 | 0.1555| 0.3130   |
| Logistic Regression              | SMOTE       | 0.7820   | 0.7847 | 0.7050 | 0.0539    | 0.1002| 0.0705 | 0.1521| 0.2850   |
| Ada Boost Classifier             | SMOTE       | 0.7755   | 0.7891 | 0.7027 | 0.0523    | 0.0973| 0.0675 | 0.1479| 0.6510   |
| Ridge Classifier                 | SMOTE       | 0.7940   | 0.0000 | 0.6956 | 0.0564    | 0.1043| 0.0749 | 0.1563| 0.2880   |
| Random Forest Classifier         | -           | 0.8577   | 0.6797 | 0.3674 | 0.0459    | 0.0815| 0.0526 | 0.0881|    -     |

After comparing the top eight models, we can choose the best three models based on recall value, and the best three models are **Logistic Regression with Penalized and ADASYN and Ada Boost Classifier with ADASYN**. The following are the reasons for choosing these three models:

1.  **Logistic Regression with Penalized Technique:**
    It has the highest recall of 0.9042, which best identifies all relevant instances in the dataset. This model is beneficial when the cost of a false negative is high, as it prioritizes minimizing the false negatives.
    However, its accuracy is very low (0.2941), and precision is the weakest of all models (0.0216), which suggests it incorrectly labels many non-relevant instances as relevant.
2.  **Logistic Regression with ADASYN:**
    This model has a good balance between recall (0.7097) and precision (0.0530), which may make it a candidate for a model that doesn't sacrifice accuracy as much for recall.
    The Accuracy (0.7766) and AUC (0.7838) are substantially higher than the penalized logistic regression, making it a more balanced model overall.
3.  **Ada Boost Classifier with ADASYN:**
    The recall (0.7073) is slightly lower than the logistic regression with ADASYN but still relatively high. The precision (0.0512) is similar to the logistic regression with ADASYN.
    It has a higher F1 score (0.0955) than the logistic regression with ADASYN (0.0986), suggesting a slightly better balance between precision and recall. The Accuracy (0.7692) and AUC (0.7881) are also comparable, indicating it is a robust model.


### **Hyperparameter Tuning Three Best Model**

[The critical role of hyperparameter tuning in machine learning](​(https://sparkbyexamples.com/machine-learning/hyperparameter-tuning-in-machine-learning/)), emphasizing its importance in defining model behavior and effectiveness, and the standard methods like grid search, random search, and Bayesian optimization used in this process are well-documented in the broader field of machine learning​​​​​. Because of the critical role of hyperparameters, hyperparameter tuning will be carried out, especially on three models, namely logistic regression with penalized and ADASYN and Ada Boost Classifier with ADASYN. First step for hyperparameter tuning is set the parameter for penalized and ADASYN logistic regression and Ada Boost Classifier, then use the parameter for tuning the model to optimize the Recall value.

### **Model Selection**
After tuning the best three models, we compared the three models to consider which model to use for the final model.

#### **Before VS After tuning**

**Logistic Regression Performance Comparison (Penalized)**

| Model                             | Before/After Tuning | Accuracy | AUC    | Recall | Precision | F1 Score | Kappa | MCC   |
|-----------------------------------|---------------------|----------|--------|--------|-----------|----------|-------|-------|
| Logistic Regression (Penalized)   | Before Tuning       | 0.2941   | 0.7775 | 0.9042 | 0.0216    | 0.0422   | 0.0089| 0.0544|
| Logistic Regression (Penalized)   | After Tuning        | 0.7820   | 0.7847 | 0.7050 | 0.0539    | 0.1002   | 0.0705| 0.1521|

**Insight:**
1. **Accuracy**: The model's accuracy has remarkably increased from 29.41% to 78.20% after tuning. This result represents a more than two-fold improvement, indicating the model's enhanced performance in correctly predicting outcomes.
2. **AUC**: The AUC has seen a slight increase from 0.7775 to 0.7847. This improvement, although marginal, suggests better discriminative ability of the model post-tuning.
3. **Recall**: There has been a noticeable decrease in recall, from 90.42% before tuning to 70.50% after adjusting. This decrease suggests that while being more accurate, the model is now less sensitive in identifying all relevant instances.
4. **Precision**: The precision of the model has increased from 2.16% to 5.39% after tuning. Despite this increase, the accuracy remains relatively low, indicating that the model makes many false-positive predictions.
5. **F1 Score**: After tuning, the F1 Score has improved from 4.22% to 10.02%. This score, which balances precision and recall, indicates that the model's predictive power has increased overall despite a trade-off with recall.
6. **Kappa**: The Kappa statistic has increased from 0.0089 to 0.0705, which suggests a better agreement between the predictions and the actuals beyond what would be expected by random chance.
7. **MCC**: The Matthews correlation coefficient has improved substantially from 0.0544 to 0.1521. That number indicates a better quality of classifications made by the model after tuning.

In conclusion, the tuning process has significantly enhanced the model's accuracy, precision, F1 score, Kappa, and MCC. However, it has also reduced recall, meaning the model is now less capable of detecting all positive cases.


**Logistic Regression with ADASYN Performance Comparison**

| Model | Before/After Tuning | Accuracy | AUC | Recall | Precision | F1 Score | Kappa | MCC |
|-------|---------------------|----------|-----|--------|-----------|----------|-------|-----|
| Logistic Regression (ADASYN) | Before Tuning | 0.7766 | 0.7838 | 0.7097 | 0.0530 | 0.0986 | 0.0688 | 0.1506 |
| Logistic Regression (ADASYN) | After Tuning | 0.7766 | 0.7838 | 0.7097 | 0.0530 | 0.0986 | 0.0688 | 0.1506 |

**Insight:**
1. **Accuracy**: There was no change in the model's accuracy after tuning, which remains at 77.66%. This indicates that the tuning process has not affected the model's ability to predict outcomes in this case correctly.
2. **AUC**: The AUC stays constant at 0.7838, showing that the model's ability to differentiate between the classes has yet to improve with tuning.
3. **Recall**: The recall rate is also unchanged at 70.97% post-tuning. This suggests that the model's sensitivity to detecting positive cases is consistent, and the tuning did not enhance this aspect.
4. **Precision**: Precision remains at 5.30%, indicating no improvement in the ratio of accurate optimistic predictions out of all positive predictions after tuning.
5. **F1 Score**: The F1 Score, which combines precision and recall, stays at 9.86%, meaning the harmonic balance between precision and recall has not been impacted by the tuning.
6. **Kappa**: The Kappa statistic remains at 0.0688, suggesting that the agreement between the predicted and actual outcomes, adjusted for chance, has not improved with tuning.
7. **MCC**: The Matthews correlation coefficient has not changed and is still at 0.1506 after tuning, indicating that the overall quality of the binary classifications remains the same.

In conclusion, the tuning process did not impact the performance metrics of the Logistic Regression model with the ADASYN technique. The model's accuracy, discriminative ability, sensitivity, and overall classification quality remain consistent before and after the tuning process. This could suggest that the model was already well-tuned to the data or that the adjustments made during the tuning process needed to be more significant and make a measurable difference.

**Ada Boost Classifier with ADASYN Performance Comparison**

| Model | Before/After Tuning | Accuracy | AUC | Recall | Precision | F1 Score | Kappa | MCC | TT (Sec) |
|-------|---------------------|----------|-----|--------|-----------|----------|-------|-----|----------|
| Ada Boost Classifier (ADASYN) | Before Tuning | 0.7692 | 0.7881 | 0.7073 | 0.0512 | 0.0955 | 0.0655 | 0.1459 | 0.3720 |
| Ada Boost Classifier (ADASYN) | After Tuning | 0.6000 | 0.7830 | 0.7748 | 0.0337 | 0.0644 | 0.0325 | 0.1005 | - |

**Insight:**
1. **Accuracy**: There is a significant decrease in the model's accuracy from 76.92% before tuning to 60.00% after tuning. This suggests a decline in the model's overall performance in correctly predicting outcomes.
2. **AUC**: The AUC has slightly decreased from 0.7881 to 0.7830, indicating a minor drop in the model's ability to discriminate between the positive and negative classes.
3. **Recall**: The recall has improved from 70.73% to 77.48%. This increase suggests that the model is better at identifying all relevant instances.
4. **Precision**: The precision has decreased from 5.12% to 3.37% post-tuning, indicating a lower ratio of accurate positive predictions out of all optimistic predictions.
5. **F1 Score**: There is a decrease in the F1 Score from 9.55% to 6.44%. This metric's reduction indicates that the tuning process has negatively affected the balance between precision and recall.
6. **Kappa**: The Kappa statistic has decreased from 0.0655 to 0.0325, suggesting a lesser agreement between the predicted and actual outcomes than before tuning.
7. **MCC**: The Matthews correlation coefficient has also decreased, going from 0.1459 to 0.1005, indicating a decline in the quality of binary classifications after tuning.

In conclusion, the tuning process for the Ada Boost Classifier with the ADASYN technique has resulted in a trade-off. While the model's sensitivity to detect positive cases has increased, as seen in the improved recall, this has come at the cost of accuracy, precision, and overall classification effectiveness. Further tuning may be necessary to improve the balance between recall and other performance metrics.

**Summary Insight**:
In conclusion, the tuning processes had varied impacts on the models. While the Logistic Regression model with penalization showed marked improvements, the Logistic Regression with ADASYN remained unchanged, and the Ada Boost Classifier with ADASYN saw a decrease in performance. These outcomes highlight the necessity for a careful and tailored approach to model tuning to ensure an optimal balance between all performance metrics.**I decided to use the original rather than the tuned model**.

### **Determine the Final Model**

| Model                            | Technique   | Accuracy |    AUC | **Recall** | Precision |    F1 | Kappa  |   MCC |
|----------------------------------|-------------|----------|--------|--------|-----------|-------|--------|-------|
| Logistic Regression              | Penalized   |  0.0975  | 0.7999 | **0.9719** |   0.0182  | 0.0357|  0.0020| 0.0258|

Based on the analysis of the provided performance metrics for the **penalized logistic regression model**, it is evident that the model demonstrates an exceptionally high recall of 97.19% before tuning. This result indicates that the model is particularly effective at correctly identifying positive cases, a critical advantage in applications where the cost of missing positives is substantial. Thus, prioritizing recall, this pre-tuned penalized logistic regression model could be deemed the most suitable choice for scenarios where catching every possible positive case outweighs the need for precision or overall accuracy. 

After determining the final model, next step is to evaluate the model.

## **Evaluate Final Model**

### **Learning Curve**

The learning curvve provide for logistic regression show the models peformance in terms of training score and cross validation score as the number of training instance increases. 

**Insight**:

1. **Training Score Decrease with More Data**: Initially, when the number of training instances is low (around 25,000), the training score is relatively high, close to 0.8. This score is typical as the model can fit closely to a smaller dataset. However, as more data is introduced, the training score decreases steadily, reaching closer to 0.6 when the number of training instances approaches 40,000. This pattern suggests that the model starts to generalize as it sees more data; hence, the training data fit could be better. This is a normal behavior, as the model is likely moving from overfitting to a more generalized state.

2. **Cross-Validation Score Increase with More Data**: In contrast, the cross-validation score, which measures how well the model performs on unseen data, increases as more training data is provided. Starting below 0.65, it experiences a significant improvement as the number of training instances grows, converging towards the training score. This convergence indicates that the model is improving at predicting outcomes on data it has not been trained on, showing an improvement in model generalization.

3. **Convergence of Training and Validation Scores**: Towards the end of the curve, the training and cross-validation scores converge, which is a good indicator of model stability. This convergence suggests that adding more data beyond this point significantly improves the model's performance.

4. **Business Implications**: From a business perspective, the key takeaway is that the Logistic Regression model stabilizes and offers more reliable predictions as more data is fed into it. The initial high training score indicates potential overfitting, which would not be ideal for making predictions in a real-world setting. As more data is used for training, the model's generalization ability is improving, which is crucial for making robust decisions based on the model's outputs.

5. **Decision-Making**: For decision-makers, this curve reinforces the importance of using a sufficiently large and representative dataset to train the model. The fact that the validation score improves with more data suggests that investing in data collection and preprocessing could yield a model that performs better in practical applications.

**In summary**, the learning curve indicates a successful transition from a possibly overfitted model to a more general and reliable one as more data is introduced. This transition is crucial for deploying the model in a real-world business environment where it needs to make accurate predictions on new, unseen data.

### **ROC Curves**

**Insight**:
1. **OC Curve Interpretation**: The ROC curve is a graphical representation that illustrates the diagnostic ability of a binary classifier system as its discrimination threshold is varied. It is created by plotting the True Positive Rate (TPR) against the False Positive Rate (FPR) at various threshold settings.

2. **Class-Specific AUC**: The curves represent two classes, 0 and 1, with an Area Under the Curve (AUC) of 0.78. The AUC measures the model's ability to distinguish between the two classes. An AUC of 0.78 is generally considered reasonable and suggests the model has a 78% chance of correctly distinguishing between positive and negative types.

3. **Micro-Average ROC**: The micro-average ROC curve, with an AUC of 0.14, is considerably lower than the class-specific AUCs. This result might suggest an imbalance in the dataset or an issue with calculating the micro-average. A micro-average AUC should be comparable to the individual class AUCs, especially in a balanced dataset.

4. **Macro-Average ROC**: The macro-average ROC curve, which gives equal weight to each class, also has an AUC of 0.78, aligning with the individual class AUCs. This result indicates consistent model performance across both types when evaluated independently.

5. **Business Implications**: An AUC of 0.78 for the Logistic Regression model indicates that the model has a relatively high level of predictive accuracy and can effectively differentiate between positive and negative cases. In business terms, this translates to a reliable model that can be used to make informed decisions, whether it's for customer segmentation, risk assessment, or another binary classification task.

6. **Model Performance**: The fact that both classes have the same AUC value suggests that the model performs equally well for both types. This is beneficial when looking to maintain a balance in predictive performance across different segments of data.

7. **Decision Thresholds**: The ROC curve can also be used to select an appropriate threshold for classification. The point on the curve closest to the top-left corner represents the optimal trade-off between sensitivity (actual positive rate) and specificity (1 - false positive rate) for the model.

8. **Considerations for Deployment**: Before deploying such a model in a business environment, it is crucial to consider the cost of false positives and false negatives. The threshold should be set accordingly to minimize the more costly type of error.

**In summary**, the Logistic Regression model depicted by the ROC curves suggests a capable predictive model for business applications, balancing sensitivity and specificity. Considering the need to prioritize recall, the model is deemed suitable for this business need, provided it maintains a high recall value. The business should also be prepared to handle the increased workload with more false positives due to lower precision. 

In a business context where the primary concern is to minimize false negatives, such as in fraud detection, medical diagnostics, or any other domain where failing to detect true positives (actual cases) carries a high cost or risk, the recall (also known as sensitivity) becomes the main focus. 

**Insight**:

1. **High Recall Priority**: The PR curve shows that the model can achieve high recall, which means it can identify the most positive instances. However, the precision is low, suggesting that while the model is good at catching true positives, it does so at the expense of misclassifying some negative instances as positive.

2. **Average Precision**: The average precision across all thresholds is 0.07. While this number is low, it is less critical in scenarios where recall is prioritized over precision. The model is designed to minimize the risk of false negatives, which is more important in your business case.

3. **Business Strategy**: The model's ability to capture as many true positives as possible is beneficial for businesses that prioritize recall. It means the model is erring on the side of caution, preferring to flag potential issues for review rather than missing them.

4. **Cost of False Negatives**: If a false negative (missing a confirmed positive case) is high, it may justify accepting more false positives. This trade-off is acceptable when the consequences of missing an actual positive are severe.

5. **Further Model Use**: In practice, the model could be used as a first step in a multi-stage process. For example, it could flag potential cases for further investigation or a second, more precise level of analysis. This way, the business ensures that no critical issues are overlooked.

6. **Decision Thresholds**: The PR curve can help determine an appropriate threshold that maximizes recall while keeping precision acceptable for your business needs. This threshold calibration is crucial to align the model's predictions with business objectives.

**In summary**, based on the provided Precision-Recall Curve and the high-recall requirement of the business context, the model is deemed appropriate for the initial identification of positive cases. However, it is recommended to adjust the decision threshold to balance recall with an acceptable precision level and consider additional review processes or secondary models to refine the predictions and manage the false positives. With these considerations, the Logistic Regression model can be a valuable component of decision-making in scenarios where missing a confirmed positive case is not an option.

### **Time Complexity**

The time taken to train the model is 14.39 seconds, which is relatively fast. However, the training speed does not translate into high predictive accuracy, given the metrics. For a business application, improving these metrics by collecting more data, feature engineering, or trying different models to ensure that the model can reliably inform decision-making processes is critical.

### **Validation**

#### **Prediction on Seen Data**

## **Prediction on Unseen Data**

**Insight from**  : 

The two images reflect the performance metrics of the logistic regression model on seen data (training data) and unseen data (test or validation data), respectively. Here is a business interpretation that takes into account this distinction:

**Performance on Seen Data** :

- **Accuracy (19.95%)**: The model's ability to correctly predict outcomes on the data it was trained on is low, which could indicate the model is not capturing the underlying patterns effectively for this dataset.
- **AUC (0.7924)**: A pretty good score, suggesting that the model can rank the positive class higher than the negative class with reasonable confidence when dealing with familiar data.
- **Recall (94.57%)**: Very high recall indicates that the model is quite effective at identifying true positives in the training set, which is favorable for scenarios like fraud detection where failing to flag fraudulent activities can be costly.
- **Precision (2.00%)**: Low precision means the model generates many false positives. This could translate to unnecessary follow-ups in a business setting, which can be resource-intensive.
- **F1 Score (0.0391)**, **Kappa (0.0056)**, **MCC (0.0443)**: These metrics indicate that the model's ability to balance the trade-off between precision and recall is not strong, and its predictive performance is hardly better than random chance on the training data.

**Performance on Unseen Data**:

- **Accuracy (19.81%)**: Similar to the training data, suggesting consistent but low predictive performance across new, unseen data.
- **AUC (0.8250)**: This increased AUC score on unseen data suggests that the model can distinguish between classes even when encountering new examples.
- **Recall (96.27%)**: The slight increase in recall for unseen data is impressive, as it indicates that the model can generalize its ability to identify true positives to new data. This is crucial for businesses where the cost of a false negative is very high.
- **Precision (2.03%)**: A minimal increase in precision indicates a slight improvement in the model's ability to identify true positives out of all optimistic predictions correctly, but the number of false positives remains high.
- **F1 Score (0.0398)**, **Kappa (0.0062)**, **MCC (0.0498)**: These metrics show that the model's performance is marginally better on unseen data, but it still does not demonstrate a solid predictive ability.

**Business Implications of the Comparison**:

The comparison shows that the model's ability to recall true positives slightly improves when applied to unseen data, which is a positive sign of generalization. However, since the accuracy remains low and the precision is very low on both seen and unseen data, the model would only be considered reliable for most business applications with further improvements. 

In a business context where recall is the main focus, the model may still be helpful, especially if it can be paired with cost-effective strategies for handling the high volume of false positives it generates. For example, a tiered approach to customer follow-up could be implemented, where flagged cases by the model are first subjected to a low-cost automated review before escalating to more costly manual review processes.

The key takeaway for a business focused on recall is to ensure that while the model is good at catching true positives, systems must be in place to deal with the high number of false positives efficiently. This might involve additional screening steps before taking any significant action based on the model's predictions. The business must carefully weigh the operational costs of handling false positives against the benefits of high recall.


## **Confusion Matrix**:

1. True Positives (TP): The top-right cell, with the number 6215, represents the cases where the model correctly predicted the positive class.
2. True Negatives (TN): The top-left cell, with the number 1408, shows the cases where the model correctly predicted the negative class.
3. False Positives (FP): The bottom-left cell, with the number 5, represents the instances where the model incorrectly predicted the positive class.
4. False Negatives (FN): The bottom-right cell, with the number 129, shows the cases where the model incorrectly predicted the negative class.

Summary Insight:
The model has a high number of true positives and true negatives, which suggests it performs well in identifying both classes.
The number of false negatives is relatively low, which is good as it means the model often needs to identify the positive class.
The false positives are very low, indicating that the model rarely misclassifies the negative class as positive.
Overall, the high values of TP and TN and the low values of FP and FN suggest the **Logistic Regression model has a good predictive performance**.

**Calculate potential benefit using the model**

From the confusion matrix we can calculate the total estimated FP and FN cost.


1. **Assign Costs to False Predictions**
    - Cost per FP (False Positive) case is calculated from the total estimated FP cost of 128.65 million rupiah for 5 FP cases, which is approximately <span style="color:yellow">25.73 million rupiah per FP case</span>.
    - Cost per FN (False Negative) case is calculated from the total estimated FN cost of 344.85 million rupiah for 129 FN cases, which is approximately <span style="color:yellow">2.67 million rupiah per FN case</span>.

2. **Calculate Potential Costs Without the Model**
    - The potential FN cost without using the model = Cost per FN case * (TP + FN)
= 2.67 million rupiah per case * (6215 + 129 cases) = <span style="color:yellow">approximately 16.959 billion rupiah</span>.

3. **Calculate Actual Cost With the Model**
    - Actual FN cost with the model = Cost per FN case * FN
    = 2.67 million rupiah per case * 129 cases = <span style="color:yellow">344.85 million rupiah</span>.

4. **Determine Potential Saving**
    - Savings = Potential FN cost without the model - Actual FN cost with the model
1   = 6.959 billion rupiah (without model) - 344.85 million rupiah (with model) = <span style="color:yellow">approximately 16.614 billion rupiah</span>.

5. **Potential Saving in Percentage**
    (16.614 billion rupiah / 16.959 billion rupiah) * 100 = <span style="color:yellow">97.97%</span>

**Conclusion**

The potential savings by using the logistic regression model is approximately <span style="color:yellow">16.614 billion rupiah</span>.This represents a saving of about <span style="color:yellow">97.97%</span> compared to not using the model at all.

The use of the logistic regression model significantly reduces the costs associated with false negatives by identifying true positives accurately. This substantial saving highlights the financial benefit of implementing such a predictive model in insurance claim predictions.The use of the logistic regression model significantly reduces the costs associated with false negatives by identifying true positives accurately. This substantial saving highlights the financial benefit of implementing such a predictive model in insurance claim predictions.

## **Model Interpretation**

### **How the model work?**

Logistic regression is a predictive analysis used for binary classification, which is ideal for dichotomous outcomes (yes/no, true/false). It estimates the probability that a given input point belongs to a particular class. The core of logistic regression is the logistic function, which is used to model the odds of the positive course and convert these odds into a probability using the sigmoid function. 

The model calculates the probability that a particular instance falls into the positive class. This can be crucial for decision-making processes like credit scoring, medical diagnosis, or any scenario requiring a yes/no decision. Logistic regression shines when the relationship between the input variables and the output is approximately linear, and the problem is relatively straightforward.

However, logistic regression has its constraints. It assumes a linear boundary between classes, which is only sometimes valid in complex real-world data. The independence assumption may not hold in data with inherent correlations, like time series. Multicollinearity can also skew the model's coefficient estimates, leading to unreliable predictions. Logistic regression's sensitivity to outliers and its limited capacity to express feature importance further constrict its effectiveness. When data exhibits complex relationships and interactions, logistic regression might fall short, necessitating more sophisticated non-linear models or advanced data preprocessing to ensure its applicability.

### **Feature Importance**

Feature Importance Plot is commonly used in machine learning to visualize the importance of different variables in a predictive model. The vertical axis (y-axis) lists various features used in the model, while the horizontal axis (x-axis) shows the "Variable Importance" of each part. Attributes are ranked from bottom to top, with "Product Name_0" at the bottom, suggesting it has minor importance, and "Distribution Channel_2" at the top, indicating it has the highest priority. The points on the plot represent the importance score for each feature, with scores ranging from 0 to above 6.

In a business context, this plot is used to understand the factors that most influence the model's decisions. Here's a detailed interpretation:

1. **Distribution Channel_2**: This feature has the highest importance score, significantly higher than the others, indicating that it is the most influential variable in the model's predictions. In a business sense, this could mean that the second category of distribution channels (perhaps an online platform vs. a physical agency) strongly predicts the outcome (e.g., the likelihood of a customer purchasing a product or service).

2. **Distribution Channel_1**: The second most important feature, although with less than half the importance of Distribution Channel_2, suggests that this distribution channel category also plays a key role but is less decisive than Channel_2.

3. **Agency Type_2**: This suggests that the second type of agency involved in the business process moderately impacts the model's predictions. For example, this could differentiate between in-house sales vs. third-party brokers in insurance or travel businesses.

4. **Agency_2**, **Destination_2**, **Agency_0**, **Agency_3**, **Agency_1**, **Destination_1**, **Product Name_0**: These features have progressively lesser importance scores. While they contribute to the model's decisions, their influence is relatively minor compared to the top features. These could represent various agencies or destinations in a travel or insurance context, suggesting that while they affect customer behavior or outcomes, they are less critical than the distribution channels.

From a business perspective, this model indicates that how products or services are distributed to customers (e.g., via different channels or agencies) is more influential on the outcome than the specific products or destinations themselves. This insight could inform strategic decisions, such as investing more in the most effective distribution channels, training the most influential agencies, or reevaluating how certain products are marketed based on their impact.

In terms of using the numbers to support business decisions, the company could allocate more resources to improve or expand the most influential distribution channels. For instance, if Distribution Channel_2 represents online sales, the company might enhance its digital marketing efforts or improve the online purchasing process. On the other hand, features with less importance could be reviewed to identify potential efficiencies or areas where resources may be reallocated for better returns on investment.


### **Limitation Model**

1. **Imbalance in Dataset**: An imbalanced dataset may compromise the model's ability to predict accurately, which can lead to a bias towards the majority class.
2. **Focus on FN Reduction**: While reducing False Negatives is prioritized due to their higher costs, this focus may lower precision by increasing False Positives.
3. **Dynamic Behavior Patterns**: The model may not adapt quickly to sudden changes in traveler behavior, which can affect its accuracy over time without regular updates.

## Conclusion and Reccomendation

The final model is a logistic regression classifier used for binary classification. It's encapsulated within a pipeline, including data preprocessing steps like encoding categorical variables, scaling numerical data, and cleaning up column names. These preprocessing steps are crucial as they convert raw data into a suitable format for the model to learn from.

The logistic regression model has demonstrated excellent performance, accurately identifying 6215 cases as True Positives and 1408 as True Negatives. This effectiveness is crucial in reducing costs associated with incorrect predictions. The costs arising from 5 False Positives and 129 False Negatives have been meticulously calculated, revealing significant potential savings of approximately **16.614 billion rupiah** due to the implementation of this model, reflecting a cost-saving of 97.97%.

The model's current performance indicates it is doing well in the context of minimizing the more costly FN errors. Any steps taken to improve the model should maintain or improve upon these performance metrics.

The model works well when the following conditions are met:

- High Recall: The model has a high recall (90.02%), indicating it is very good at correctly identifying prospective travelers who will make claims (TP). - his is particularly important for the company because the cost associated with FN (potential travelers who make claims but are predicted not to) is higher than for FP. High recall means the model minimizes the loss of potential claimants.
- Low FP and FN: The absolute numbers of FP and FN are low (5 and 129, respectively), which implies that the model makes very few errors in both predicting claims and non-claims.
- Cost Savings: The model's performance in terms of recall, combined with the low numbers of FP and FN, leads to substantial cost savings. As calculated, the net profit percentage is very high (approximately 97.97%), indicating the model is cost-effective.

**Recommendation**:
To optimize our risk assessment and pricing strategy, we'll focus on these critical areas:

1. **FN Reduction Focus**: Prioritize minimizing False Negatives (FN) due to their higher cost impact than False Positives (FP). This may involve enhanced feature engineering, more comprehensive data collection, or experimenting with alternative algorithms better suited for FN reduction.
   
2. **Model Monitoring & Updating**: Continuously monitor model performance to maintain high recall and low error rates, incorporating new data to keep pace with evolving traveler behaviors and to ensure the model's predictive relevance.

3. **Cost-Benefit Analysis Enhancement**: Conduct thorough cost-benefit analyses, accounting for the financial impact of True Positives (TP) and True Negatives (TN), to fully understand the model's profitability and guide decisions on potential enhancements.

4. **Business Strategy Alignment**: Align model evaluation metrics with core business objectives, emphasizing recall if reducing FN aligns with strategic priorities.

5. **Data Collection Expansion**: Collect more granular data on the financial implications of TP and TN, as well as customer lifetime value, the cost of claim processing, and the influence of customer satisfaction on retention to refine the cost-benefit analysis further.

6. **Robustness Through Cross-Validation**: Implement cross-validation techniques to ensure the model's robustness and generalizability across various data segments.

7. **Customer Segmentation for Targeted Strategies**: Utilize model predictions to segment customers by risk level for more focused marketing and customer service approaches.

8. **Risk Management Improvements**: Leverage model insights for enhanced risk management, identifying claim patterns and implementing strategies to mitigate those risks.

By integrating these strategies with the model's predictive capabilities to segment customers and develop tailored insurance packages, we can proactively manage risk, align with our business strategy, and provide value-added services to tourists, ultimately enhancing our financial standing and customer service quality.

## **Deployment**

Import final model to pickle

