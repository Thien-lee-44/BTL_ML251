# BTL2 môn Học Máy, mã môn học CO3117 - Học máy với dữ liệu text
## *Group info*

+ **Semester** : 1
+ **Year** : 2025
+ **Group name** : MLP2
+ **Giảng viên hướng dẫn**: TS. Lê Thành Sách
+ **Members** : 
 
    | **MSSV** | **Full Name** | **Email** |
    |:---|:---:|:--:|
    | 2312046 | Bùi Ngọc Minh | minh.buingocbkhoa@hcmut.edu.vn |
    | 2313233 | Lê Trọng Thiện | thien.lee@hcmut.edu.vn |


## **Nội dung**
This BTL applies both the traditional ML pipeline and one modern ML model (BERT) for classification.

## *Setup and run*

Clone the project:

```bash
git clone https://github.com/Thien-lee-44/BTL_ML251.git
cd BTL_ML251/BTL2
```

Then run the collab file located in ```./notebook```, the collab file is set up to be able to run all from the start

## **Dataset**
Dataset includes 10k poems, spread across 5 topics:

+ Nature
+ Art & Sciences
+ Love
+ Relationships
+ Religion

Dữ liệu này được lấy tại [kraggle](https://www.kaggle.com/datasets/djdonpablo/poem-classification-dataset).

## Traditional ML Pipeline
### *EDA*

#### *General info*
The dataset has 10064 entries, with 2 feature collumns: poem and topic

```python
df.info()
```

```
RangeIndex: 10064 entries, 0 to 11913
RangeIndex: 10064 entries, 0 to 10063
Data columns (total 3 columns):
 #   Column      Non-Null Count  Dtype 
---  ------      --------------  ----- 
 0   unnamed:_0  10064 non-null  int64 
 1   poem        10064 non-null  object
 2   topic       10064 non-null  object
dtypes: int64(1), object(2)
```

There are no null entries

```python
duplicate_rows_df = df[df.duplicated()]
print("number of duplicate rows: ", duplicate_rows_df.shape)
```

This code yields no duplicated values.

#### *Data distribution*
```python
topic_percent = df['topic'].value_counts(normalize=True) * 100
```
topic
nature           24.135533
arts&sciences    21.691176
love             20.687599
relationships    17.955087
religion         15.530604

![](./plots/class.png)

Summary:
+ The dataset is fairly evenly distributed
+ Religion is the lowest with nature being the highest up to nearly a quarter
  
Graphing the distribution of poems' word counts: 
```python
df["word_count"] = df["poem"].str.split().str.len()
plt.hist(df["word_count"], bins=10)
plt.title("Distribution of Poem's Lengths (in word count)")
plt.xlabel("Number of words")
plt.ylabel("Frequency")
plt.show()
```
![](./plots/word_distribution.png)

The conclusions are:
+ The data is very balanced.
+ Almost all poems in the dataset is 1-1000 words long.

Graphing word frequencies across the entire dataset:
```python
all_poems = " ".join(df["poem"]).split()
word_counter = Counter(all_poems)
word_freq = pd.DataFrame(
    word_counter.items(),
    columns=["word", "count"]
).sort_values(by="count", ascending=False)

print(word_freq.head(10))

top_n = 50
plt.figure(figsize=(10,6))
plt.bar(word_freq["word"].head(top_n), word_freq["count"].head(top_n))
plt.title(f"Top {top_n} Most Frequent Words")
plt.xlabel("Word")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.show()
```

```
      word  count
42     one   9394
182   will   6382
276    now   5809
217   love   5285
194     us   5129
245   time   4152
89    know   4107
37    back   4081
227    see   4058
119  still   3770
```

![](./plots/word_frequency.png)

Conclusion:
+ "one", "will", "now" and "love" is very highly saturated
+ The distrubution curve is very similar to the actual word distribution curve for English as a whole

Word cloud generation:

```python
wc = WordCloud(width=800, height=400,
               background_color="white",
               stopwords=set(STOPWORDS),
               colormap="viridis").generate(" ".join(all_poems))

plt.figure(figsize=(12,6))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud of Poem's Words")
plt.show()
```

![](./plots/wordcloud.png)

N-grams analysis:

```python
vectorizer = CountVectorizer(ngram_range=(2,4), stop_words = "english") # data contains latin / archaic english, but just english for now to avoid so many the
X = vectorizer.fit_transform(df["poem"])
ngram_freq = X.sum(axis=0).A1
ngrams = vectorizer.get_feature_names_out()
ngram_df = pd.DataFrame({"ngram": ngrams, "count": ngram_freq})
ngram_df = ngram_df.sort_values(by="count", ascending=False)

print(ngram_df.head(20))
```

```
               ngram  count
813212     dont know    404
3249424     thou art    227
3716710    years ago    217
2152037     new york    191
1897086    love love    154
1019288     far away    151
2214047      old man    150
3251468    thou hast    146
816006     dont want    143
765945    didnt know    140
1860713    long time    134
694849       day day    130
1854598     long ago    129
698314     day night    121
1682753    know know    111
3300318    time time     95
1619961     ive seen     91
3253476   thou shalt     89
3718847  years later     87
1363630    hand hand     80
```

Conslusion:
+ N-grams are set up to run from 2-4, but only pick up 2 words tri-grams, which means they are very popular, outweighing other 3-grams or 4-grams
+ There are latin words among the most popular 2-grams
+ "new york" is the most popular 2-gram of a location

### *Preprocessing*

There are no nulls and the data is very balanced so there is no pre-processing neccesary

### Train and test models

Models used
+ Logistic regression
+ SVM (Support Vector Machine)
+ Naive Bayes
+ BERT

With BERT being a modern text classification model using transformers.

The model can be found in the folder ```./features/BERT```

The folder's project structure is:
```
/features/BERT
|-> data.npy
|-> label.npy
|-> label_encoder.pkl 
```

### *Results*

### Models' performances

| Model                   | Accuracy | Macro F1 | Weighted F1 | Comment                                                             |
| :---------------------- | :------- | :------- | :---------- | :----------------------------------------------------------------------- |
| **Logistic Regression** | 0.51     | 0.50     | 0.51        | Best traditional model, balanced across classes                          |
| **SVM**                 | 0.50     | 0.50     | 0.50        | Very similar to Logistic Regression, slightly less consistent            |
| **Naive Bayes**         | 0.29     | 0.29     | 0.30        | Performs poorly; unable to capture contextual nuances                    |
| **BERT**                | 0.50     | 0.50     | 0.51        | Matches Logistic Regression overall, but with different class trade-offs |

The confusion matrices of these models:

Logistic Regression:
![](./plots/logistic.png)

SVM:
![](./plots/SVM.png)

Naive Bayes
![](./plots/NaiveBayes.png)

BERT
![](./plots/BERT.png)

### Topic differences

| Category            | Predict-ability                                                                                                                                  |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Nature**          | Consistently one of the easiest classes to classify (Highest recall).                  |
| **Relationships**   | Hard for classic models to classify, but BERT significantly boosts recall (0.73). Contextual/semantic understanding helps here.               |
| **Religion**        | Moderately hard to classify across all models; tends to confuse with "arts&sciences" or "love".                                                               |
| **Love**            | Fairly stable performance across all models (around 0.5 f1).                                                       |
| **Arts & Sciences** | Variable. Logistic Regression performs well but drops on BERT — likely due to more abstract or niche jargons. |


### **Conslusions:**

#### Logistic regression and SVM

Both achieve around 50% accuracy, which is normal and rather high for text classification without deep contextual embeddings. Likely stemmed from keyword-based patterns from TF-IDF input vector.

#### Naive Bayes

Severely under-performs. Which is understandable due to the assumption of conditional independence between classes being violated in poetic context.

#### BERT

Despite being a transformer-based model. BERT's accuracy do not raise much highert than the other model (i.e take Logistic regression as a comparison). However, per-class accuracy raises, especially for "relationships" poems.

#### Overall

+ No models significantly out-performs the 50% mark, which indicates this task is highly difficult, maybe due to overlapping categories or high innate randomness between people who labels the poems.

+ Logistic Regression is a solid baseline for this dataset.

+ BERT performs worse than expected, which may due to not enough data or lack of semantic specific pre-training. 



