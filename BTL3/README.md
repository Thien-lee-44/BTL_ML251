# BTL3 môn Học Máy, mã môn học CO3117 - Học máy với dữ liệu hình ảnh
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


## **Content**
This BTL applies both the traditional ML pipeline with one tradidional CNN based feature extractor and 3 modern transformer based feature extractors.

## *Setup and run*

Clone the project:

```bash
git clone https://github.com/Thien-lee-44/BTL_ML251.git
cd BTL_ML251/BTL3
```

Then run the collab file located in ```./notebook```, the collab file is set up to be able to run all from the start

## **Dataset**
Dataset includes 6862 weather images, Including 11 types:

+ dew 
+ fogsmog 
+ frost 
+ glaze 
+ hail 
+ lightning 
+ rain 
+ rainbow 
+ rime 
+ sandstorm 
+ snow

This dataset is available at [kraggle](https://www.kaggle.com/datasets/jehanbhathena/weather-dataset).

## Pipeline

Each image go through a feature extractor before a classical classifier head.

The extracted features can be found in ```./features``` for all feature exctractors innvolved. 

## Feature Extractors

| Model                | Feature Dimension | Description                                          |
| :------------------- | :---------------: | :--------------------------------------------------- |
| **EfficientNet**     |       1,536       | CNN-based, strong baseline for visual features       |
| **DINOv2**           |        384        | Self-supervised ViT, highly semantic features        |
| **Swin Transformer** |        768        | Hierarchical transformer, good spatial understanding |
| **DeiT**             |        384        | Data-efficient Vision Transformer                    |

## Classifier

Each extractor was evaluated with three classical classifiers:

+ SVM (Support Vector Machine)

+ Logistic Regression

+ Random Forest (n=50 and n=100)

## Accuracy Comparison of results

| Feature Extractor |  SVM | Logistic Regression | Random Forest (k = 50) | Random Forest (k = 100) |
| :---------------- | :--: | :-----------------: | :----------------: | :-----------------: |
| **EfficientNet**  | 0.83 |         0.84        |        0.74        |         0.76        |
| **DINOv2**        | 0.87 |       **0.91**      |        0.87        |         0.88        |
| **Swin**          | 0.88 |       **0.90**      |        0.80        |         0.82        |
| **DeiT**          | 0.82 |         0.84        |        0.76        |         0.79        |


## Best Performing Configuration

| Extractor        | Classifier              | Accuracy | Macro F1 | Weighted F1 | Notes                                                      |
| :--------------- | :---------------------- | :------- | :------- | :---------- | :--------------------------------------------------------- |
| **DINOv2**       | **Logistic Regression** | **0.91** | 0.92     | 0.91        | Excellent balance of recall & precision across all classes |
| **Swin**         | Logistic Regression     | 0.90     | 0.90     | 0.90        | Competitive transformer, strong generalization             |
| **DINOv2**       | Random Forest (100)     | 0.88     | 0.89     | 0.88        | Very stable results                                        |
| **EfficientNet** | Logistic Regression     | 0.84     | 0.84     | 0.84        | Best CNN-based baseline                                    |
| **DeiT**         | Logistic Regression     | 0.84     | 0.85     | 0.84        | Consistent but slightly weaker than DINOv2/Swin            |

## Notes about the performance
 
### CNN Extractor (EfficientNet)

1. EfficientNet provides solid baseline results.

2. The model performs best with Logistic Regression.

3. The model have slightly lower accuracy than Transformer-based extractors, reflecting the representational advantage of ViT-style (Vision Transformer) features.

### Transformer-Based Extractors (DINOv2, Swin, DeiT)

1. **DINOv2**: Shows the best overall performance, benefiting from strong semantic self-supervised embeddings.

2. **Swin**: Performs very closely to **DINOv2**, slightly behind in macro F1 but strong in consistent class recall.

3. **DeiT**: Offers competitive results with smaller feature vectors, suggesting better efficiency for moderate datasets.

### Random Forests

+ Random forests accuracy improves modestly from 50 → 100 estimators (which may indicates we have not hit the "dilution" threshold yet). But remains ~10% lower than SVM or Logistic Regression on average.

+ Random forests tends to have higher recall for frequent classes - i.e more representation in the data (e.g., rime, fogsmog) but gradually lower precision for smaller classes (snow, rainbow).

### Overall notes:

+ DINOv2 + Logistic Regression head gives the highest accuracy (91%) and macro F1 (0.92).

+ Swin + Logistic Regression head is a very close second (90% accuracy). More test data is required to indicate which one ended up generalized the data better.

+ Random Forest models are less effective for these feature dimensions. Which may indicate normal splitting based question strategy do not capture the intricacies of the higher-dimensional embedding so well.

+ SVM and Logistic Regression handle high-dimensional embeddings efficiently and generalize better. Logistic Regression is especially note-worthy since it performs well inndicates the data becomes linear in this new high dimension.

+ Transformer-based features outperform CNN-based ones for diverse visual weather conditions. This highlighting the power of self-supervised and hierarchical attention mechanisms to embed the data for this task - classifying whether phenomenons.

## Summary of top 5 Feature extractor + Head Results
| Extractor        | Classifier              | Accuracy | Macro F1 | Weighted F1 | Notes                                                      |
| :--------------- | :---------------------- | :------- | :------- | :---------- | :--------------------------------------------------------- |
| **DINOv2**       | **Logistic Regression** | **0.91** | 0.92     | 0.91        | Excellent balance of recall & precision across all classes |
| **Swin**         | Logistic Regression     | 0.90     | 0.90     | 0.90        | Competitive transformer, strong generalization             |
| **DINOv2**       | Random Forest (100)     | 0.88     | 0.89     | 0.88        | Very stable results                                        |
| **EfficientNet** | Logistic Regression     | 0.84     | 0.84     | 0.84        | Best CNN-based baseline                                    |
| **DeiT**         | Logistic Regression     | 0.84     | 0.85     | 0.84        | Consistent but slightly weaker than DINOv2/Swin            |
