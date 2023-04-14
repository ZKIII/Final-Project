[toc]

# Final Project



## Introduction

In this project, we build two text classification models, to test effectiveness of RNN and Transformer. Basically, the structure of these two models is similar, the first layer is embedding lookup layer, the second layer is feature extractor layer, and a mlp layer is concatenated after feature extractor to output logits. The task we solved in this project is IMDB review sentiment classification task, it is a binary classification problem, the label 1 for positive and 0 for negative. The input of model is words of each review and output the probability of this review is classified as 1. 



## Model:

![img](https://cs.utm.utoronto.ca/~zhengz45/CSC413/CSC413_Final_Project_Imgs/model.jpg)

There are two models in our project, and overall model structure is shown as above, the extractor of the first model is a bidirectional LSTM, while in second model, the extractor is a Transformer encoder. 



## Model Parameters:

| Parameter       | Size                                                         | Annotation                                                   |
| --------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Embedding layer | 123039 * 128                                                 | Common in two models                                         |
| LSTM Layer      | 2 *4 * [64 * (128 + 64) + 64]                                | Existing in the model in which the extractor is LSTM         |
| Transformer     | 128 * 128 * 3 + 128 * 128 * 2                                | Existing in the model in which the extractor is Transformer Encoder |
| MLP             | 128 * 1 in the model which extractor is Transformer encoder and 128 * 2 * 1 in the model which extractor is LSTM |                                                              |

The parameter of Embedding layer is vocab_size * embedding, it is a huge matric to to lookup for each word. For the model using LSTM as feature extractor, the hidden layer of LSTM is 1, and it is bidirectional LSTM, so the total number parameters of this layer is 2 * #Params(LSTM). For the model using Transformer encoder as feature extractor, the layer of the encoder is 1, the number of headers is 2. The MLP is the last layer of model to output logits, and the input size should be matched with output size of feature extractor. For the model using LSTM, we use the concat of two hidden cells of different directions as input of MLP. For the model using Transformer encoder, we use the last token’s embedding as input of MLP. 

## Dataset Description

The dataset we use to train the model is IMBD review dataset

| Dataset                 | URL                                                          | Statistics                                                 | Purpose                    |
| ----------------------- | ------------------------------------------------------------ | ---------------------------------------------------------- | -------------------------- |
| IMDB 50K Review dataset | [IMDB Dataset of 50K Movie Reviews \| Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) | Total sample num: 50000,Vocab Size: 123039,Avg_seq_len:231 | Text binary classification |

 

## Data PreProcess

1. Extract text between <br></br> use reg expression.
2. Remove punctuation for text.
3. Map the text token to an idx value in vocabulary. 
4. Build DataLoader which can generate batch samples.

 

## Data split

Split the dataset into training set and test test corresponding ratio is 0.8 and 0.2.



## Training curve

We use the Tensorboard to record batch loss while training to test whether the training process is normal or not.  For TransformerEncoderClsModel, the training process like: 

![img](https://cs.utm.utoronto.ca/~zhengz45/CSC413/CSC413_Final_Project_Imgs/TransformerEncoderClsModel.jpg) 

For model with LSTM:

![img](https://cs.utm.utoronto.ca/~zhengz45/CSC413/CSC413_Final_Project_Imgs/LSTM.jpg) 



## Hyperparameter Tuning

There are many parameters we can tune, for sequence length, since the average length of reviews is 230, we set the sequence length to be 200. The vocab size is large, we set the word embedding size to be 128. We set the learning rate to be 0.0005 to make sure the training process is steady.

Quantitative Measures

We use the validation auc to measure the model effectiveness, for model with TransformerEncoder, validation auc is:

![img](https://cs.utm.utoronto.ca/~zhengz45/CSC413/CSC413_Final_Project_Imgs/TransformerEncoderClsModel_Acc.jpg) 

 

For model with LSTM, validation auc is:

![img](https://cs.utm.utoronto.ca/~zhengz45/CSC413/CSC413_Final_Project_Imgs/LSTM_Acc.jpg) 

 

## Quantitative and Qualitive Results:

The ratio of positive to negative is 1:1, so we can valid the model effectiveness from validation auc.



## Justification

In this project, we want to compare the effectiveness of LSTM and Transformer. We assume that Transformer is better than LSTM when deal with text sequence data. After do some experiments on IMDB review dataset, we can see the model with Transformer as feature extractor performs better than the model with bidirectional LSTM according to validation auc.  Which within our expectation, because even bidirectional LSTM can memory as much as useful information for a whole sequence, it still can not beat the Multi-head attention in which each item of a sequence can interact with others directly, which is more straightforward compared with hidden cell. Both the hidden cell or last item of LSTM and Transformer encoder encodes the whole sequence information to some extent, so LSTM cell and Transformer encoder is very powerful sequence feature extractor, and they can replace the MLP module  in an ordinary DNN model to form a simple text classifier. However, LSTM is more sensitive to the hyper-parameter like epoch, sequence and learning rate, and we see there is no convergence if we set the learning rate too large like 0.01.

 ![img](https://cs.utm.utoronto.ca/~zhengz45/CSC413/CSC413_Final_Project_Imgs/Justification.jpg)

And we observe the model with Transformer is more stablized compare with LSTM.

 

## Ethical Consideration

This model is only trained on IMDB review dataset, which can be strongly impacted by the training data distribution. And may be not applicable to other scenarios.



## Division Of Labor

imdb_data_preprocess.py (Contributors: Ziqi Zheng)

multi_head_attention.py (Contributors: Zhenyu Wang)

imdb_text_classifier.py (Contributors: Ziqi Zheng & Zhenyu Wang)

README.md (Contributors: Ziqi Zheng & Zhenyu Wang)

Data Source: [IMDB Dataset of 50K Movie Reviews \| Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)