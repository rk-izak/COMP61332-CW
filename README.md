
# COMP61332 Text Mining Coursework - Relation Extraction

## Group Members
- Radoslaw Izak (radoslaw.izak@postgrad.manchester.ac.uk)
- Patricio Jaime Porras (patricio.jaimeporras@postgrad.manchester.ac.uk)
- Roshan Bhalaji Ramesharavind Lathamahesh (roshan.bhalaji@postgrad.manchester.ac.uk)

## Dependencies
To install the required dependencies, create a `venv` and activate it via:
```
python3.7 -m venv venv
source venv/bin/activate
```

and run:

```
pip3 install -r requirements.txt
```
The same `venv` should be used for all Jupyter notebooks found within this repository. For more information visit [here](https://code.visualstudio.com/docs/datascience/jupyter-notebooks).

Core Dependencies Include:
- Python 3.7.4
- PyTorch 1.13.1
- Tensorflow 2.11.0
- Transformers 4.30.2
- scikit-learn 1.0.2
- pandas 1.3.5
- numpy 1.21.6
- matplotlib 3.5.3

## Saved Checkpoints for Inference and Scoring
Due to size restrictions, we could not share them in this repository. The same applies for `GloVe` embeddings. However, all relevant checkpoints and other files can be downloaded from [here](https://drive.google.com/drive/folders/1T1bDhG6Dfd-xuFTEnW6D53o95GBHDhxc).

All models should be placed in `checkpoints/PRETRAINED/{lstm/nb_svm}` folders respectively, as seen in the attached download.

The `GloVe` embedding in `.txt` format should be placed in the `utils/` folder.

## Dataset
We have chosen the [SemEval 2010 Task 8 dataset](https://aclanthology.org/S10-1006.pdf) for this coursework. This dataset is widely used in relation extraction tasks and includes annotated sentences for multi-way classification of semantic relations between pairs of common-noun entities.

The raw dataset, as well as our transformer `.json` files can be found in `data/raw/` and `data/full/` respectively.

## Relation Extraction Methods

### Method 1: Traditional Machine Learning Approaches via Naive Bayes (NB) and Support Vector Machines (SVMs)
- Category: Traditional machine learning-based approach
- Description: We have tested several methods and have found the following:
    * The best approach uses SVC with a pre-trained embedding (BERT) and a rbf kernel without bagging;
    * The best Naive Bayes variation was the Complement NB with TF-IDF vectorization, but BoW had similar results. Both vectorizations seem to affect the model similarly, since NB can't handle context very well;
    * Since the dataset is unbalanced, Gaussian and Bernulli Naive Bayes variations return poor results;
    * Both Complement and Multinomial are better suited for this task, even though they still have room for improvement;
    * Naive Bayes does not gain any performance by using embeddings (since it is not suitable for them as is);
    * Even though some pre-processing techniques were tested (i.e. parts-of-speech) it didn't help the models at all;
    * SVM take too long to compute if a BoW or TF-IDF vectorization technique is used due to dimensionality and sparce matrices;
    * SVM take more advantage of embeddings only when they use a non-linear kernel such as rbf;
    * Complete final architectures can be viewed in `utils/nb_svm_models.py`
- Adaptations: Experimentation with different model variations, pre-procesing techniques, use of different vectorizations, test with different kernels, tried adding bagging for SVMs, added embeddings for SVM of different sizes. No hyperparameter tuning needed.
- Instructions:
    * To run model training, please refer to `TRAIN_NB_SVM.ipynb` notebook.
    * To run model scoring, please refet to `SCORE_NB_SVM.ipynb` notebook (needs pretrained checkpoints).
    * To run model inference, please refer to `INFERENCE_NB_SVM.ipynb` notebook (needs pretrained checkpoints).


### Method 2: Deep Learning via Bi-Directional Long Term Short Memory networks (BiLSTMs)
- Category: Deep learning-based approach using recurrent networks
- Description: We have tested several methods, and have found the following:
    * The best network with non-pretrained embedding utilised contextual information from BiLSTMs, and Feature Extraction from Global Max Pooling;
    * We have focused on smaller models to make the architectural changes more noticable;
    * We started with a basic double-layered LSTM, and increased the complexity from there;
    * Complete final architectures can be viewed in `utils/lstm_models.py`;
    * A simple double layered LSTM behaved extremely poorly, but making it bi-directional increased the accuracy from 17% to approx. 51%, which underlines the importance of good semantic information flow through the network.
    * Similar methods for feature extraction, such as Convolution or Attention, did not work as well;
    * We believe Convolution has distilled the relations too much, leading to worse performance;
    * We believe Attention did not work well here, because of small data size, making it unable to capture details;
    * Although regularization of LSTMs have helped, it was hardly noticable, therefore could be ignored;
    * Average Max Pooling, in lign with findings from Convolution, has distilled the text nuances making classification more difficult;
    * Embedding is key: even though the best method achieved significant improvement due to Global Max Pooling, pretrained embeddings have increased accuracy exponentially. Interestingly enough, GloVe embeddings improved performance almost as much as BERT's. This means that sometimes Global information is good enough, against contextual information, which aligns with our findings with Global Max Pooling;
- Adaptations: Experimentation with different model architectures, layers, and methods. No hyperparameter tuning. Smart modelling.
- Instructions:
    * To run model training, please refer to `TRAIN_LSTM.ipynb` notebook.
    * To run model scoring, please refet to `SCORE_LSTM.ipynb` notebook (needs pretrained checkpoints).
    * To run model inference, please refer to `INFERENCE_LSTM.ipynb` notebook (needs pretrained checkpoints).


## Evaluation Metrics
Both methods were evaluated on the SemEval 2010 Task 8 dataset test set, using accuracy, precision, recall, and F1 score as metrics.

## Results and Discussion in a nutshell
The deep learning models outperformed the traditional ML in our experiments, demonstrating the effectiveness of deep learning and pre-trained embeddings in capturing classification nuances. However, Naive Bayes' and SVM's interpretability and lower computational requirements are notable. Moreover, we underlined the importance of competent word embeddings, which do not have to be contextual in order to achieve good performance. For more information, please refer to the attached report.

## Code Attribution
- The official SemEval 2010 Task 8 dataset and processing guidelines were followed.
- Naive Bayes implementations based on scikit-learn library.
- SVM implementations based on scikit-learn library.
- BiLSTM implementations based on Tensorflow library.
- BERT checkpoints utilize the Transformers library by Hugging Face.

## File Structure
- `README.md`: Overview of the project.
- `requirements.txt`: Required Python dependencies.
- `INFERENCE_*.ipynb`: Notebook for relevant models inference.
- `SCORE_*.ipynb`: Notebook for relevant models scoring.
- `TRAIN_*.ipynb`: Notebook for relevant models training.
- `checkpoints/`: folder for pretrained model checkpoints, as well as new training.
- `utils/`: scripts for preprocessing, training, architectures, and GUI.
- `data/`: SemEval 2010 Task 8 dataset files, clean and transformed.
- `COMP61332.pdf`: Detailed description of our work.

## References
- Hendrickx, Iris, Kim, Su Nam, Kozareva, Zornitsa, Nakov, Preslav, Ó Séaghdha, Diarmuid, Padó, Sebastian, Pennacchiotti, Marco, Romano, Lorenza and Szpakowicz, Stan. "SemEval-2010 Task 8: Multi-Way Classification of Semantic Relations between Pairs of Nominals". Proceedings of the 5th International Workshop on Semantic Evaluation, July 2010, Uppsala, Sweden. Association for Computational Linguistics, pages 33-38. https://www.aclweb.org/anthology/S10-1006

- Hong, G. (2005). Relation Extraction Using Support Vector Machine. In Second International Joint Conference on Natural Language Processing: Full Papers.

- Zhang, S., Zheng, D., Hu, X., \& Yang, M. (2015). Bidirectional long short-term memory networks for relation classification. In Proceedings of the 29th Pacific Asia conference on language, information and computation (pp. 73-78).

- Thillaisundaram, A., \& Togia, T. (2019). Biomedical relation extraction with pre-trained language representations and minimal task-specific architecture. In Proceedings of the 5th Workshop on BioNLP Open Shared Tasks (pp. 84-89).

- Vaswani, Ashish, Shazeer, Noam, Parmar, Niki, Uszkoreit, Jakob, Jones, Llion, Gomez, Aidan N, Kaiser, Łukasz and Polosukhin, Illia. "Attention is all you need". Advances in neural information processing systems, pages 5998-6008, 2017.

- Devlin, Jacob, Chang, Ming-Wei, Lee, Kenton and Toutanova, Kristina. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding". CoRR, abs/1810.04805, 2018. http://arxiv.org/abs/1810.04805

- Vikramkumar, Vijaykumar B and Trilochan. "Bayes and Naive Bayes Classifier". CoRR, abs/1404.0933, 2014. http://arxiv.org/abs/1404.0933

- Zhou, Peng, Qi, Zhenyu, Zheng, Suncong, Xu, Jiaming, Bao, Hongyun and Xu, Bo. "Text Classification Improved by Integrating Bidirectional LSTM with Two-dimensional Max Pooling". CoRR, abs/1611.06639, 2016. http://arxiv.org/abs/1611.06639

- Zhang, Dongxu and Wang, Dong. "Relation Classification via Recurrent Neural Network". CoRR, abs/1508.01006, 2015. http://arxiv.org/abs/1508.01006

- Zhou, GuoDong, Su, Jian, Zhang, Jie and Zhang, Min. "Exploring Various Knowledge in Relation Extraction". Proceedings of the 43rd Annual Meeting of the Association for Computational Linguistics (ACL'05), June 2005, Ann Arbor, Michigan. Association for Computational Linguistics, pages 427-434. https://aclanthology.org/P05-1053. doi: 10.3115/1219840.1219893
