# Paper: Explanations Based on Item Response Theory (eXirt): A Model-Specific Method to Explain Tree-Ensemble Model in Trust Perspective.

Autors: 

José Ribeiro - site: https://sites.google.com/view/jose-sousa-ribeiro

Lucas Cardoso - site: http://lattes.cnpq.br/9591352011725008

Raíssa Silva - site: https://sites.google.com/site/silvarailors

Vitor Cirilo - site: https://sites.google.com/site/vitorciriloaraujosantos/

Níkolas Carneiro - site: https://br.linkedin.com/in/nikolas-carneiro-62b6568

Ronnie Alves (Leader) - site: https://sites.google.com/site/alvesrco

## Absrtact

Solutions based on tree-ensemble models represent a considerable alternative to real-world prediction problems, but these models are considered black box, thus hindering their applicability in problems of sensitive contexts (such as: health and safety). The Explainable Artificial Intelligence - XAI aims to develop techniques that aim to generate explanations of black box models, since normally these models are not self-explanatory. According to the nature of human reasoning, it can be considered that the explanation of a model can make it reliable or unreliable, since a human individual can only trust what he can understand. In recent years, XAI researchers have been formalizing proposals and developing new methods to explain black box models, with no general consensus in the community on which method to use to explain these models, with this choice being almost directly linked to the popularity of a specific method. Methods such as Ciu, Dalex, Eli5, Lofo, Shap and Skater emerged with the proposal to explain black box models through global rankings of feature relevance, which based on different methodologies, generate global explanations that indicate how the model's inputs explain its predictions. In this context, 41 datasets, 4 tree-ensemble algorithms (LightGradientBoost, CatBoost, Random Forest, and Gradient Boosting), and 6 XAI methods were used to support the launch of a new XAI method, called eXirt, based on Item Response Theory - IRT and aimed at tree-ensemble black box models that use tabular data referring to binary classification problems. In the first set of analyses, the 164 global feature relevance ranks of the eXirt were compared with 984 ranks of the other XAI methods present in the literature, seeking to highlight their similarities and differences. In a second analysis, exclusive explanations of the eXirt based on Explanation-by-example were presented that help in understanding the model trust. Thus, it was verified that eXirt is able to generate global explanations of models and also local explanations of instances of models through IRT, showing how this consolidated theory can be used in machine learning in order to obtain explainable and reliable models.

**Note: This repository was created to contain all additional information from the article "Explanations Based on Item Response Theory (eXirt): A Model-Specific Method to Explain Tree-Ensemble Model in Trust Perspective", for reproducibility purposes.
**

**Description for execution:**
All data regarding the reproducibility of this work can be found in this repository.
  
  - **Cluster 0**: all datasets, performance graphs, models and analyzes coming from cluster 0 data.

  - **Cluster 1**: all datasets, performance graphs, models and analyzes coming from cluster 1 data.

  - **Cluster 2**: all datasets, performance graphs, models and analyzes coming from cluster 2 data.

  - **Cluster 3**: all datasets, performance graphs, models and analyzes coming from cluster 3 data.

  - **eXirt - Pipeline - v0.3.2 - Remake Import Exirt - M1 a M4.ipynb**: all the source code used to execute the experiments presented in this research. It should be noted that this notebook is properly commented, documented and separated into sections for better understanding in case of an execution.

  - **df_dataset_properties**: dataset with all 15 properties analyzed in the Multiple Correspondence Analysis - MCA;

  - **Analisys of Datasets by eXirt.ipynb**: all analysis of item parameter values for the specifics datasets;
  
  - **eXirt - Simple execution.ipynb**: simple execution of eXirt;
  
  - **https://pypi.org/project/eXirt/:** eXirt distribution repository.

To run the notebook XAI - IRT Notebook.ipynb, it is suggested to use Google Colab, for a better and faster execution of the tool.
