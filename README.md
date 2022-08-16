# Paper: Global Explanation of Tree-Ensembles Models Based in Item Response Theory.

Autors: 

José Ribeiro - site: https://sites.google.com/view/jose-sousa-ribeiro

Lucas Cardoso - site: http://lattes.cnpq.br/9591352011725008

Raíssa Silva - site: https://sites.google.com/site/silvarailors

Vitor Cirilo - site: https://sites.google.com/site/vitorciriloaraujosantos/

Níkolas Carneiro - site: https://br.linkedin.com/in/nikolas-carneiro-62b6568

Ronnie Alves (Leader) - site: https://sites.google.com/site/alvesrco

## Absrtact

A Explainable Artificial Intelligence - XAI, é uma sub-área de Machine Learning que tem como principal objetivo, o estudo e o desenvolvimento de técnicas voltadas a explicação de modelos computacionais caixa preta, ou seja, modelos que apresentam limitada capacidade de auto explicação de suas predições. Nos últimos anos, pesquisadores desta área vêm formalizando propostas e desenvolvendo novas medidas que se propõem explicar como os modelos caixa preta realizam determinadas predições. Esta é considerada, pela comunidade de pesquisa, uma área de pesquisa em voga, pois cada vez mais pode-se notar sistemas inteligentes baseados em modelos de aprendizagem de máquina caixa preta presentes no dia-a-dia da sociedade, crescendo também a necessidade de explicá-los, principalmente quando o problema de predição a ser resolvido envolve um contexto sensível. Em publicações anteriores, foram encontrados indícios de como a complexidade do modelo (dataset e algoritmo) afeta explicações globais baseadas em ranques geradas pelas medidas de XAI Ciu, Dalex, Eli5, Lofo, Shap e Skater. Sendo que, no estudo em questão foi verificada a existência de modelos baseados em tree-ensemble mais fáceis de serem explicados e outros mais difíceis, mostrando as necessidades e limitações das atuais medidas de explicabilidade. Neste sentido, esta pesquisa surge com o objetivo de apresentar uma nova medida de XAI chamada de Explainable based in Item Response Theory - eXirt, capaz de explicar modelos caixa preta tree-ensemble utilizando as propriedades (adivinhação, dificuldade e discriminação) da Teoria de Resposta ao Item - IRT, técnica esta já consolidada em outras áreas. Para isto, foi criado um benchmark que a partir de 40 diferentes datasets e 2 diferentes algoritmos (Random Forest e Gradient Boosting) gerou 6 diferentes ranques de explicabilidades por meio de medidas de XAI já conhecidas pela comunidade de computação, juntamente com 1 ranque de pureza de dados e 1 ranque da medida de XAI proposta, eXirt, totalizando assim 8 ranques globais para cada um modelo criado (total de 640 ranques). Os resultados mostraram que a medida eXirt apresentou ranques diferentes dos demais ranques analisados, o que demonstra que a metodologia aqui defendida, gera um conjunto de explicações globais de atributos dos modelos tree-ensemble ainda não explorados pelas atuais medidas de XAI, seja para os modelos mais fáceis de se explicar ou mesmo os mais difíceis.

## This repository was created to contain all additional information from the article "Global Explanation of Tree-Ensembles Models Based in Item Response Theory", for reproducibility purposes.

Description for execution:
All data regarding the reproducibility of this work can be found in this repository.

  - data cluster 0: all datasets, performance graphs, models and analyzes coming from cluster 0 data.

  - data cluster 1: all datasets, performance graphs, models and analyzes coming from cluster 1 data.

  - XAI - IRT Notebook.ipynb: s all the source code used to execute the experiments presented in this research. It should be noted that this notebook is properly commented, documented and separated into sections for better understanding in case of an execution.

  - df_dataset_properties: dataset with all 15 properties analyzed in the Multiple Correspondence Analysis - MCA.

To run the notebook XAI - IRT Notebook.ipynb, it is suggested to use Google Colab, for a better and faster execution of the tool.
