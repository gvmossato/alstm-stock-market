
<br />

<p align="center">
  <img src="./misc/alstm-logo-yellow.svg" alt="alstm-logo" width="550px" />
</p>

<p align="center">  
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python%203.11-3776AB?style=for-the-badge&logo=python&logoColor=yellow&color=3776AB" alt="python-badge" />
  </a>
  <a href="https://www.tensorflow.org/">
    <img src="https://img.shields.io/badge/TensorFlow%202.14-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="tensoflow-badge" />
  </a>
  <a href="https://www.ibm.com/products/cloudant">
    <img src="https://img.shields.io/badge/IBM%20Cloudant-1261FE?style=for-the-badge&logo=IBM%20Cloud&logoColor=white" alt="python-badge" />
  </a>
</p>

<br />

<p align="center">
Projeto desenvolvido como <b>Trabalho de Conclusão de Curso</b> durante o último ano de graduação em <b>Engenharia Mecatrônica</b> na Escola Politécnica da Universidade de São Paulo
</p>

<br />

## 🔎 Sumário

WIP

Neste repositório:

* <a href="#-sobre">📜 Sobre</a> - Breve apresentação do projeto

* <a href="#-rede">🧠 Rede</a> - Arquitetura implementada

* <a href="#-treinamento">🦾 Treinamento</a> - Otimizações feitas e dados utilizados

* <a href="#-treinamento">📈 Resultados</a> - O que alcançamos com o modelo

* <a href="#-aplicação">🌎 Aplicação</a> - Divulgação aberta dos resultados

* <a href="#-colaboradores">🤝 Colaboradores</a> - Partes envolvidas

## 📜 Sobre

> *History never repeats itself, but it does often rhyme*  
> [Mark Twain](https://pt.wikipedia.org/wiki/Mark_Twain)

Este repositório contém o código do estudo sobre a **previsão do índice [S&P 500](https://www.spglobal.com/spdji/en/indices/equity/sp-500/#overview)**, no qual foi desenvolvido um modelo utilizando **células de memória de longo-curto prazo ([LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory)) combinadas com [mecanismos de atenção](https://en.wikipedia.org/wiki/Attention_(machine_learning))**. O modelo passou por otimizações com técnicas de [*Grid Search* e *Bayesian Seach*](https://en.wikipedia.org/wiki/Hyperparameter_optimization), demonstrando um desempenho promissor na previsão de preços de fechamento.

O trabalho foi majoritariamente inspirado no artigo *[Forecasting stock prices with long-short term memory neural network based on attention mechanism](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0227222)* (2020) de Jiayu Qiu, Bin Wang e Changjun Zhou, recebendo honras como um dos cinco melhores projetos de 2023 no curso.

No estudo também foram exploradas aplicações mais práticas do modelo através de técnicas de gestão de banca, avaliando a rentabilidade das previsões em um ambiente controlado e com resultados igualmente promissores.

#### 📕 **[Versão final da monografia](https://www.github.com/gvmossato/alstm-stock-market/misc/Previsão_do_Índice_SP_500_Utilizando_LSTM_e_Mecanismos_de_Atencao.pdf)**

## Problemática & Motivação

WIP

## Dados & Pre-Processamento

Fluxograma simplificado dos dados sendo processados até serem consumidos pelo modelo:

![dados](https://i.ibb.co/tmhPnPk/dados.png)

Conforme a imagem:

1. **Fonte:** o conjunto de dados utilizado nesse projeto é um recorte histórico abrangendo de 03 de janeiro de 1983 a 01 de setembro de 2023, retirado diretamente do [Yahoo! Finance](https://finance.yahoo.com/quote/%5EGSPC/).

2. **Redução de Ruído:** [transformada wavelet](https://towardsdatascience.com/the-wavelet-transform-e9cfa85d7b34) com a família de funções Coiflets até terceira ordem, a qual é particularmente eficaz na redução de ruído em sinais não estacionários como os de preços de ações.

3. **Normalização:** para assegurar que todas as variáveis tenham o mesmo peso durante o treinamento da rede, aplicamos a normalização [Z-Score](https://www.statology.org/z-score-normalization/). Isso coloca todas as variáveis na mesma escala, neutralizando o efeito de disparidades nas magnitudes de preços e volumes transacionais.

4. **Repartição:** os dados são divididos em três segmentos: treino, validação e teste, na proporção de 95/2.5/2.5. Escolhemos essa divisão para permitir um ajuste fino dos hiperparâmetros (usando o conjunto de validação) e uma avaliação honesta do desempenho do modelo (usando o conjunto de teste). Ao manter a ordem temporal (sem embaralhamento), respeitamos a sequência natural dos eventos no mercado de ações.

5. **Janelamento:** implementamos uma técnica de janela deslizante com tamanho de 20 dias (aproximadamente um mês em dias úteis), que constitui o passo temporal do nosso modelo. Dentro dessa janela, seis séries temporais distintas — abertura, fechamento, máxima, mínima, [fechamento ajustado](https://help.yahoo.com/kb/SLN28256.html) e volume (em quantidades) — são fornecidas ao modelo. Com esses dados ele prediz o valor de fechamento no vigésimo primeiro dia.

## 🦾 Otimização

Como citado, realizamos a tunagem de hiperparâmetros em duas fases distintas:

### Grid Search

Explorar deterministicamente as redondezas do modelo apresentado no artigo base utilizado.

- **Tamanho do Estado Oculto**: valores de 10, 20, 50 e 100 para o tamanho da memória das células LSTM.
- **Taxa de Aprendizado**: experimentamos com 0,001, 0,01 e 0,1 para entender o impacto na convergência do modelo.
- **Tamanho do Lote**: variamos entre 64, 128, 256 e 512 para observar as diferenças no treinamento do modelo.

Um total de **48 combinações únicas** de hiperparâmetros foram avaliadas, utilizando a metodologia de [validação cruzada com k-dobras](https://medium.com/@soumyachess1496/cross-validation-in-time-series-566ae4981ce4) (k=3), resultando em **144 execuções distintas**.

### Bayesian Search

Na sequência, um Bayesian Search mais refinado foi implementado com base nas combinações de hiperparâmetros mais promissoras do Grid Search.

- **Tamanho do Lote (Batch Size)**: valores específicos de 64, 128, 256, 512 e 1024 foram considerados.
- **Taxa de Aprendizado (Learning Rate)**: um intervalo contínuo de 0,0001 a 0,01 foi explorado.
- **Taxa de Dropout**: avaliamos um espectro contínuo de 0% a 30% para encontrar o equilíbrio ideal na prevenção de overfitting.

Decidimos fixar o **tamanho do estado oculto** em 20, baseando-nos nas descobertas do Grid Search e na sintonia com o tamanho de entrada de dados (20 dias úteis).

Ao longo de **100 configurações**, avaliamos e validamos diferentes modelos, totalizando **300 execuções** por meio da validação cruzada com k-dobras (k=3). Essa abordagem mais exaustiva foi desenhada para identificar a combinação ideal de hiperparâmetros que aprimoraria a performance do modelo.

### Resumo

Consolidado dos testes:

| Hiperparâmetro | Grid Search | Bayesian Search |
|----------------|-------------|-----------------|
| Tamanho do Estado Oculto | 10, 20, 50, 100 | Não testado |
| Taxa de Aprendizado | 0,001; 0,01; 0,1 | 0,0001 a 0,01 |
| Tamanho do Lote | 64, 128, 256, 512 | 64, 128, 256, 512, 1024 |
| Taxa de Dropout | Não testado | 0% a 30% |

## 🧠 Rede & Treinamento

A arquitetura proposta para o modelo de previsão do índice S&P 500 incorpora um total de **20 células LSTM**, em consonância com a janela de 20 dias que é analisada. Isto é, para uma janela móvel de 20 dias, são lidos [OHLC](https://www.investopedia.com/terms/o/ohlcchart.asp) + Fechamento Ajustado + Volume (quantidades), para então predizer o fechamento do 21º dia.

![modelo](https://i.ibb.co/jk7DVzR/modelo.png)

Como msotra imagem anterior, após o processamento pelas células LSTM, as saídas são submetidas ao mecanismo de *[soft attention](https://stackoverflow.com/questions/35549588/soft-attention-vs-hard-attention)*. Esse mecanismo avalia as contribuições de cada célula LSTM e pondera sua influência, permitindo que a importância de momentos distintos no tempo seja diferenciada, ao invés de focar apenas na informação mais recente. Isso se baseia na premissa de que eventos passados dentro da janela de tempo podem ter relevância semelhante ou até maior do que os mais recentes.

A "camada de atenção" então agrega as saídas das células LSTM ponderadas em um vetor de contexto que concentra as informações relevantes detectadas pela rede. Esse vetor de contexto é então passado por uma camada de [dropout](https://towardsdatascience.com/dropout-in-neural-networks-47a162d621d9#waht-is-a-dropout), com uma taxa de desativação de 12,241%, antes de ser apresentado à última camada da rede.

A fase final da arquitetura é composta por uma camada densa com um único neurônio, cuja **função de ativação linear** é adequada para tarefas de regressão como a previsão de índices de ações. Esse neurônio processa o vetor e produz o output final da rede: a previsão do valor de fechamento do S&P 500.

Por fim, o treinamento da rede ocorre ao longo de **2000 epochs** com um **batch size de 128**. O algoritmo **[ADAM](https://medium.com/@LayanSA/complete-guide-to-adam-optimization-1e5f29532c3d)** é o escolhido para otimização, operando com uma **taxa de aprendizado de 0,00018** - esses parâmetros foram selecionados com base nos resultados da busca em grid, da busca bayesiana e da análise da curva de aprendizado.

## 📈 Resultados

WIP

## 🌎 Aplicação

WIP

## Uso

WIP

## 🤝 Colaboradores

O trabalho foi desenvolvido por mim, [Gabriel Mossato](https://br.linkedin.com/in/gvmossato), em parceria com o à época também graduando [Paulino Fonseca](https://br.linkedin.com/in/paulinoveloso) sob a orientação do [Prof. Dr. Oswaldo Luiz do Valle Costa](https://bv.fapesp.br/en/pesquisador/191/oswaldo-luiz-do-valle-costa) do departamento de Engenharia Elétrica da EP-USP.
