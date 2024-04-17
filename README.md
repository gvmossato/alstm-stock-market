
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
Projeto desenvolvido como <b>Trabalho de Conclusão de Curso</b> durante o último ano de graduação em <b>Engenharia Mecatrônica</b> na Escola Politécnica da Universidade de São Paulo (EP-USP)
</p>

<br />

## 🔎 Sumário

Neste repositório:

* <a href="#-sobre">📜 Sobre</a> - Breve apresentação

* <a href="#-problemática--motivação">⁉️ Problemática & Motivação</a> - Porquês e objetivos

* <a href="#-dados--pre-processamento">⚙️ Dados & Pré-Processamento</a> - Pipeline de dados

* <a href="#-otimização">🦾 Otimização</a> - Ajuste dos hiperparâmetros

* <a href="#-rede--treinamento">🧠 Rede & Treinamento</a> - Arquitetura implementada

* <a href="#-resultados">📈 Resultados</a> - Desempenho do modelo

* <a href="#-aplicação">🌎 Aplicação</a> - Visualização prática

* <a href="#-uso--código">👨‍💻 Uso & Código</a> - Orientações gerais

* <a href="#-colaboradores">🤝 Colaboradores</a> - Equipe envolvida


## 📜 Sobre

> *History never repeats itself, but it does often rhyme*  
> [Mark Twain](https://pt.wikipedia.org/wiki/Mark_Twain)

Este repositório contém o código do estudo sobre a **previsão do índice S&P 500**, no qual foi desenvolvido um modelo utilizando **células de memória de longo-curto prazo (LSTM) combinadas com mecanismos de atenção**. O modelo passou por otimizações com técnicas de Grid Search e Bayesian Seach, demonstrando um desempenho promissor na previsão de preços de fechamento.

O trabalho foi majoritariamente inspirado no artigo **[Forecasting stock prices with long-short term memory neural network based on attention mechanism](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0227222)** (2020) de Jiayu Qiu, Bin Wang e Changjun Zhou, recebendo honras como um dos cinco melhores projetos de 2023 no curso.

No estudo também foram exploradas aplicações mais práticas do modelo através de técnicas de gestão de banca, avaliando a rentabilidade das previsões em um ambiente controlado e com resultados igualmente promissores.

#### 📕 **[Versão final da monografia](https://github.com/gvmossato/alstm-stock-market/blob/main/misc/Previs%C3%A3o%20do%20%C3%8Dndice%20S%26P%20500%20Utilizando%20LSTM%20e%20Mecanismos%20de%20Aten%C3%A7%C3%A3o.pdf)**

## ⁉️ Problemática & Motivação

O aprofundamento financeiro do Brasil, marcado pela liberalização, desregulamentação e inovação, catalisou um aumento expressivo no número de participantes do mercado financeiro, incluindo também pequenas gestoras e investidores individuais. Entretanto, esse crescimento acompanha também uma ascensão da desinformação, exacerbada pela vulnerabilidade à fraude e pela difusão de análises de fontes não especializadas.

<p align="center"> 
  <img src="https://i.ibb.co/BjvJbqq/decision-making-process.png" alt="decision-making-process" height="190px" />
  <span>       </span>
  <img src="https://i.ibb.co/XkNtbC2/fraud.png" alt="fraud" height="190px" />
</p>

Em paralelo, entende-se que a economia global é, de certo modo, refletido pelo índice S&P 500 devido à vasta operação internacional das empresas listadas nesse. Tal cenário abre espaço para que modelos preditivos possam empoderar o processo de tomada de decisão de investimentos, especialmente para aqueles recém-chegados ao mercado.

Assim, a motivação desse trabalho reside na necessidade de fortalecer o processo decisório no contexto complexo e volátil do mercado de ações. O modelo proposto busca ser uma ferramenta auxiliar, não substituindo, mas sim potencializando o raciocínio estratégico de investidores. Para tal, os recentes desenvolvimentos em mecanismos de atenção em redes neurais são aplicados aqui às séries temporais financeiras oferecendo aos investidores, especialmente aqueles com recursos limitados e acesso tardio a informações, uma fonte a mais para suas decisões.

## ⚙️ Dados & Pré-Processamento

Fluxograma simplificado do pré-processamento dos dados até serem consumidos pelo modelo:

![dados](https://i.ibb.co/tmhPnPk/dados.png)

Conforme a imagem:

1. **Fonte:** o conjunto de dados utilizado nesse projeto é um recorte histórico abrangendo de 03 de janeiro de 1983 a 01 de setembro de 2023, retirado do Yahoo! Finance.

2. **Redução de Ruído:** [transformada wavelet](https://towardsdatascience.com/the-wavelet-transform-e9cfa85d7b34) com a família de funções Coiflets até terceira ordem, a qual é particularmente eficaz na redução de ruído em sinais não estacionários como os de preços de ações.

3. **Normalização:** para assegurar que todas as variáveis tenham o mesmo peso durante o treinamento da rede, aplicamos a normalização [Z-Score](https://www.statology.org/z-score-normalization/). Isso coloca todas as variáveis na mesma escala, neutralizando o efeito de disparidades nas magnitudes de preços e volumes transacionais.

4. **Repartição:** os dados são divididos em três segmentos: treino, validação e teste, na proporção de 95/2.5/2.5. Escolhemos essa divisão para permitir um ajuste fino dos hiperparâmetros (usando o conjunto de validação) e uma avaliação honesta do desempenho do modelo (usando o conjunto de teste). Ao manter a ordem temporal (sem embaralhamento), respeitamos a sequência natural dos eventos no mercado de ações.

5. **Janelamento:** implementamos uma técnica de janela deslizante com tamanho de 20 dias (aproximadamente um mês em dias úteis), que constitui o passo temporal do nosso modelo. Dentro dessa janela, seis séries temporais distintas — abertura, fechamento, máxima, mínima, [fechamento ajustado](https://help.yahoo.com/kb/SLN28256.html) e volume (em quantidades) — são fornecidas ao modelo. Com esses dados ele prediz o valor de fechamento no vigésimo primeiro dia.

## 🦾 Otimização

Como citado, realizamos a tunagem de hiperparâmetros em duas fases distintas: um Grid Search e Bayesian Search. No primeiro, buscamos explorar deterministicamente as redondezas do modelo apresentado no artigo base utilizado. Assim, com um total de **48 combinações únicas** de hiperparâmetros avaliadas utilizando a metodologia de [validação cruzada com k-dobras](https://medium.com/@soumyachess1496/cross-validation-in-time-series-566ae4981ce4) obtivemos **144 execuções distintas** (k=3).

Na sequência, o Bayesian Search foi implementado com base nas combinações de hiperparâmetros mais promissoras do Grid Search, em uma tentativa de refinar a rede. Nesse ponto, decidimos fixar o tamanho do estado oculto em 20, baseando-nos nas descobertas do Grid Search e na sintonia com o tamanho de entrada de dados (20 dias úteis). Ao longo de **100 configurações** de redes, avaliamos e validamos diferentes modelos, totalizando **300 execuções**.

Um resumo dos testes encontra-se na tabela abaixo:

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

A seção de resultados se prolonga por algumas dezenas de páginas da monografia, então, não sendo pertinente trazer todos os resultados, abordamos aqui um recorte conveniente do que alcançamos com o modelo.

Qualitativamente, os resultados no conjunto de teste e o retorno acumulado ao longo do período são bastante fidedignos ao que se observou no mercado para à época:

<p align="center">
  <img src="https://i.ibb.co/71mF1gB/test-results.png" alt="test-results" width="350px" />
  <span>       </span>
  <img src="https://i.ibb.co/mhkQhh6/return-results.png" alt="return-results" width="350px" />
</p>

Optamos ainda por fazer um estudo comparativo do mecanismo de atenção, testando variações desse: um rede sem atenção, uma rede com a atenção como proposta no artigo de referência (benchmark) e a "atenção clássica", proposta no artigo inaugural [Attention Is All You Need](https://arxiv.org/abs/1706.03762):

<p align="center">
  <img src="https://i.ibb.co/TkDq633/attention-results.png" alt="attention-results" width="900px" />
</p>

É observável, portanto, a influência e consequente melhora do desempenho do modelo com o uso do mecanismo clássico. Quando passamos, todavia, a avaliar quantitativamente o modelo frente ao benchmark, encontramos um problema inicial:

|                  | Benchmark    | Modelo         |
|------------------|--------------|----------------|
| Data mínima (teste) | 2019-05-17   | 2022-09-23  |
| Data máxima (teste) | 2019-07-01   | 2023-09-01  |
| Observações        | 31           | 237          |
| Preço Mínimo       | 2751,53      | 3577,03      |
| Preço Máximo       | 2971,41      | 4588,96      |
| Amplitude de Preços| 219,88       | 1011,93      |
| Preço Médio        | 2870,27      | 4095,33      |
| Volatilidade Anualizada | 11,28%  | 17,58%       |

Como sintetiza a tabela anterior, o conjunto de teste do nosso modelo encontrava-se em um mercado muito mais complexo: mais observações com maiores amplitudes de preço e uma volatilidade notadamente superior. Assim, a comparação direta não poderia ser considerada justa, optamos então por normalizar a comparação das métricas de erro por dois métodos, preço médio e amplitude:

|                        | Benchmark    | Modelo       |
|------------------------|--------------|--------------|
| **Métricas**           |              |              |
| RMSE                   | 0,3475       | 19,5238      |
| MAE                    | 0,1935       | 13,9011      |
| $R^2$                  | 0,8783       | 0,9940       |
| **Normalização pelo Preço Médio** |   |              |
| RMSE                   | 0,00012107   | 0,00476734   |
| MAE                    | 0,00006742   | 0,00339437   |
| **Normalização pela Amplitude de Preços** | |        |
| RMSE                   | 0,00158041   | 0,01929363   |
| MAE                    | 0,00088003   | 0,01373717   |

Todavia, mesmo após a normalização, com exceção do $R^2$, não superamos o benchmark. Não obstante, esses resultados forneceram um *insight* valioso: embora não estejamos acertando adequadamente o preço de fechamento, estamos fazendo uma leitura muito satisfatória da tendência. Ora, vamos então aprofundar essa análise:

<p align="center">
  <img src="https://i.ibb.co/C77TdZZ/trend-results.png" alt="trend-results" width="350px" />
  <span>       </span>
  <img src="https://i.ibb.co/Fz7XPkn/confusion-matrix.png" alt="confusion-matrix" width="350px" />
</p>

Constatamos que o modelo de fato parece seguir muito bem as oscilações, prevendo com consistência quando o mercado irá subir ou cair:

* Dado que o índice subiu, acertamos **84,72%** das vezes.
* Dado que o índice caiu, acertamos **80,43%** das vezes.

Frente a esses resultados, optamos então por tentar validar o modelo em uma abordagem um pouco mais prática, donde surge a iniciativa de aplicar estratégias de gestão de banca para operar no mercado tendo em vista as previsões. Foram exploradas diversas estratégias (Martingale, Paroli, D'Alembert, etc.) em um ambiente simulado simplificado, cujas hipóteses foram:

1. Livre de custos
2. Liquidez e volume suficientes no mercado
3. Operações ao preço de fechamento
4. Sem alavancagem
5. **Compra e *short selling* são igualmente complexos**

Destarte, mediante a previsão do modelo e a estratégia de gestão escolhida, o investidor entra comprado ou vendido no ativo, ganhando ou perdendo consoante a variação do índice no período. Os resultados para uma das estratégias mais rentáveis — Paroli — pode ser visto no gráfico abaixo:

<p align="center">
  <img src="https://i.ibb.co/0KZDTdk/paroli-results.png" alt="paroli-results" width="600px" />
</p>

Finalmente, como um todo, o quadro de resultados da gestão de banca fica expresso por:

<p align="center">
  <img src="https://i.ibb.co/DYxjYSs/bet-results.png" alt="bet-results" width="700px" />
</p>

## 🌎 Aplicação

Como um todo, a implementação do projeto pode ser segmentada entre duas grandes frentes: o **modelo**, que compreende basicamente a tudo que fora exposto até aqui, como a rede, o treinamento, as validações técnicas e práticas, etc. Em paralelo, existe ainda a **aplicação**: uma plataforma web, hospedada em um outro [repositório dedicado](https://github.com/gvmossato/alstm-front), para proporcionar uma interface intuitiva o suficiente a fim de permitir que usuários comuns pudessem usufruir das predições do modelo sem conhecimento técnico em programação.

A aplicação permaneceu operante até meados de abril de 2024, sendo incorporados os dados mais recentes disponíveis à época a cada **seis meses**, em treinamentos incrementais automáticos com **15 epochs** e integração ao [IBM Cloudant](https://www.ibm.com/br-pt/products/cloudant), provedor do banco de dados NoSQL utilizado.

## 👨‍💻 Uso & Código

Para executar o **modelo** e definir opções específicas de ajuste de parâmetros e carregamento de pesos de sessões de treinamento anteriores (se necessário), utilize:

```css
poetry run model [-t {grid,bayes}] [-w]
```

Parâmetros opcionais:

* `-t`, `--tuning`: especifica o método de otimização a ser executado. Se não especificado, o ajuste de parâmetros não será realizado. As configurações para cada tipo de ajuste devem ser definidas diretamente no código. Aceita:

  * `grid`: utiliza o Grid Search para otimizar os parâmetros.

  * `bayes`: utiliza o Bayesian Search para otimizar os parâmetros.

* `-w`, `--load-weights`: carrega os pesos salvos da sessão de treinamento mais recente.

  * Padrão: `False` (não carrega os pesos automaticamente).

<br />

Para executar a **aplicação** para realizar previsões com os dados mais recentes disponíveis, sincronização com a nuvem e treinamentos incrementais (se necessário), utilize:

```
poetry run app
```

Note que para integração com o banco de dados será necessário especificar as varáveis de ambiente requeridas pelo serviço de nuvem: `DATABASE`, `CLOUDANT_USERNAME`, `CLOUDANT_PASSWORD` e `CLOUDANT_HOST`.

***

Em relação ao código, a árvore de arquivos do projeto está organizada como:

```
📦alstm_stock_market
 ┣ 📂images
 ┣ 📂logs
 ┣ 📂src
 ┃ ┣ 📂app
 ┃ ┃ ┗ 📜app.py
 ┃ ┣ 📂data
 ┃ ┃ ┗ 📜preprocessor.py
 ┃ ┣ 📂helpers
 ┃ ┃ ┣ 📂calendars
 ┃ ┃ ┃ ┗ 📜us.cal
 ┃ ┃ ┣ 📜plotter.py
 ┃ ┃ ┗ 📜utils.py
 ┃ ┣ 📂manager
 ┃ ┃ ┣ 📜manager.py
 ┃ ┃ ┗ 📜strategies.py
 ┃ ┣ 📂model
 ┃ ┃ ┣ 📂weights
 ┃ ┃ ┣ 📜evaluator.py
 ┃ ┃ ┣ 📜model.py
 ┃ ┃ ┗ 📜params.py
 ┗ 📜run.py
```

Ela está dividida em módulos que concentram as distintas operações do código:

* `📂src/model/`: contém os arquivos referentes ao modelo em si, como arquitetura, hiperparâmetros da rede e métricas de avaliação. Na subpasta `weights` encontram-se os arquivos `.h5` com os pesos do modelo após treinamentos.

* `📂src/app/`: contém a lógica que permite ao modelo ser executado em produção, conforme trabalhado na seção "Aplicação". As rotinas de treinamentos incrementais e comunicação com a nuvem (IBM Cloudant) encontram-se aqui.

* `📂src/manager/`: contém os arquivos referentes à gestão de banca. Subdivisão do código implementada para avaliar o desempenho do modelo em um cenário ainda controlado, mas mais próximo da prática, operando com distintas estratégias frente às previsões.

* `📂src/data/`: contém os arquivos referentes a todo o pipeline de dados exposto, capaz de lidar com cada um dos casos de uso esperados (treinamento inicial, treinamentos adicionais, uso em produção, etc.).

* `📂src/helpers/`: contém os arquivos gerais e de uso compartilhado entre os demais módulos, funções e métodos auxiliares.

<br />

Por fim, a pasta `images` é utilizada para salvar os plots em formatados vetorizados, se desejável, enquanto `logs` armazena registros de execução do módulo de aplicação.

## 🤝 Colaboradores

Este projeto foi desenvolvido por [Gabriel Mossato](https://br.linkedin.com/in/gvmossato) em colaboração com [Paulino Fonseca](https://br.linkedin.com/in/paulinoveloso), ambos à época graduandos sob orientação do [Prof. Dr. Oswaldo Luiz do Valle Costa](https://bv.fapesp.br/en/pesquisador/191/oswaldo-luiz-do-valle-costa), pertencente ao Departamento de Engenharia Elétrica da Escola Politécnica da Universidade de São Paulo (EP-USP).
