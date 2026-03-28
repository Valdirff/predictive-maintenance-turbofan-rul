# Metodologia Exponencial para Predição de RUL

Este documento consolida a abordagem metodológica focada em degradação exponencial (ou baseada em filtragem de estados em curva não-linear) inspirada e detalhada pelos artigos do diretório `artigos/` que possuem a tag `exp` no nome.

## 1. Visão Geral da Abordagem

Ao contrário de métodos de *black-box* puros (como Deep Learning, LSTM ou CNN puro para regressão direta do RUL), a abordagem **exponencial / state-filtering** visa ser *física-inspirada* e **explicável**. O princípio central é que o desgaste de componentes rotativos (como um motor turbofan) se acumula vagarosamente no começo da vida útil e acelera (frequentemente de forma exponencial) próximo à falha.

A metodologia se divide tipicamente em duas grandes macro-etapas:
1. **Construção do Indicador de Saúde (Health Indicator - HI):** Fusão de múltiplos sensores crus em uma única curva 1D (unidimensional) de degradação.
2. **Modelagem Estocástica da Degradação e Predição:** Ajustar um modelo matemático (como o exponencial) a este HI e extrapolar a curva para o futuro até cruzar um *Threshold* de falha pré-definido.

---

## 2. Construção do Health Indicator (HI)

A leitura multi-variada crua do motor (como temperatura, pressão, velocidade do fan) é muito ruidosa e mal-comportada. Os artigos propõem técnicas estruturadas para encontrar o HI.

### 2.1 Mapeamento e Seleção de Sensores
Primeiro avalia-se a qualidade dos sensores através de métricas como:
- **Monotonicidade:** A curva vai consistentemente numa mesma direção (subindo ou descendo com o tempo)?
- **Robustez/Tendência:** A proporção de tendência vs. ruído é aceitável?
- **Prognosabilidade (Prognosability):** Os diferentes motores ao falharem chegam em uma variância aceita ou terminam num limiar estatisticamente em comum?

Apenas sensores que passam nesses filtros (altos scores) são selecionados para fusão.

### 2.2 Fusão (Data Fusion)
Os artigos evidenciam que manter o modelo simples requer fundir esses sinais. Métricas comuns encontradas nos artigos `exp` iteram entre:
- **Abordagem C-MAPSS Artigo 1 (GP-based / Non-linear):** Constrói um HI usando Algoritmo Genético/Programação Genética (GP) combinando operações não lineares (`sin`, `cos`, multiplicações) para forçar as curvas de degradação dos motores de treino a seguirem o mesmo padrão.
- **Abordagem C-MAPSS Artigo 2 (Logistic Regression / Probabilística):** Estima um HI entre `1` (Saudável) e `0` (Falha) recolhendo amostras seguras do início da vida ("saudável") e logo antes de quebrar ("falha") para todos os motores de treino. Ajusta-se uma Regressão Logística (ou Generalised Linear Model - GLM) transformando dados do sensor em uma probabilidade/escala de saúde.

Ambas abordagens garantem matematicamente que a saúde do motor inicie estável / alta e degrade até um patamar ou ponto conhecido (baseline failure).

---

## 3. A Curva de Degradação Exponencial

Com o HI construído para cada motor, o sinal perdeu sua interpretabilidade crua, mas ganhou um aspecto de **curva de degradação**.

### A Hipótese de Degradação
Assume-se que o HI obedece a um processo mecânico de desgaste clássico, que muitas vezes é assintótico ou estritamente convexo (exponencial).
Uma formulação comum é a família geométrica/exponencial descrita como:

$$ HI(t) = a \cdot e^{b \cdot t} + c + \epsilon $$

Onde:
- $t$ é o tempo / ciclo operacional.
- $a, b, c$ são parâmetros da física de desgaste/condição inicial do componente. O valor $b$ (a taxa de decaimento exponencial) governará a aceleração da quebra.
- $\epsilon$ é o ruído do sensor ou do processo daquele instante.

## 4. Como a RUL é Calculada

Tendo um modelo ajustado aos dados observados desde o ciclo 0 até o ciclo atual $t_{current}$, a predição da RUL obedece um raciocínio de extrapolação explícita:

1. **Definição de Threshold de Falha ($Failure Threshold$ - FT):** Escolhe-se estatisticamente o ponto em que o HI decreta a quebra do componente. Pode ser $HI = 0$ (se for um score de sobrevivência normalizado) ou a média empírica do HI no momento da quebra para os dados de treino ($\mu \pm 2\sigma$).
2. **Extrapolação:** Resolve-se a equação exponencial algebricamente para encontrar qual ciclo $t_{fail}$ fará a função cruzar o Threshold.
   $$ t_{fail} = \frac{\ln \left( \frac{FT - c}{a} \right)}{b} $$
3. **Equação RUL:**
   $$ RUL = t_{fail} - t_{current} $$

### Variações Sofisticadas (Ex. Artigo 2)
No artigo 2 referenciado, o cálculo pontual algébrico é complementado usando o **Unscented Kalman Filter (UKF)** e conceitos bayesianos. Em vez de uma curva engessada estática, os parâmetros $a, b$ são tratados como *Variáveis de Estado* que evoluem a cada novo lote de dados. O UKF computa não apenas o valor médio da RUL, mas também quantifica incerteza (Intervalo de Confiança), rodando milhares de simulações de trajetória estocástica para oferecer previsões robustas (90% Confidence Interval).

---

## 5. Vantagens e Limitações

### Vantagens
- **Alta Transparência (Explainability):** Um mecânico chefe ou engenheiro pode "ver" o parâmetro $b$ aumentando vertiginosamente, provando a aceleração do dano do equipamento em tempo real, gerando confiança na inteligibilidade e estabilidade do prognóstico, superando as tradicionais "caixas-pretas".
- **Necessita de Menos Dados:** Ajustar a curva requer muitos menos parâmetros / amostras comparado a uma deep CNN ou LSTM pesada.
- **Quantificação de Incerteza Matemática:** Filtros Bayesianos embutidos a extrapolação exponencial conferem naturalmente intervalos formais de confiança da RUL a cada passo do tempo `t`.

### Limitações e Fraquezas Teóricas
- **Rigidez do Modelo (Underfitting):** Sistemas operando com defeitos múltiplos e em regimes não lineares e trocas de altitude frequentes (como o dataset FD004) produzem curvas altamente caóticas para um fit exponencial puro modelar, derrubando severamente o _score_ da RUL. Modelos físicos base são ineficazes nesses casos sem um tratamento pesado nos dados.
- **Regime Inicial Plano:** Se um motor opera perfeitamente sem dano durante seus 100 primeiros ciclos, a curva do HI será praticamente uma "linha reta" em $1.0$. Qualquer tentativa de "Fitar" uma curva exponencial sobre uma linha puramente horizontal resulta matematicamente numa indeterminação extrema dos parâmetros matemáticos resultando na "explosão" do RUL que dita que a máquina viverá "no infinito". Esta "cegueira precoce" dita que o modelo estocástico prediz pobremente antes da fase inicial de degradação expor seus contornos convexos.
