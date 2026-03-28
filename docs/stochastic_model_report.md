# Modelo Estocástico de Degradação — Relatório Técnico
### Projeto: Predição de Vida Útil Remanescente (RUL) — NASA C-MAPSS FD001

---

## 1. Dados Utilizados

### Dataset: NASA C-MAPSS — Subset FD001

O dataset C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) é um benchmark clássico de prognostics, disponibilizado pela NASA. O subset **FD001** representa uma condição operacional única (single operating condition) com um único modo de falha.

| Conjunto     | Motores | Linhas de dados | Uso                          |
|:------------|--------:|----------------:|:-----------------------------|
| **Treino**  | 100     | 20.631          | Ajuste de parâmetros do modelo |
| **Teste**   | 100     | ~13.000         | Avaliação e predição de RUL  |
| **RUL real**| 100 valores | —           | Ground truth do conjunto de teste |

- Cada motor começa saudável (ciclo 1) e opera até a falha.
- Cada linha possui **26 colunas**: `unit_id`, `cycle`, 3 configurações operacionais e **21 leituras de sensores** (temperatura, pressão, rotações, etc.).
- 7 sensores foram descartados por apresentarem variância praticamente nula em FD001 (ex: `sensor_1`, `sensor_5`, `sensor_6`).

> [!IMPORTANT]
> **O modelo é treinado exclusivamente nos 100 motores do conjunto de treino** (ciclos completos, do início à falha). O conjunto de teste contém apenas *sequências parciais* — o motor é observado até um ponto desconhecido antes da falha, e o objetivo é predizer quantos ciclos restam.

---

## 2. Pipeline Completa

```
Dados Brutos
    │
    ▼
Pré-processamento
  ├─ Drop de sensores constantes
  ├─ Cálculo do RUL real (apenas no treino)
  └─ Normalização (StandardScaler por sensor)
    │
    ▼
Construção do Health Indicator (HI)
  └─ Regressão Logística por motor (logistic HI)
    │
    ▼
Ajuste do Modelo Estocástico (apenas no treino)
  └─ WLS (Weighted Least Squares) por motor
     ├─ Calcula parâmetros θ⁰ e θ¹ individuais
     └─ Estima o threshold de falha (p75 do HI final)
    │
    ▼
Predição no Conjunto de Teste
  ├─ Reconstrução de parâmetros (combinação convexa - SLSQP)
  └─ Extrapolação até HI < threshold → RUL predito
    │
    ▼
Quantificação de Incerteza
  └─ Bootstrap Monte Carlo (200 amostras) → IC 90%
```

---

## 3. Etapa 1 — Construção do Health Indicator (HI)

### O que é o HI?

O Health Indicator é uma variável escalar que resume o estado de saúde do motor em um único número entre **0 (falha) e 1 (saudável)**. Em vez de usar os 14 sensores diretamente, comprimimos a informação em uma curva de degradação limpa.

### Como é construído?

Para cada motor do treino:
1. **Amostras saudáveis**: primeiros 5 ciclos de vida (motor intacto).
2. **Amostras degradadas**: últimos 5 ciclos antes da falha.
3. Uma **Regressão Logística** é treinada para separar os dois estados.
4. A probabilidade de ser "saudável" em cada ciclo vira o HI: `HI(t) = P(saudável | sensores(t))`.

O HI resultante começa próximo de 1.0 e converge para ~0 à medida que a falha se aproxima — de forma suave e monotônica.

Os **9 sensores com maior correlação monotônica** com o ciclo são selecionados para alimentar a regressão logística (via coeficiente de Spearman), garantindo que apenas os sinais mais informativos entrem no modelo.

---

## 4. Etapa 2 — Modelo de Degradação WLS

### Formulação matemática

O modelo assume que a degradação do HI segue uma **lei de potência**:

```
HI(t) = φ + exp(θ⁰) · t^(θ¹)
```

Onde:
- `φ` (phi) = nível de piso (valor mínimo assintótico do HI). Estimado como a média dos HI finais da frota.
- `θ⁰` = intercepto no espaço log-log (log da escala de degradação).
- `θ¹` = expoente de potência (velocidade de decaimento). Negativo → HI decresce com o tempo.

**Linearização:** aplicando `ln` nos dois lados:

```
ln(HI(t) - φ) = θ⁰ + θ¹ · ln(t)
```

Isso transforma o problema em uma **regressão linear simples** no espaço log-log, resolvida por **WLS (Weighted Least Squares)** com pesos geométricos:

```
w(i) = q^i   onde  q = 1.2
```

Os ciclos mais recentes recebem pesos maiores, pois capturam melhor o comportamento atual do motor — inspirado diretamente no `artigo_exp_1`.

### O que é aprendido no treino?

Para cada um dos **100 motores de treino**, o modelo ajusta individualmente os parâmetros `{θ⁰, θ¹}`. Ao final, temos uma **biblioteca de 100 pares de parâmetros**.

Também é estimado o **threshold de falha** = percentil 75 dos HI finais de todos os motores de treino. Isso garante um critério robusto: motor é considerado falho quando `HI < threshold`.

---

## 5. Etapa 3 — Predição no Teste (Reconstrução de Parâmetros)

Para um motor de **teste** (cuja sequência termina antes da falha):

1. O HI é computado usando o `LogisticHIBuilder` já treinado.
2. Os parâmetros `{θ⁰, θ¹}` do motor de teste são **reconstruídos como combinação convexa** dos 100 motores de treino (otimização SLSQP), minimizando o erro de ajuste nos dados observados.
3. A curva `HI(t)` é **extrapolada para frente** até atingir o threshold.
4. A diferença entre o último ciclo observado e o ciclo predito de falha é o **RUL estimado**.

A extrapolação é limitada a **200 ciclos** para evitar instabilidades numéricas em motores com dados escassos.

---

## 6. Etapa 4 — Quantificação de Incerteza (Monte Carlo Bootstrap)

Para cada motor de teste, **200 amostras bootstrap** são geradas:
- Reamostram os ciclos observados com reposição.
- Reajustam os parâmetros WLS em cada amostra.
- Geram 200 previsões de RUL distintas.

O resultado é uma **distribuição empírica do RUL**, da qual extraímos:
- **Média** → predição pontual reportada.
- **Percentis 5% e 95%** → Intervalo de Confiança de 90%.

---

## 7. Métricas Finais (Conjunto de Teste — 100 motores)

| Métrica                    | Valor        |
|:---------------------------|:-------------|
| **RMSE**                   | 46.9 ciclos  |
| **MAE**                    | 37.4 ciclos  |
| **NASA Score**             | 19.925       |
| **Largura média do IC 90%**| 199.6 ciclos |
| **Threshold de falha**     | HI = 0.019   |
| **phi (φ)**                | 0.000        |
| **Tempo de treino**        | 0.10 s       |
| **Tempo de inferência**    | 6.5 s (Bootstrap) |

> [!NOTE]
> O NASA Score penaliza assimetricamente: predições tardias (subestimar o RUL) recebem penalidade muito maior que predições antecipadas. Um score de ~20k é moderado — o modelo tende a subestimar ligeiramente (como mostra o bias de -30.9 ciclos nos resíduos).

---

## 8. Explicação de Cada Imagem

---

### Fig. 1 — `hi_fleet_trajectories.png` — Trajetórias de HI da Frota

**O que mostra:** As curvas de HI ao longo do tempo para 20 motores selecionados aleatoriamente do conjunto de **treino**.

**Como ler:**
- Eixo X → ciclo operacional (tempo de uso do motor).
- Eixo Y → Health Indicator (1 = saudável, 0 = falha).
- **Cor da curva** → vida total do motor (amarelo = vida curta ~140 ciclos, azul escuro = vida longa ~330 ciclos).
- **Linha vermelha tracejada** → threshold de falha automático (HI = 0.019), estimado como p75 dos HI finais.

**O que aprender:**
- Todos os motores partem de HI ≈ 1.0 e convergem para próximo de 0 ao falhar — o HI logístico captura bem a degradação monotônica.
- A dispersão horizontal (diferentes comprimentos de vida) é esperada: FD001 tem motores que vivem de ~130 a ~360 ciclos.
- O trecho inicial ruidoso (HI oscilando entre 0.9–1.0) é ruído de sensor normal na fase saudável.
- A queda íngreme no terço final da vida é a **assinatura da degradação acelerada** — exatamente o que o modelo WLS captura.

> **Dado:** 100% treino. Não há nenhum dado de teste neste gráfico.

---

### Fig. 2 — `hi_degradation_detail.png` — Ajuste WLS por Motor (Treino)

**O que mostra:** Para 6 motores individuais do **treino**, sobrepõe a curva HI observada com a curva WLS ajustada e o threshold de falha.

**Como ler:**
- Linha cinza contínua → HI observado (calculado pelo logistic HI builder).
- Linha roxa tracejada → curva WLS ajustada: `HI(t) = φ + exp(θ⁰) · t^(θ¹)`.
- Linha pontilhada vermelha → threshold de falha (HI = 0.019).

**O que aprender:**
- O ajuste WLS (roxo) captura muito bem a queda final do HI — especialmente nos últimos 30–40% da vida do motor.
- No início da vida (fase saudável), o modelo não encosta bem na curva, mas isso não importa: a predição de RUL só precisa do comportamento na fase de degradação.
- Motores com degradação mais suave (ex: Engine 92, ~350 ciclos) têm θ¹ menos negativo que motores com queda abrupta (ex: Engine 18, ~190 ciclos).
- O ponto onde a curva roxa cruza a linha vermelha é o ponto de falha predito — base para calcular o RUL.

> **Dado:** 100% treino. Estas são as curvas aprendidas durante o `.fit()`.

---

### Fig. 3 — `rul_actual_vs_predicted.png` — Scatter Real vs. Predito (Teste)

**O que mostra:** Scatter plot com 100 pontos — um por motor de **teste** — comparando o RUL verdadeiro (eixo X) com o RUL predito pelo modelo (eixo Y).

**Como ler:**
- **Diagonal tracejada preta** → predição perfeita (y = x). Quanto mais próximos os pontos, melhor.
- **Faixa violeta** → zona de ±10% de erro. Pontos ali são muito bons.
- **Faixa cinza** → zona de ±20% de erro.
- **Cor dos pontos** → magnitude do erro absoluto (amarelo = erro pequeno, roxo/escuro = erro grande).

**O que aprender:**
- Para **RUL baixo (0–40 ciclos)** os pontos ficam colados à diagonal → o modelo é excelente quando o motor já está claramente degradado.
- Para **RUL alto (80–150 ciclos)** a nuvem desce abaixo da diagonal → o modelo **subestima** o RUL restante (prediz que o motor vai falhar antes do esperado). Isso explica o bias médio de -30.9 no histograma de resíduos.
- RMSE = 46.9 ciclos e MAE = 37.4 ciclos — para um dataset onde os RUL variam de 0 a ~150, isso representa ~25–30% de erro médio relativo.

> **Dado:** 100% teste. Cada ponto = 1 motor de teste com seu RUL real (do arquivo `RUL_FD001.txt`) vs. predição do modelo.

---

### Fig. 4 — `rul_per_engine_with_ci.png` — RUL por Motor com IC 90% (Teste)

**O que mostra:** Para os primeiros 30 motores de teste, mostra a série temporal de RUL predito (violeta) vs. RUL real (laranja), com a faixa de confiança de 90% em azul claro.

**Como ler:**
- Eixo X → índice do motor de teste (1 a 30).
- Eixo Y → RUL em ciclos.
- Linha violeta com círculos → predição média (Bootstrap).
- Linha laranja tracejada com quadrados → RUL real (ground truth).
- Faixa azul → IC 90% gerado por Monte Carlo Bootstrap (200 resamples).

**O que aprender:**
- A faixa de IC é **muito larga** (média de ~200 ciclos de largura), especialmente no início onde o motor ainda tem muito RUL. Isso reflete a incerteza real do modelo: com poucos dados de degradação observados, é difícil saber com precisão quanto falta.
- O modelo tende a **subpredizer**: a linha violeta fica abaixo da laranja na maioria dos motores — confirmando o bias observado no histograma.
- Para motores com RUL real baixo (ex: motor 20 com ~16 ciclos), a predição converge bem.
- A largura do IC seria mais útil em produção: mesmo que o ponto central erre, a sinalização "atenção nos próximos 0–200 ciclos" já é valiosa para manutenção preditiva.

> **Dado:** 100% teste. Os valores "Actual RUL" vêm do arquivo `RUL_FD001.txt` fornecido pela NASA.

---

### Fig. 5 — `residuals_analysis.png` — Análise de Resíduos (Teste)

**O que mostra:** Dois painéis de diagnóstico estatístico dos erros de predição.

#### Painel Esquerdo — Histograma + KDE dos Resíduos

- Resíduo = Predito − Real. Resíduo negativo = subestimou o RUL.
- **Média = -30.9 ciclos** → bias sistemático de subestimação.
- **Std = 35.2 ciclos** → dispersão dos erros.
- A distribuição está deslocada para a **esquerda** (maioria dos pontos com resíduo entre -80 e 0), confirmando que o modelo tende a ser conservador (prediz falha antes do esperado).

#### Painel Direito — QQ-Plot Normal

- Compara os quantis empíricos dos resíduos com os quantis de uma distribuição normal teórica.
- **R² = 0.978** → os resíduos seguem distribuição aproximadamente normal, com pequenos desvios nas caudas (pontos se afastando da linha nos extremos).
- Isso é um bom sinal: significa que o modelo de incerteza Bootstrap pode ser calibrado futuramente com uma distribuição Normal paramétrica.

> **Dado:** 100% teste. Os resíduos são calculados comparando as 100 predições de RUL com os 100 valores reais do arquivo da NASA.

---

### Fig. 6 — `wls_loglog_fit.png` — Ajuste WLS no Espaço Log-Log (Treino)

**O que mostra:** Para 6 motores de treino, rende o ajuste WLS em escala logarítmica dupla (log-log), evidenciando a estrutura de lei de potência.

**Como ler:**
- Eixo X (log) → ciclo em escala log. Os ciclos iniciais ficam comprimidos à esquerda; os finais, expandidos à direita.
- Eixo Y (log) → HI em escala log. Valores próximos de 1 ficam comprimidos no topo; valores pequenos (degradação avançada) ficam na parte inferior.
- **Pontos coloridos** → observações de HI. Cor representa o peso WLS (mais claro = mais antigo, mais escuro = mais recente e mais influente).
- **Linha roxa** → reta WLS ajustada. Em escala log-log, a lei de potência vira uma reta — confirma que o modelo é aplicável.
- **Linha vermelha horizontal** → threshold de falha.

**O que aprender:**
- A linearidade no espaço log-log confirma que a escolha do modelo de lei de potência `HI(t) = exp(θ⁰) · t^(θ¹)` é matematicamente adequada para estes dados.
- O expoente θ¹ (inclinação da reta) varia entre motores — motores com decaimento mais abrupto têm θ¹ mais negativo.
- Os pontos mais pesados (escuros, ciclos finais) ficam mais próximos da reta ajustada devido ao esquema de pesos geométricos — o modelo prioriza acertar o comportamento recente.

> **Dado:** 100% treino.

---

## 9. Resumo do que é Treino vs. Teste

| Imagem                        | Conjunto usado | O que representa                              |
|:------------------------------|:--------------:|:----------------------------------------------|
| Fig. 1 — Fleet HI Trajectories | **Treino**    | HI calculado para motores completos (0 → falha) |
| Fig. 2 — WLS Fit per Engine    | **Treino**    | Parâmetros θ⁰, θ¹ ajustados por motor         |
| Fig. 3 — Actual vs. Predicted  | **Teste**     | Predições finais de RUL vs. ground truth       |
| Fig. 4 — RUL with CI           | **Teste**     | Incerteza Bootstrap nos primeiros 30 motores   |
| Fig. 5 — Residuals Analysis    | **Teste**     | Análise estatística dos erros de predição      |
| Fig. 6 — WLS Log-Log Fit       | **Treino**    | Linearidade no espaço log-log por motor        |

---

## 10. Limitações Observadas

1. **Bias de subestimação (-30.9 ciclos em média):** o modelo tende a predizer falha antecipada. Isso acontece porque a lei de potência é ajustada nos dados de treino inteiros, mas só se observa uma fração nos testes — e a curva extrapolada cruza o threshold "cedo demais".
2. **IC muito largo:** a largura média de ~200 ciclos do IC 90% indica alta incerteza. Isso é esperado com Bootstrap simples — para estreitar, seria necessário incorporar prior bayesiano ou aumentar `n_bootstrap`.
3. **Cap de 200 ciclos:** RULs verdadeiros maiores que 200 são automaticamente truncados, o que contribui para o bias em motores com longa vida residual.
4. **FD001 apenas:** este modelo foi treinado e avaliado somente no subset FD001 (condição operacional única). A extensão para FD002/FD003/FD004 (múltiplas condições) requer ajustes na etapa de construção do HI.
