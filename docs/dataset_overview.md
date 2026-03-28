# NASA C-MAPSS Dataset Overview

Este documento apresenta uma introdução detalhada ao **NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)**, um dos bancos de dados abertos mais populares e extensamente utilizados na área de Prognostics and Health Management (PHM) e manutenção preditiva para predição de vida útil remanescente (RUL).

## 1. O Que Representa o Banco de Dados?

O banco C-MAPSS contém simulações de degradação ("run-to-failure") de motores turbofan de aeronaves. A simulação, construída pela NASA, não é gerada diretamente de sensores reais montados em uma asa, mas sim de um modelo termodinâmico de altíssima fidelidade. Esse simulador introduz perfis operacionais e injeta perfis de dano progressivo nas partes críticas do motor (como HPC - High Pressure Compressor, HPT - High Pressure Turbine, fan, etc.).

O objetivo principal de disponibilizar esse banco é permitir a pesquisa e a validação de algoritmos preditivos (data-driven models, machine learning, deep learning, modelos estatísticos/de degradação) para descobrir **quantos ciclos de voo restam antes que o motor atinja uma falha funcional ou limite de utilidade** – o que chamamos de RUL.

## 2. Como os Dados São Organizados?

O repositório padrão do C-MAPSS é dividido em quatro sub-conjuntos principais (FD001 a FD004), diferenciados com base na complexidade das condições operacionais e nos modos de falha simulados. Cada vez que um voo (ou simulação de ciclo) ocorre, o motor degrada um pouco mais, e todos os sensores medem o estado daquele motor naquele tempo `t`.

### Entendendo os Sub-conjuntos (FD00x)

| Dataset | Modos de Falha | Condições Operacionais (Regimes) | Treino (Unidades) | Teste (Unidades) |
| :--- | :---: | :---: | :---: | :---: |
| **FD001** | 1 (Degradação na HPC) | 1 (Nível do Mar / Regime único) | 100 | 100 |
| **FD002** | 1 (Degradação na HPC) | 6 (Múltiplas altitudes e Mach) | 260 | 259 |
| **FD003** | 2 (HPC + fan) | 1 (Nível do Mar / Regime único) | 100 | 100 |
| **FD004** | 2 (HPC + fan) | 6 (Múltiplas altitudes e Mach) | 249 | 248 |

Dessa forma:
- O **FD001** é o mais simples e, via de regra, considerado a linha base (baseline) nos artigos.
- O **FD004** é o cenário mais complexo, combinando múltiplos modos de falha misturados de forma oculta e dados fortemente corrompidos (ou influenciados) por 6 regimes de voo totalmente distintos (que mascaram a degradação natural dos sensores).

### Significado de Cada Linha

Cada linha nos arquivos de texto (`train_FD00x.txt` ou `test_FD00x.txt`) representa o registro do motor (captura de vários sensores da aeronave) em **exatamente 1 ciclo de operação** (aproximadamente 1 voo completo, do takeoff ao landing). Conforme as linhas avançam para uma mesma aeronave/unidade, o desgaste aumenta intrinsecamente, mesmo que de maneira não visualizada diretamente nos valores numéricos ruidosos de um único sensor isolado.

## 3. As Colunas de Dados

Cada arquivo possui normalmente 26 colunas. Elas não possuem cabeçalho no .txt original, mas a documentação da NASA instrui que as variáveis representam o seguinte:

- **1. Unit ID (`unit_id`):** O identificador único de qual "motor" estamos medindo (ex.: motor 1, 2, ..., n). 
- **2. Cycle (`time_in_cycles`):** O tempo contínuo de operação. O registro n° 1 de uma unidade tem cycle=1, o próximo tem cycle=2, e assim por diante.
- **3 a 5. Operational Settings (OS_1, OS_2, OS_3):** Estes são controles ou condições ambientais em que o voo aconteceu (Altitute, Número de Mach e Posição do Acelerador do Manete). No FD001, eles são basicamente constantes ou com ruído mínimo, pois é "1 condição operacional". No FD002/FD004, essas três variáveis agrupam os 6 regimes operacionais de voo. Elas introduzem alta não-linearidade e deslocamento de valor escalar (offsets) nos sensores.
- **6 a 26. Sensores Físicos (s_1 a s_21):** São 21 valores representando leituras vitais termodinâmicas vindas do CMAPSS (Temperatura em diferentes estágios, Pressão nos dutos, Velocidades físicas e corrigidas do core/fan, Consumo de combustível e etc). Importante: **muitos sensores apresentam valor perfeitamente constante durante toda a vida do motor**, o que exige métodos de feature selection / eliminação de ruídos e variáveis sem variância.

## 4. Diferença Entre Treino, Teste e Vetor de RUL

A simulação e a forma competitiva de utilizar o C-MAPSS exigem uma arquitetura forte de validação, daí a necessidade da tripartição conceptual dos arquivos fornecidos pela NASA.

### Conjunto de Treino (`train_FD00x.txt`)
Contém historicos completos "run-to-failure" de dezenas de motores.
Isso significa que o registro da Unidade X ali listada abrange desde o ciclo 1 até o ciclo final, digamos, 192 (ou o número N que for). No último ciclo registrado daquela unidade, a NASA atesta que o motor falhou/ultrapassou limites operacionais estritos e **"quebrou"**. Portanto, a Vida Útil Remanescente (RUL) é 0 no último registro. No primeiro registro (ciclo 1), a RUL equivale a (Tamanho Total - 1).

### Conjunto de Teste (`test_FD00x.txt`)
Contém históricos truncados de motores adicionais.
Em vez dos motores em teste rodarem até quebrarem, eles funcionaram até um ponto desconhecido `t` "aleatório" (simulando a vida real: uma frota que ainda tem aeronaves em operação em diferentes estágios de desgaste). O motor 1 rodou até o ciclo 31. O motor 2 operou até o 145. O motor 3 operou até o 99, etc. O arquivo final de teste termina nos ciclos truncados. A meta é descobrir **quanto tempo a mais de vida cada uma destas unidades de fato sobreviveria se continuasse operando daquele ciclo truncado para o futuro.** 

### O Vetor Real de RUL (`RUL_FD00x.txt`)
Para podermos auferir se o modelo de machine learning acertou na sua predição, a NASA revelou as respostas verdadeiras de prognóstico no vetor de `RUL`. Este arquivo contém apenas um número por linha — cada linha referente àquela Unidade trunca do dataset de teste correspondente. Se a Unidade 1 de teste, que foi cortada no ciclo 31, tem como resposta verdadeira "112" no RUL vector, isso significa que ela estava a exatos 112 ciclos da falha catastrófica quando foi feito o último registro. O desafio empírico e prático de submissões PHM (Prognostics and Health Management) baseia-se muitas vezes na métrica RMSE contra os dados listados pontualmente em RUL_FD00x.

## 5. Por Que Esse Banco é Tão Utilizado?

Para pesquisa em RUL, a comunidade PHM lida frequentemente com severa restrição de disponibilidade de dados críticos da indústria aeronáutica. Os fabricantes de motores comerciais genuinamente operacionais raras vezes correm os equipamentos com defeitos para avistar quebra e relutar a compartilhar esse tipo de métrica publicamente. Sendo derivado de modelagem física autêntica com métricas da FAA, contendo perturbações e ruído que simulam perfeitamente o caos do voo aerodinâmico (ruídos nas leituras de sensor, falhas não-explicitas por regimes dinâmicos), os conjuntos C-MAPSS tornaram-se onipresentes como laboratórios _benchmark_ padrão acadêmico e teste industrial. Um algoritmo que é perfeitamente apto para "descobrir" o tempo de quebra via sensor no motor de C-MAPSS costuma ser, em base abstrata e conceitual, transferível e comprovadamente estável.
