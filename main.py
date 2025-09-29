# Instalar bibliotecas (caso necessário)
# pip install pulp kagglehub pandas

import pandas as pd
import kagglehub
import pulp

# 1) Baixar dataset (custos de seguro de saúde)
path = kagglehub.dataset_download(
    "mosapabdelghany/medical-insurance-cost-dataset"
)
df = pd.read_csv(path + "/insurance.csv")

# 2) Suponha que queremos escolher quais clientes aceitar
#    para maximizar receita, respeitando um limite de custo médio

# Receita hipotética = valor do seguro
df["receita"] = 5000  # simplificação
df["custo"] = df["charges"]  # coluna real do dataset

# 3) Definir modelo de otimização
clientes = list(df.index)
x = pulp.LpVariable.dicts("x", clientes, lowBound=0, upBound=1, cat="Binary")

prob = pulp.LpProblem("SelecaoSeguros", pulp.LpMaximize)

# Objetivo: maximizar receita total
prob += pulp.lpSum(df.loc[i, "receita"] * x[i] for i in clientes)

# Restrição: custo médio ≤ 15.000
prob += pulp.lpSum(df.loc[i, "custo"] * x[i] for i in clientes) <= 15000 * pulp.lpSum(x[i] for i in clientes)

# 4) Resolver
prob.solve()

# 5) Mostrar resultado
selecionados = [i for i in clientes if x[i].value() == 1]
print("=== CONSULTA 1: SELEÇÃO DE CLIENTES PARA MAXIMIZAR RECEITA ===")
print(f"Clientes escolhidos: {len(selecionados)}")
print(f"Receita total: ${pulp.value(prob.objective):,.2f}")
if selecionados:
    custo_medio_selecionados = df.loc[selecionados, "custo"].mean()
    print(f"Custo médio dos selecionados: ${custo_medio_selecionados:,.2f}")

print("\n" + "="*80 + "\n")

# ===================================================================
# CONSULTA 2: OTIMIZAÇÃO DE PORTFÓLIO DE CLIENTES POR REGIÃO
# ===================================================================

# Criar variável de região simulada baseada no IMC
df["regiao"] = df["bmi"].apply(lambda x: "Norte" if x < 25 else "Sul" if x < 30 else "Centro")

# Modelo para distribuir clientes por região, minimizando risco total
prob2 = pulp.LpProblem("PortfolioRegiao", pulp.LpMinimize)

# Variáveis: quantos clientes aceitar de cada região
regioes = df["regiao"].unique()
y = pulp.LpVariable.dicts("clientes_regiao", regioes, lowBound=0, cat="Integer")

# Objetivo: minimizar custo total ponderado por região
custos_por_regiao = df.groupby("regiao")["custo"].mean().to_dict()
prob2 += pulp.lpSum(custos_por_regiao[r] * y[r] for r in regioes)

# Restrições:
# - Pelo menos 50 clientes no total
prob2 += pulp.lpSum(y[r] for r in regioes) >= 50

# - Máximo 30% de clientes de qualquer região
total_clientes = pulp.lpSum(y[r] for r in regioes)
for r in regioes:
    prob2 += y[r] <= 0.3 * total_clientes

# - Pelo menos 10 clientes por região
for r in regioes:
    prob2 += y[r] >= 10

prob2.solve()

print("=== CONSULTA 2: OTIMIZAÇÃO DE PORTFÓLIO POR REGIÃO ===")
for r in regioes:
    print(f"Clientes da região {r}: {int(y[r].value())}")
print(f"Custo total estimado: ${pulp.value(prob2.objective):,.2f}")

print("\n" + "="*80 + "\n")

# ===================================================================
# CONSULTA 3: PLANEJAMENTO DE CAPACIDADE HOSPITALAR
# ===================================================================

# Simular demanda por tipo de serviço baseada em características dos clientes
df["demanda_emergencia"] = ((df["age"] > 50) | (df["smoker"] == "yes")).astype(int)
df["demanda_cirurgia"] = ((df["bmi"] > 35) | (df["age"] > 60)).astype(int)
df["demanda_consulta"] = 1  # todos precisam de consultas

# Modelo para determinar capacidade ideal de cada tipo de serviço
prob3 = pulp.LpProblem("CapacidadeHospitalar", pulp.LpMinimize)

# Variáveis: capacidade de cada serviço
cap_emergencia = pulp.LpVariable("cap_emergencia", lowBound=0, cat="Integer")
cap_cirurgia = pulp.LpVariable("cap_cirurgia", lowBound=0, cat="Integer")
cap_consulta = pulp.LpVariable("cap_consulta", lowBound=0, cat="Integer")

# Custos fixos por unidade de capacidade
custo_emergencia = 10000
custo_cirurgia = 15000
custo_consulta = 2000

# Objetivo: minimizar custo total de capacidade
prob3 += (cap_emergencia * custo_emergencia + 
          cap_cirurgia * custo_cirurgia + 
          cap_consulta * custo_consulta)

# Restrições: atender toda a demanda
prob3 += cap_emergencia >= df["demanda_emergencia"].sum()
prob3 += cap_cirurgia >= df["demanda_cirurgia"].sum()
prob3 += cap_consulta >= df["demanda_consulta"].sum()

# Restrição de recursos: total não pode exceder um orçamento
prob3 += (cap_emergencia * custo_emergencia + 
          cap_cirurgia * custo_cirurgia + 
          cap_consulta * custo_consulta) <= 2000000

prob3.solve()

print("=== CONSULTA 3: PLANEJAMENTO DE CAPACIDADE HOSPITALAR ===")
print(f"Capacidade recomendada:")
print(f"  - Emergência: {int(cap_emergencia.value())} unidades")
print(f"  - Cirurgia: {int(cap_cirurgia.value())} unidades") 
print(f"  - Consulta: {int(cap_consulta.value())} unidades")
print(f"Custo total de capacidade: ${pulp.value(prob3.objective):,.2f}")

print("\n" + "="*80 + "\n")

# ===================================================================
# CONSULTA 4: ANÁLISE DE ALOCAÇÃO POR FAIXA ETÁRIA
# ===================================================================

# Criar faixas etárias para análise
df["faixa_etaria"] = pd.cut(df["age"], bins=[0, 30, 50, 100], labels=["Jovem", "Adulto", "Senior"])

# Modelo para alocar recursos por faixa etária
prob4 = pulp.LpProblem("AlocacaoFaixaEtaria", pulp.LpMaximize)

# Variáveis: número de clientes a aceitar por faixa etária
faixas = df["faixa_etaria"].dropna().unique()
clientes_faixa = pulp.LpVariable.dicts("clientes", faixas, lowBound=0, cat="Integer")

# Receita média por faixa etária (simulada)
receita_faixa = {"Jovem": 4000, "Adulto": 5000, "Senior": 6000}

# Custo médio por faixa etária
custo_faixa = df.groupby("faixa_etaria", observed=True)["charges"].mean().to_dict()

# Objetivo: maximizar lucro total
prob4 += pulp.lpSum((receita_faixa[f] - custo_faixa[f]) * clientes_faixa[f] for f in faixas)

# Restrições:
# - Não exceder o número disponível de clientes por faixa
disponivel_faixa = df["faixa_etaria"].value_counts().to_dict()
for f in faixas:
    prob4 += clientes_faixa[f] <= disponivel_faixa[f]

# - Total de clientes deve ser pelo menos 100
prob4 += pulp.lpSum(clientes_faixa[f] for f in faixas) >= 100

# - Pelo menos 20% de cada faixa etária
total_clientes = pulp.lpSum(clientes_faixa[f] for f in faixas)
for f in faixas:
    prob4 += clientes_faixa[f] >= 0.2 * total_clientes

prob4.solve()

print("=== CONSULTA 4: ANÁLISE DE ALOCAÇÃO POR FAIXA ETÁRIA ===")
for f in faixas:
    if clientes_faixa[f].value() is not None:
        lucro_unit = receita_faixa[f] - custo_faixa[f]
        print(f"Faixa {f}:")
        print(f"  - Clientes aceitos: {int(clientes_faixa[f].value())}")
        print(f"  - Receita unitária: ${receita_faixa[f]:,.2f}")
        print(f"  - Custo unitário: ${custo_faixa[f]:,.2f}")
        print(f"  - Lucro unitário: ${lucro_unit:,.2f}")

if pulp.value(prob4.objective) is not None:
    print(f"Lucro total estimado: ${pulp.value(prob4.objective):,.2f}")

print("\n" + "="*80 + "\n")

# ===================================================================
# CONSULTA 5: ALOCAÇÃO ÓTIMA DE RECURSOS DE MARKETING
# ===================================================================

# Definir canais de marketing e suas eficiências
canais = ["Digital", "TV", "Radio", "Impresso"]
custo_por_lead = {"Digital": 50, "TV": 200, "Radio": 75, "Impresso": 100}
conversao_rate = {"Digital": 0.15, "TV": 0.08, "Radio": 0.10, "Impresso": 0.05}

# Modelo para alocar orçamento de marketing
prob5 = pulp.LpProblem("AlocacaoMarketing", pulp.LpMaximize)

# Variáveis: orçamento por canal
orcamento = pulp.LpVariable.dicts("orcamento", canais, lowBound=0)

# Objetivo: maximizar número total de novos clientes
prob5 += pulp.lpSum((orcamento[c] / custo_por_lead[c]) * conversao_rate[c] for c in canais)

# Restrições:
# - Orçamento total limitado
prob5 += pulp.lpSum(orcamento[c] for c in canais) <= 500000

# - Mínimo de investimento por canal
for c in canais:
    prob5 += orcamento[c] >= 20000

# - Máximo 40% do orçamento em qualquer canal
total_orcamento = pulp.lpSum(orcamento[c] for c in canais)
for c in canais:
    prob5 += orcamento[c] <= 0.4 * total_orcamento

prob5.solve()

print("=== CONSULTA 5: ALOCAÇÃO ÓTIMA DE RECURSOS DE MARKETING ===")
total_novos_clientes = 0
for c in canais:
    if orcamento[c].value() is not None:
        leads = orcamento[c].value() / custo_por_lead[c]
        novos_clientes = leads * conversao_rate[c]
        total_novos_clientes += novos_clientes
        print(f"Canal {c}:")
        print(f"  - Orçamento: ${orcamento[c].value():,.2f}")
        print(f"  - Leads gerados: {leads:.0f}")
        print(f"  - Novos clientes: {novos_clientes:.0f}")

print(f"Total de novos clientes: {total_novos_clientes:.0f}")

print("\n" + "="*80 + "\n")

# ===================================================================
# CONSULTA 6: OTIMIZAÇÃO DE REDE DE ATENDIMENTO
# ===================================================================

# Simular localização de clientes e centros de atendimento
import random
random.seed(42)

n_centros = 5
n_clientes_sample = 50  # usar subset para exemplo

# Coordenadas simuladas
centros = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(n_centros)]
clientes_coords = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(n_clientes_sample)]

# Calcular distâncias
def distancia(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

distancias = {}
for i in range(n_clientes_sample):
    for j in range(n_centros):
        distancias[(i, j)] = distancia(clientes_coords[i], centros[j])

# Modelo de localização-alocação
prob6 = pulp.LpProblem("RedeAtendimento", pulp.LpMinimize)

# Variáveis:
# z[j] = 1 se centro j está aberto
z = pulp.LpVariable.dicts("centro_aberto", range(n_centros), cat="Binary")
# w[i,j] = 1 se cliente i é atendido pelo centro j
w = {}
for i in range(n_clientes_sample):
    for j in range(n_centros):
        w[(i, j)] = pulp.LpVariable(f"atende_{i}_{j}", cat="Binary")

# Objetivo: minimizar distância total + custo fixo dos centros
custo_fixo_centro = 10000
prob6 += (pulp.lpSum(distancias[(i, j)] * w[(i, j)] 
                     for i in range(n_clientes_sample) 
                     for j in range(n_centros)) +
          pulp.lpSum(custo_fixo_centro * z[j] for j in range(n_centros)))

# Restrições:
# - Cada cliente deve ser atendido por exatamente um centro
for i in range(n_clientes_sample):
    prob6 += pulp.lpSum(w[(i, j)] for j in range(n_centros)) == 1

# - Cliente só pode ser atendido por centro aberto
for i in range(n_clientes_sample):
    for j in range(n_centros):
        prob6 += w[(i, j)] <= z[j]

# - Pelo menos 2 centros devem estar abertos
prob6 += pulp.lpSum(z[j] for j in range(n_centros)) >= 2

# - Máximo 4 centros abertos
prob6 += pulp.lpSum(z[j] for j in range(n_centros)) <= 4

prob6.solve()

print("=== CONSULTA 6: g ===")
centros_abertos = [j for j in range(n_centros) if z[j].value() == 1]
print(f"Centros de atendimento abertos: {centros_abertos}")

for j in centros_abertos:
    clientes_atendidos = [i for i in range(n_clientes_sample) if w[(i, j)].value() == 1]
    print(f"Centro {j} atende {len(clientes_atendidos)} clientes")

distancia_total = sum(distancias[(i, j)] * w[(i, j)].value() 
                     for i in range(n_clientes_sample) 
                     for j in range(n_centros) 
                     if w[(i, j)].value() == 1)
print(f"Distância total: {distancia_total:.2f}")
print(f"Custo total: ${pulp.value(prob6.objective):,.2f}")

print("\n" + "="*80)
print("ANÁLISES PRESCRITIVAS CONCLUÍDAS!")
print("="*80)
