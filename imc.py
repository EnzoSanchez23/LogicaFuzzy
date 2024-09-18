import numpy as np
import skfuzzy as fuzzy
from skfuzzy import control as ctrl

# Definindo variáveis de entrada
altura = ctrl.Antecedent(np.arange(1, 2.5, 0.01), 'altura')
peso = ctrl.Antecedent(np.arange(30, 250, 0.5), 'peso')
caloria = ctrl.Antecedent(np.arange(2, 550, 1), 'caloria')

# Definindo a variável de saída
IMC = ctrl.Consequent(np.arange(0, 60, 0.1), 'IMC')

# Definindo os conjuntos fuzzy para IMC
IMC['Abaixo do peso'] = fuzzy.trapmf(IMC.universe, [-1, 0, 17.5, 18.5])
IMC['Normal'] = fuzzy.trapmf(IMC.universe, [17.5, 18.5, 24, 25])
IMC['Sobrepeso'] = fuzzy.trapmf(IMC.universe, [24, 25, 29, 30])
IMC['Obesidade'] = fuzzy.trapmf(IMC.universe, [29, 30, 60, 100])

# Conjuntos fuzzy para altura
altura['baixo'] = fuzzy.trimf(altura.universe, [1, 1.5, 1.65])
altura['mediano'] = fuzzy.trimf(altura.universe, [1.55, 1.65, 1.75])
altura['alto'] = fuzzy.trimf(altura.universe, [1.70, 1.80, 1.90])
altura['muito alto'] = fuzzy.trimf(altura.universe, [1.85, 2, 3])

# Conjuntos fuzzy para peso
peso['muito magro'] = fuzzy.gaussmf(peso.universe, 30, 8)
peso['magro'] = fuzzy.gaussmf(peso.universe, 50, 10)
peso['gordo'] = fuzzy.gaussmf(peso.universe, 100, 12)
peso['obeso'] = fuzzy.gaussmf(peso.universe, 150, 20)

# Conjuntos fuzzy para calorias
caloria['baixa'] = fuzzy.trimf(caloria.universe, [50, 150, 200])
caloria['media'] = fuzzy.trimf(caloria.universe, [197, 250, 300])
caloria['alta'] = fuzzy.trimf(caloria.universe, [297, 350, 400])

# Visualização das variáveis de entrada e saída
altura.view()
peso.view()
caloria.view()
IMC.view()

# Definindo as regras fuzzy
regra1 = ctrl.Rule(altura['muito alto'] | (altura['alto'] & peso['muito magro'] & caloria['baixa']), IMC['Abaixo do peso'])
regra2 = ctrl.Rule(altura['mediano'] | (altura['alto'] & peso['magro'] & caloria['media']), IMC['Normal'])
regra3 = ctrl.Rule((altura['baixo'] | altura['mediano'] | altura['alto']) & peso['gordo'] & caloria['alta'], IMC['Sobrepeso'])
regra4 = ctrl.Rule((altura['baixo'] | altura['mediano']) & peso['obeso'] & caloria['alta'], IMC['Obesidade'])

# Criando o sistema de controle
controlador = ctrl.ControlSystem([regra1, regra2, regra3, regra4])
calculoIMC = ctrl.ControlSystemSimulation(controlador)

# Definindo os valores de entrada para simulação
calculoIMC.input['altura'] = 1.78
calculoIMC.input['peso'] = 68
calculoIMC.input['caloria'] = 100

# Executando o cálculo
calculoIMC.compute()

# Exibindo o resultado do IMC
print(f"IMC calculado: {calculoIMC.output['IMC']}")

# Visualização da simulação
altura.view(sim=calculoIMC)
peso.view(sim=calculoIMC)
caloria.view(sim=calculoIMC)

input()
