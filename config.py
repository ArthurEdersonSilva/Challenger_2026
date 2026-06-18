import numpy as np

# Configurações antigas mantidas...
PATH_MODELO = "best.pt"
CONFIDENCIA_MINIMA = 0.5
TAMANHO_IMAGEM = 640
LARGURA_CAM = 640
ALTURA_CAM = 480
PONTOS_ZONA_RISCO = np.array([[100, 400], [250, 200], [450, 200], [600, 400]], np.int32)

# =====================================================================
# NOVOS PARÂMETROS ADICIONADOS (CIÊNCIA DE DADOS E ALERTA EXTERNO)
# =====================================================================
PATH_LOGS_CSV = "historico_incidentes.csv"

# Configurações do Alerta Sonoro (Frequência em Hz, Duração em milissegundos)
FREQ_BEEP_CRITICO = 1500  
DURACAO_BEEP_CRITICO = 400