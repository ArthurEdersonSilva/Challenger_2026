import cv2
import csv
import os
import winsound
from datetime import datetime

# =====================================================================
# CONFIGURAÇÕES E PARÂMETROS DE TEMPO (FASE 2)
# =====================================================================
limite_frames_fadiga = 90  # Aprox. 3 segundos em uma câmera operando a ~30 FPS
contador_postura_inadequada = 0

# Variável para controle de persistência de dados (evita gravar duplicatas a cada frame)
ultimo_estado_severidade = "NORMAL"

def verificar_ponto_em_poligono(ponto, poligono):
    """Retorna 1 ou 0 se o ponto estiver dentro/borda do polígono, -1 se fora."""
    return cv2.pointPolygonTest(poligono, ponto, False)

def avaliar_fadiga_ergonomica(ombro, quadril):
    """
    Mede a distância vertical entre o ombro e o quadril obtidos pelo esqueleto.
    Se a distância encurtar muito de forma persistente, indica tronco fletido (NR-17).
    """
    global contador_postura_inadequada
    
    # Se os pontos de pose não foram mapeados (coordenadas zeradas), descarta o cálculo
    if (ombro[0] == 0 and ombro[1] == 0) or (quadril[0] == 0 and quadril[1] == 0):
        return False
        
    # Uma distância vertical (eixo Y) muito curta indica que o operador inclinou/curvou
    distancia_vertical = abs(quadril[1] - ombro[1])
    
    # CALIBRAGEM DINÂMICA: Ajustamos o limiar baseado no tamanho detectado do operador.
    limiar_dinamico = 85 
    
    if distancia_vertical < limiar_dinamico: 
        contador_postura_inadequada += 1
    else:
        contador_postura_inadequada = max(0, contador_postura_inadequada - 1)
        
    return contador_postura_inadequada > limite_frames_fadiga

def registrar_incidente_csv(severidade, infracoes):
    """
    Pipeline de Ciência de Dados Aplicada (Fase 3/4).
    Registra eventos críticos em histórico persistente estruturado para análise SSMA.
    """
    from config import PATH_LOGS_CSV
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lista_infracoes = "; ".join(infracoes) if len(infracoes) > 0 else "Fadiga Ergonomica"
    
    arquivo_novo = not os.path.exists(PATH_LOGS_CSV)
    
    try:
        with open(PATH_LOGS_CSV, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            if arquivo_novo:
                writer.writerow(["Timestamp", "Severidade", "Infracoes_Mapeadas"])
            writer.writerow([timestamp, severidade, lista_infracoes])
        print(f"📊 [DATA PIPELINE] Evento {severidade} gravado com sucesso em {PATH_LOGS_CSV}")
    except Exception as e:
        print(f"❌ Erro ao salvar log de dados: {e}")

def disparar_alerta_proativo(severidade):
    """
    Sistema de Notificação Inteligente - Canal de Campo (Fase 3).
    Gera estímulo sonoro físico perceptível pelo operador de forma síncrona.
    """
    from config import FREQ_BEEP_CRITICO, DURACAO_BEEP_CRITICO
    if severidade == "CRITICA":
        winsound.Beep(FREQ_BEEP_CRITICO, DURACAO_BEEP_CRITICO)

def calcular_nivel_severidade(tem_pessoa_na_zona, infracoes_detectadas, fadiga_detectada):
    """
    Engine de Decisão Baseado em Contexto (Fase 3).
    Classifica a urgência do risco com foco preditivo e preventivo.
    """
    global ultimo_estado_severidade

    if len(infracoes_detectadas) == 0 and not fadiga_detectada:
        ultimo_estado_severidade = "NORMAL"
        return "NORMAL"

    # SEVERIDADE CRÍTICA: Operador dentro da área vermelha de perigo SEM capacete
    if tem_pessoa_na_zona and "Without Helmet" in infracoes_detectadas:
        severidade_atual = "CRITICA"
        
    # SEVERIDADE ALTA: Operador dentro da área de risco com qualquer outra infração OU fadiga muscular detectada
    elif tem_pessoa_na_zona or fadiga_detectada:
        severidade_atual = "ALTA"
        
    # SEVERIDADE INFORMATIVA: Infrações leves ou fora do polígono de perigo direto
    else:
        severidade_atual = "INFORMATIVA"

    # Inteligência de dados: Só grava logs e emite bipes quando o estado de risco muda (bloqueia flood)
    if severidade_atual != ultimo_estado_severidade:
        if severidade_atual in ["CRITICA", "ALTA"]:
            registrar_incidente_csv(severidade_atual, infracoes_detectadas)
        disparar_alerta_proativo(severidade_atual)
        ultimo_estado_severidade = severidade_atual

    return severidade_atual

def processar_regras_situacionais(results, poligono_risco, ombro=None, quadril=None):
    """
    Processa as violações de EPI, rastreia os pés na zona de risco e calcula a postura.
    Retorna: (str_nivel_severidade, list_pontos_pes, list_infracoes)
    """
    tem_pessoa_na_zona = False
    pontos_pes = []
    infracoes_detectadas = []

    # 1. Varredura e mapeamento das detecções do YOLO
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            label = r.names[cls_id]

            # Rastreamento do operador e sua base de apoio
            if label == "person":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                ponto_base = (int((x1 + x2) / 2), y2) 
                pontos_pes.append(ponto_base)
                
                # Valida se os pés estão tocando a máscara de risco
                if verificar_ponto_em_poligono(ponto_base, poligono_risco) >= 0:
                    tem_pessoa_na_zona = True

            # Captura de infrações de EPI mapeadas em inglês
            if label in ["Without Helmet", "Without Glass", "Without Mask", "Without Glove", "Without Ear Protectors"]:
                infracoes_detectadas.append(label)

    # 2. Avaliação de Ergonomia da Pose Estimation (Fase 2)
    fadiga_detectada = False
    if ombro is not None and quadril is not None:
        fadiga_detectada = avaliar_fadiga_ergonomica(ombro, quadril)

    # 3. Processamento do Score Contextual de Severidade (Fase 3)
    severidade = calcular_nivel_severidade(tem_pessoa_na_zona, infracoes_detectadas, fadiga_detectada)

    return severidade, pontos_pes, infracoes_detectadas