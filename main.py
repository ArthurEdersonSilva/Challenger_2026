import cv2
import os
import torch
from ultralytics import YOLO

# Importações dos nossos módulos customizados
import config
from decision_engine import processar_regras_situacionais

# Dicionário de tradução e mapeamento de cores para os EPIs
MAPEAMENTO_EPIS = {
    "Without Helmet": {"nome": "Sem Capacete", "cor": (0, 0, 255)},          
    "Without Glass": {"nome": "Sem Oculos de Protecao", "cor": (0, 165, 255)}, 
    "Without Mask": {"nome": "Sem Mascara", "cor": (0, 255, 255)},           
    "Without Glove": {"nome": "Sem Luvas", "cor": (0, 255, 0)},              
    "Without Ear Protectors": {"nome": "Sem Protetor Auricular", "cor": (255, 0, 0)}, 
    "person": {"nome": "Operador", "cor": (255, 255, 0)}                     
}

# =====================================================================
# INICIALIZAÇÃO DE COMPONENTES
# =====================================================================

if os.path.exists(config.PATH_MODELO):
    model_epi = YOLO(config.PATH_MODELO)
    print(f"✅ Modelo de EPIs carregado: {config.PATH_MODELO}")
else:
    print(f"❌ Arquivo {config.PATH_MODELO} não encontrado. Usando padrão.")
    model_epi = YOLO("yolov8n.pt")

model_pose = YOLO("yolov8n-pose.pt")
print("✅ Modelo de Pose Estimation carregado com sucesso.")

device = "0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
model_epi.to(device)
model_pose.to(device)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.LARGURA_CAM)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.ALTURA_CAM)

print("\n🚀 Pipeline Unificada SPI Concluída. Monitorando Planta...")
print("Pressione 'q' para sair.")

# =====================================================================
# LOOP PRINCIPAL DE PROCESSAMENTO
# =====================================================================
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Erro ao acessar a transmissão da câmera.")
        break

    results_epi = list(model_epi(frame, conf=config.CONFIDENCIA_MINIMA, imgsz=config.TAMANHO_IMAGEM, stream=True, device=device))
    results_pose = list(model_pose(frame, conf=0.5, stream=True, device=device))

    annotated_frame = frame.copy()
    
    ombro_esquerdo = [0, 0]
    quadril_esquerdo = [0, 0]

    for r_pose in results_pose:
        annotated_frame = r_pose.plot(img=annotated_frame)
        if r_pose.keypoints is not None and len(r_pose.keypoints.xy) > 0:
            kp = r_pose.keypoints.xy[0].cpu().numpy()
            if len(kp) > 11:  
                ombro_esquerdo = kp[5]
                quadril_esquerdo = kp[11]
                
    severidade, lista_pes, epis_detectados = processar_regras_situacionais(
        results_epi, config.PONTOS_ZONA_RISCO, ombro=ombro_esquerdo, quadril=quadril_esquerdo
    )

    for r in results_epi:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            label_original = r.names[cls_id]
            conf = float(box.conf[0])

            info_epi = MAPEAMENTO_EPIS.get(label_original, {"nome": label_original, "cor": (255, 255, 255)})
            texto_exibicao = f"{info_epi['nome']} {conf:.2f}"

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), info_epi['cor'], 2)
            (w, h), _ = cv2.getTextSize(texto_exibicao, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated_frame, (x1, y1 - h - 5), (x1 + w, y1), info_epi['cor'], -1)
            cv2.putText(annotated_frame, texto_exibicao, (x1, y1 - 3), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Renderização da máscara da Zona de Risco
    overlay = annotated_frame.copy()
    cv2.fillPoly(overlay, [config.PONTOS_ZONA_RISCO], (0, 0, 255))
    cv2.addWeighted(overlay, 0.25, annotated_frame, 0.75, 0, annotated_frame)

    for pe in lista_pes:
        cv2.circle(annotated_frame, pe, 6, (0, 255, 255), -1)

    # Exibição dos Alertas Visuais Baseados no Engine de Severidade
    if severidade == "CRITICA":
        cv2.putText(annotated_frame, "ALERTA CRITICO: OPERADOR NA ZONA SEM CAPACETE!", (15, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
    elif severidade == "ALTA":
        cv2.putText(annotated_frame, "ALERTA ALTO: INFRACAO OU FADIGA ERGONOMICA DETECTADA", (15, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2, cv2.LINE_AA)
    elif severidade == "INFORMATIVA":
        cv2.putText(annotated_frame, "AVISO: DESCONFORMIDADE LEVE FORA DA AREA DE RISCO", (15, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("FIAP x SPI Challenge 2026 - Protecao Ativa", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()