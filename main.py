import cv2
import os
import csv
import torch
from ultralytics import YOLO
from deepface import DeepFace

import config
from decision_engine import processar_regras_situacionais

def carregar_dados_biometricos():
    arquivo_csv = os.path.join("banco_biometria", "dados_operadores.csv")
    operadores = {}
    if os.path.exists(arquivo_csv):
        with open(arquivo_csv, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for linha in reader:
                primeiro_nome = linha["Nome"].split()[0]
                operadores[linha["Matricula"]] = f"{primeiro_nome} ({linha['Cargo']})"
    return operadores

DADOS_OPERADORES = carregar_dados_biometricos()

MAPEAMENTO_EPIS = {
    "Without Helmet": {"nome": "Sem Capacete", "cor": (0, 0, 255)},          
    "Without Glass": {"nome": "Sem Oculos", "cor": (0, 165, 255)}, 
    "Without Mask": {"nome": "Sem Mascara", "cor": (0, 255, 255)},           
    "Without Glove": {"nome": "Sem Luvas", "cor": (0, 255, 0)},              
    "Without Ear Protectors": {"nome": "Sem Protetor", "cor": (255, 0, 0)}, 
    "Without Safety Vest": {"nome": "Sem Colete", "cor": (200, 150, 255)}, 
    "person": {"nome": "Operador", "cor": (255, 255, 0)}                     
}

if os.path.exists(config.PATH_MODELO):
    model_epi = YOLO(config.PATH_MODELO)
    print(f"✅ Modelo de EPIs carregado: {config.PATH_MODELO}")
else:
    model_epi = YOLO("yolov8n.pt")

model_pose = YOLO("yolov8n-pose.pt")

device = "0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
model_epi.to(device)
model_pose.to(device)

# =====================================================================
# SELETOR DE CÂMERA (Para trocar, inverta os comentários #)
# =====================================================================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # [ATIVO] Câmera Primária (Nativa / Celular)
#cap = cv2.VideoCapture(1)                # [DESATIVADO] Câmera Secundária (Externa / USB)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.LARGURA_CAM)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.ALTURA_CAM)

ultimo_operador_identificado = "Buscando Biometria..."
matricula_atual = "0000" 
contador_frames = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    results_epi = list(model_epi(frame, conf=config.CONFIDENCIA_MINIMA, imgsz=config.TAMANHO_IMAGEM, stream=True, device=device))
    results_pose = list(model_pose(frame, conf=0.5, stream=True, device=device))

    annotated_frame = frame.copy()
    ombro_esquerdo = [0, 0]
    quadril_esquerdo = [0, 0]

    contador_frames += 1

    for r_pose in results_pose:
        annotated_frame = r_pose.plot(img=annotated_frame, boxes=False)

        if r_pose.keypoints is not None and len(r_pose.keypoints.xy) > 0:
            kp = r_pose.keypoints.xy[0].cpu().numpy()
            if len(kp) > 11:  
                ombro_esquerdo = kp[5]
                quadril_esquerdo = kp[11]

        for box in r_pose.boxes:
            if int(box.cls[0]) == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf_pessoa = float(box.conf[0])

                if contador_frames % 15 == 0:
                    h_box = y2 - y1
                    y_peito = y1 + int(h_box * 0.55)
                    recorte_rosto = frame[max(0, y1):max(0, y_peito), max(0, x1):max(0, x2)]

                    if recorte_rosto.size > 0:
                        try:
                            match = DeepFace.find(img_path=recorte_rosto, db_path="banco_biometria", model_name="Facenet", enforce_detection=False, silent=True)
                            if len(match) > 0 and not match[0].empty:
                                arquivo_id = os.path.basename(match[0].iloc[0]["identity"])
                                
                                matricula_atual = os.path.splitext(arquivo_id)[0].split('_')[0]
                                ultimo_operador_identificado = DADOS_OPERADORES.get(matricula_atual, "Rosto Desconhecido")
                        except:
                            pass

                if "Buscando" in ultimo_operador_identificado: cor_biometria = (0, 255, 255) 
                elif "Desconhecido" in ultimo_operador_identificado: cor_biometria = (0, 0, 255) 
                else: cor_biometria = (0, 255, 0) 

                texto_pessoa = f"{ultimo_operador_identificado} [{conf_pessoa:.2f}]"
                
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), cor_biometria, 2)
                (w, h), _ = cv2.getTextSize(texto_pessoa, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated_frame, (x1, y1 - h - 5), (x1 + w, y1), cor_biometria, -1)
                cv2.putText(annotated_frame, texto_pessoa, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    severidade, lista_pes, epis_detectados = processar_regras_situacionais(
        results_epi, config.PONTOS_ZONA_RISCO, 
        matricula=matricula_atual,
        operador=ultimo_operador_identificado,
        frame=frame,
        ombro=ombro_esquerdo, 
        quadril=quadril_esquerdo
    )

    for r in results_epi:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label_original = r.names[cls_id]

            if label_original == "person": continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf_epi = float(box.conf[0])
            info_epi = MAPEAMENTO_EPIS.get(label_original, {"nome": label_original, "cor": (255, 255, 255)})

            texto_epi = f"{info_epi['nome']} [{conf_epi:.2f}]"
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), info_epi['cor'], 2)
            (w, h), _ = cv2.getTextSize(texto_epi, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated_frame, (x1, y1 - h - 5), (x1 + w, y1), info_epi['cor'], -1)
            cv2.putText(annotated_frame, texto_epi, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    overlay = annotated_frame.copy()
    cv2.fillPoly(overlay, [config.PONTOS_ZONA_RISCO], (0, 0, 255))
    cv2.addWeighted(overlay, 0.25, annotated_frame, 0.75, 0, annotated_frame)

    for pe in lista_pes:
        cv2.circle(annotated_frame, pe, 6, (0, 255, 255), -1)

    if severidade == "CRITICA":
        cv2.putText(annotated_frame, "CRITICO: OPERADOR NA ZONA SEM CAPACETE!", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
    elif severidade == "ALTA":
        cv2.putText(annotated_frame, "ALTO: INFRACAO OU FADIGA ERGONOMICA", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2, cv2.LINE_AA)

    cv2.imshow("FIAP x SPI Challenge 2026", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
cv2.destroyAllWindows()