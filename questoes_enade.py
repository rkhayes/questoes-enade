import cv2
import numpy as np

def apply_median_filter(image_path: str) -> np.ndarray:
    """
    Referente ao ENADE 2017 - Questão 26.
    Aplica filtro de mediana para mitigação de ruído sal e pimenta.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Imagem não encontrada.")
    
    # Aplica filtro de mediana com kernel 5x5
    median_filtered = cv2.medianBlur(img, 5)
    return median_filtered

def apply_thresholding(image: np.ndarray, threshold_value: int = 127) -> np.ndarray:
    """
    Referente ao ENADE 2005 - Questão 76.
    Aplica limiarização global em uma imagem em tons de cinza.
    """
    # Converte pixels > threshold_value para 255 (branco), caso contrário 0 (preto)
    _, binary_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    return binary_image

def apply_edge_detection(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Referente ao ENADE 2008 - Questão 49.
    Aplica os operadores de Sobel para calcular os gradientes em X e Y.
    """
    # Derivada de primeira ordem em X e Y
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calcula a magnitude absoluta do gradiente
    abs_grad_x = cv2.convertScaleAbs(sobel_x)
    abs_grad_y = cv2.convertScaleAbs(sobel_y)
    
    # Combina os gradientes
    grad_magnitude = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return grad_magnitude

def apply_morphological_erosion(image: np.ndarray) -> np.ndarray:
    """
    Referente ao ENADE 2021 - Questão 29.
    Aplica a operação de erosão para encolher objetos e remover componentes indesejados.
    """
    # Define um elemento estruturante em bloco (retângulo 3x3)
    kernel = np.ones((3, 3), np.uint8)
    
    # Aplica a erosão
    eroded_image = cv2.erode(image, kernel, iterations=1)
    return eroded_image

def main():
    sample_path = "baboon.png"
    
    try:
        # 1. Filtro de Mediana
        print("Aplicando Filtro de Mediana (ENADE 2017)...")
        filtered_img = apply_median_filter(sample_path)
        
        # 2. Limiarização
        print("Aplicando Limiarização (ENADE 2005)...")
        binary_img = apply_thresholding(filtered_img, threshold_value=127)
        
        # 3. Detecção de Bordas (Sobel)
        print("Extraindo Gradientes e Bordas (ENADE 2008)...")
        edges_img = apply_edge_detection(filtered_img)
        
        # 4. Morfologia Matemática (Erosão)
        print("Aplicando Erosão Morfológica (ENADE 2021)...")
        eroded_img = apply_morphological_erosion(binary_img)
        
        # Salvando os resultados
        cv2.imwrite("resultado_mediana.jpg", filtered_img)
        cv2.imwrite("resultado_limiar.jpg", binary_img)
        cv2.imwrite("resultado_bordas.jpg", edges_img)
        cv2.imwrite("resultado_erosao.jpg", eroded_img)
        print("Processamento concluído com sucesso. Resultados salvos.")
        
    except Exception as e:
        print(f"Erro na execução da pipeline: {e}")

if __name__ == "__main__":
    main()
