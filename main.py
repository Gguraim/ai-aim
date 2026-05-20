import sys
import cv2
import numpy as np
import torch  # CUDA 확인용
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from ultralytics import RTDETR

# --- [CUDA 가속 분석 엔진] ---
class CudaAnalysisWorker(QThread):
    progress_signal = pyqtSignal(int)
    frame_signal = pyqtSignal(QImage)
    log_signal = pyqtSignal(str)
    result_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(self, video_path, has_zoom):
        super().__init__()
        self.video_path = video_path
        self.has_zoom = has_zoom
        self.running = True
        
        # [CUDA 체크]
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            # RT-DETR 모델 로드 및 GPU 이동
            self.model = RTDETR('rtdetr-l.pt').to(self.device)
            print(f"Using Device: {self.device}")
        except Exception as e:
            print(f"Model Load Error: {e}")

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.error_signal.emit("영상을 열 수 없습니다. 경로를 확인하세요.")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        unscoped_data, scoped_data = [], []
        
        try:
            for i in range(0, total_frames, 2): # CUDA는 빠르므로 2프레임 간격도 충분
                if not self.running: break
                
                ret, frame = cap.read()
                if not ret: break
                
                h, w = frame.shape[:2]
                center_pt = (w // 2, h // 2)
                
                # 줌 판별 (CPU 연산)
                is_scoped = np.mean(frame[0:45, 0:45]) < 40 if self.has_zoom else False
                
                # AI 추론 (CUDA 가속)
                # persist=True 옵션으로 트래킹 안정성 강화
                results = self.model.predict(frame, classes=[0], device=self.device, verbose=False, conf=0.35)
                
                display_frame = cv2.resize(frame, (800, 450))
                
                if results[0].boxes:
                    # GPU에 있는 데이터를 CPU로 가져와서 처리
                    boxes = results[0].boxes.xywh.cpu().numpy()
                    best_box = min(boxes, key=lambda b: (b[0]-center_pt[0])**2 + (b[1]-center_pt[1])**2)
                    
                    if is_scoped: scoped_data.append((best_box[0], best_box[1]))
                    else: unscoped_data.append((best_box[0], best_box[1]))
                    
                    # 시각화
                    self.draw_overlay(display_frame, best_box, is_scoped, w, h)

                # 실시간 화면 전송
                rgb_img = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                qt_img = QImage(rgb_img.data, 800, 450, 800*3, QImage.Format.Format_RGB888).copy()
                self.frame_signal.emit(qt_img)
                self.progress_signal.emit(int((i / total_frames) * 100))

            self.result_signal.emit({"unscoped": unscoped_data, "scoped": scoped_data})

        except Exception as e:
            self.error_signal.emit(f"CUDA 분석 오류: {str(e)}")
        finally:
            cap.release()

    def draw_overlay(self, img, box, is_scoped, org_w, org_h):
        sx, sy = 800/org_w, 450/org_h
        tx, ty, tw, th = box
        color = (0, 0, 255) if is_scoped else (0, 255, 0)
        cv2.rectangle(img, (int((tx-tw/2)*sx), int((ty-th/2)*sy)), 
                      (int((tx+tw/2)*sx), int((ty+th/2)*sy)), color, 2)
        cv2.putText(img, "CUDA ACCELERATED", (10, 25), 1, 1, (0, 255, 0), 1)

# --- [메인 UI 창] ---
class OverwatchCudaApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OW2 PRO ANALYZER (CUDA ENGINE)")
        self.setStyleSheet("background-color: #0A0A0A; color: white;")
        self.setGeometry(100, 100, 1150, 850)
        self.initUI()
        self.video_path = ""

    def initUI(self):
        main_layout = QHBoxLayout()
        
        # 좌측 패널
        ctrl_layout = QVBoxLayout()
        
        config_box = QGroupBox("HARDWARE & SENS")
        config_box.setStyleSheet("color: #00FF41;") # 매트릭스 그린
        grid = QGridLayout()
        self.dpi_in = QLineEdit("800"); self.sens_in = QLineEdit("5.0"); self.zoom_in = QLineEdit("")
        grid.addWidget(QLabel("DPI:"), 0, 0); grid.addWidget(self.dpi_in, 0, 1)
        grid.addWidget(QLabel("일반 감도:"), 1, 0); grid.addWidget(self.sens_in, 1, 1)
        grid.addWidget(QLabel("줌 감도:"), 2, 0); grid.addWidget(self.zoom_in, 2, 1)
        config_box.setLayout(grid)
        ctrl_layout.addWidget(config_box)

        self.btn_file = QPushButton("영상 파일 선택")
        self.btn_file.clicked.connect(self.get_video)
        ctrl_layout.addWidget(self.btn_file)

        self.btn_start = QPushButton("CUDA 분석 시작")
        self.btn_start.clicked.connect(self.start_analysis)
        self.btn_start.setStyleSheet("background: #0081CB; height: 50px; font-weight: bold;")
        ctrl_layout.addWidget(self.btn_start)
        
        self.log_box = QTextEdit(); self.log_box.setReadOnly(True)
        self.log_box.setStyleSheet("background: #000; color: #0F0;")
        ctrl_layout.addWidget(self.log_box)
        
        main_layout.addLayout(ctrl_layout, 1)
        
        # 우측 패널
        mon_layout = QVBoxLayout()
        self.video_screen = QLabel("GPU 분석 모니터")
        self.video_screen.setFixedSize(800, 450)
        self.video_screen.setStyleSheet("border: 2px solid #00FF41; background: black;")
        mon_layout.addWidget(self.video_screen)
        
        self.p_bar = QProgressBar()
        mon_layout.addWidget(self.p_bar)
        
        self.report_box = QTextEdit(); self.report_box.setReadOnly(True)
        mon_layout.addWidget(self.report_box)
        
        main_layout.addLayout(mon_layout, 3)
        
        container = QWidget(); container.setLayout(main_layout); self.setCentralWidget(container)

    def get_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "영상 선택")
        if path: self.video_path = path

    def start_analysis(self):
        if not self.video_path: return
        self.btn_start.setEnabled(False)
        self.log_box.append(f"CUDA 상태: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else '사용 불가'}")
        
        self.worker = CudaAnalysisWorker(self.video_path, len(self.zoom_in.text()) > 0)
        self.worker.frame_signal.connect(lambda img: self.video_screen.setPixmap(QPixmap.fromImage(img)))
        self.worker.progress_signal.connect(self.p_bar.setValue)
        self.worker.result_signal.connect(self.final_calc)
        self.worker.error_signal.connect(lambda msg: QMessageBox.critical(self, "에러", msg))
        self.worker.start()

    def final_calc(self, data):
        self.btn_start.setEnabled(True)
        report = "--- [ CUDA PRO ANALYSIS REPORT ] ---\n"
        
        def process(coords, name, cur_sens):
            if len(coords) < 15: return f"{name}: 데이터 부족\n"
            overs = 0
            for i in range(2, len(coords)):
                d1 = coords[i-1][0] - coords[i-2][0]
                d2 = coords[i][0] - coords[i-1][0]
                if d1 * d2 < -15: overs += 1
            
            ov_rate = (overs / len(coords)) * 100
            txt = f"[{name}]\n - 정확도: {100-ov_rate:.1f}%\n"
            try:
                s, d = float(cur_sens), float(self.dpi_in.text())
                if ov_rate > 15: txt += f" >> 감도 높음! 추천: {s*0.93:.2f} (eDPI: {s*d*0.93:.0f})\n"
                elif ov_rate < 5: txt += f" >> 감도 낮음! 추천: {s*1.07:.2f} (eDPI: {s*d*1.07:.0f})\n"
                else: txt += " >> 감도 최적 상태\n"
            except: pass
            return txt + "\n"

        report += process(data['unscoped'], "GENERAL", self.sens_in.text())
        if len(self.zoom_in.text()) > 0:
            report += process(data['scoped'], "SCOPED", self.zoom_in.text())
        self.report_box.setText(report)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = OverwatchCudaApp()
    ex.show()
    sys.exit(app.exec())