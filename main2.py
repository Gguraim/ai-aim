import sys
import cv2
import numpy as np
import mss
import torch
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from ultralytics import YOLO

# --- [1. 크기 조절/이동 ROI 박스] ---
class ROIBox(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setGeometry(760, 340, 400, 400)
        self.setMinimumSize(150, 150)
        self.m_moving = False
        self.m_resizing = False
        self.border_color = QColor(255, 0, 0, 200)

    def paintEvent(self, event):
        painter = QPainter(self)
        pen = QPen(self.border_color, 4)
        painter.setPen(pen)
        rect = self.rect().adjusted(2, 2, -2, -2)
        painter.drawRect(rect)
        painter.fillRect(rect, QColor(255, 0, 0, 30))
        # 우측 하단 핸들
        painter.fillRect(self.width()-20, self.height()-20, 20, 20, QColor(255, 255, 255, 150))

    def mousePressEvent(self, event):
        if event.pos().x() > self.width() - 30 and event.pos().y() > self.height() - 30:
            self.m_resizing = True
        else:
            self.m_moving = True
        self.m_startPos = event.globalPosition().toPoint()
        self.m_startGeometry = self.geometry()

    def mouseMoveEvent(self, event):
        if not event.buttons() == Qt.MouseButton.LeftButton: return
        diff = event.globalPosition().toPoint() - self.m_startPos
        if self.m_resizing:
            self.resize(max(150, self.m_startGeometry.width() + diff.x()), 
                        max(150, self.m_startGeometry.height() + diff.y()))
        elif self.m_moving:
            self.move(self.m_startGeometry.topLeft() + diff)

    def mouseReleaseEvent(self, event):
        self.m_moving = self.m_resizing = False

# --- [2. 실시간 분석 엔진 (일반/줌 감도 개별 분석)] ---
class LiveAnalysisThread(QThread):
    frame_signal = pyqtSignal(np.ndarray, bool) # 프레임, 줌여부 전송
    stats_signal = pyqtSignal(dict)

    def __init__(self, roi_widget):
        super().__init__()
        self.roi = roi_widget
        self.running = True
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO('yolov8n.pt').to(self.device)
        self.sct = mss.mss()
        
        # 데이터 분리 저장
        self.stats = {
            "unscoped": {"pts": [], "over": 0, "under": 0},
            "scoped": {"pts": [], "over": 0, "under": 0}
        }

    def is_scoped(self, frame):
        """박스 모서리 밝기를 분석하여 줌 상태 판별"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corner_sample = gray[0:30, 0:30]
        return np.mean(corner_sample) < 45

    def run(self):
        while self.running:
            monitor = {"top": self.roi.y(), "left": self.roi.x(), "width": self.roi.width(), "height": self.roi.height()}
            img = np.array(self.sct.grab(monitor))
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            # 줌 상태 확인
            scoped_active = self.is_scoped(frame)
            mode = "scoped" if scoped_active else "unscoped"
            
            # AI 추론
            results = self.model.predict(frame, classes=[0], device=self.device, verbose=False, conf=0.4)
            
            cx, cy = frame.shape[1]//2, frame.shape[0]//2
            if results[0].boxes:
                boxes = results[0].boxes.xywh.cpu().numpy()
                best_box = min(boxes, key=lambda b: (b[0]-cx)**2 + (b[1]-cy)**2)
                tx, ty, tw, th = best_box
                
                # 데이터 포인트 기록 및 오버/언더슈팅 분석
                pts = self.stats[mode]["pts"]
                pts.append((tx, ty))
                if len(pts) > 3:
                    v1 = pts[-2][0] - pts[-3][0]
                    v2 = pts[-1][0] - pts[-2][0]
                    if v1 * v2 < -15: self.stats[mode]["over"] += 1
                    elif abs(v1) > 5 and abs(v2) < 1: self.stats[mode]["under"] += 1
                
                # 시각화 (줌 상태면 빨간색, 일반이면 초록색)
                color = (0, 0, 255) if scoped_active else (0, 255, 0)
                cv2.rectangle(frame, (int(tx-tw/2), int(ty-th/2)), (int(tx+tw/2), int(ty+th/2)), color, 2)
                cv2.line(frame, (cx, cy), (int(tx), int(ty)), (0, 255, 255), 1)

            self.frame_signal.emit(frame, scoped_active)
            self.stats_signal.emit({
                "unscoped_over": self.stats["unscoped"]["over"],
                "unscoped_under": self.stats["unscoped"]["under"],
                "unscoped_total": len(self.stats["unscoped"]["pts"]),
                "scoped_over": self.stats["scoped"]["over"],
                "scoped_under": self.stats["scoped"]["under"],
                "scoped_total": len(self.stats["scoped"]["pts"])
            })
            if len(self.stats[mode]["pts"]) > 300: self.stats[mode]["pts"].pop(0)

    def stop(self): self.running = False

# --- [3. 메인 컨트롤러] ---
class MainController(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OW2 Real-time Sniper Optimizer")
        self.setGeometry(100, 100, 450, 800)
        self.setStyleSheet("background-color: #121212; color: white;")
        
        self.roi_box = ROIBox()
        self.roi_box.show()
        self.analysis_thread = None
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # 설정 섹션
        cfg_group = QGroupBox("감도 설정")
        cfg_group.setStyleSheet("color: #F57C00; font-weight: bold;")
        grid = QGridLayout()
        self.dpi_in = QLineEdit("800"); self.sens_in = QLineEdit("5.0"); self.zoom_in = QLineEdit("37.89")
        input_style = "background: #222; color: white; border: 1px solid #444; padding: 5px;"
        for i in [self.dpi_in, self.sens_in, self.zoom_in]: i.setStyleSheet(input_style)
        
        grid.addWidget(QLabel("DPI:"), 0, 0); grid.addWidget(self.dpi_in, 0, 1)
        grid.addWidget(QLabel("일반 감도:"), 1, 0); grid.addWidget(self.sens_in, 1, 1)
        grid.addWidget(QLabel("조준(줌) 감도:"), 2, 0); grid.addWidget(self.zoom_in, 2, 1)
        cfg_group.setLayout(grid)
        layout.addWidget(cfg_group)

        self.start_btn = QPushButton("분석 시작")
        self.start_btn.clicked.connect(self.toggle_analysis)
        self.start_btn.setStyleSheet("background: #F57C00; height: 50px; font-weight: bold;")
        layout.addWidget(self.start_btn)

        self.monitor_view = QLabel("AI Monitoring Screen")
        self.monitor_view.setFixedSize(410, 250)
        self.monitor_view.setStyleSheet("background: black; border: 2px solid #555;")
        self.monitor_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.monitor_view)

        # 결과 영역
        self.report_area = QTextEdit()
        self.report_area.setReadOnly(True)
        self.report_area.setStyleSheet("background: #1E1E1E; color: #00FF41; font-family: Consolas; font-size: 13px;")
        layout.addWidget(self.report_area)

        container = QWidget(); container.setLayout(layout); self.setCentralWidget(container)

    def toggle_analysis(self):
        if self.analysis_thread and self.analysis_thread.isRunning():
            self.analysis_thread.stop()
            self.start_btn.setText("분석 시작")
            self.roi_box.border_color = QColor(255, 0, 0, 200); self.roi_box.update()
        else:
            self.analysis_thread = LiveAnalysisThread(self.roi_box)
            self.analysis_thread.frame_signal.connect(self.update_frame)
            self.analysis_thread.stats_signal.connect(self.update_report)
            self.analysis_thread.start()
            self.start_btn.setText("분석 중지")
            self.roi_box.border_color = QColor(0, 255, 0, 200); self.roi_box.update()

    def update_frame(self, frame, is_scoped):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qt_img = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1]*3, QImage.Format.Format_RGB888)
        self.monitor_view.setPixmap(QPixmap.fromImage(qt_img).scaled(410, 250, Qt.AspectRatioMode.KeepAspectRatio))
        if is_scoped: self.monitor_view.setStyleSheet("border: 2px solid #FF0000;")
        else: self.monitor_view.setStyleSheet("border: 2px solid #00FF00;")

    def update_report(self, s):
        def get_advice(over, under, current_sens):
            total = over + under
            if total < 5: return "데이터 수집 중..."
            try:
                curr = float(current_sens)
                if over > under + 3: return f"낮추세요 ➔ {curr*0.95:.2f}"
                elif under > over + 3: return f"높이세요 ➔ {curr*1.05:.2f}"
                else: return "최적입니다!"
            except: return "입력 확인"

        text = f"--- [ 실시간 분석 리포트 ] ---\n\n"
        text += f"▶ 일반 상태 (힙샷)\n  - 오버:{s['unscoped_over']} | 언더:{s['unscoped_under']}\n"
        text += f"  - 추천: {get_advice(s['unscoped_over'], s['unscoped_under'], self.sens_in.text())}\n\n"
        text += f"▶ 줌 상태 (스코프)\n  - 오버:{s['scoped_over']} | 언더:{s['scoped_under']}\n"
        text += f"  - 추천: {get_advice(s['scoped_over'], s['scoped_under'], self.zoom_in.text())}\n"
        self.report_area.setText(text)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainController()
    window.show()
    sys.exit(app.exec())