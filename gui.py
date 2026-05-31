import sys
import traceback

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QApplication, QHBoxLayout, QVBoxLayout, QWidget
from qfluentwidgets import (
    BodyLabel,
    LineEdit,
    PrimaryPushButton,
    SubtitleLabel,
    TextEdit,
    Theme,
    setTheme,
)

# 直接從你的 test10.py 引入我們寫好的終極模組
from test10 import EvolutionEngine, format_time, fwht_strategy


class XorSimulatorWindow(QWidget):
    def __init__(self):
        super().__init__()

        # 自動跟隨 Windows 系統的深色/淺色模式
        setTheme(Theme.AUTO)

        self.setWindowTitle("XOR 演化模擬器 (WinUI 3 Fluent Design)")
        self.resize(700, 500)

        # 初始化 UI 佈局
        self._init_ui()

        # 初始化演化引擎
        self.engine = EvolutionEngine(max_steps=15, max_history=5000)

    def _init_ui(self):
        self.vbox = QVBoxLayout(self)
        self.vbox.setContentsMargins(24, 24, 24, 24)
        self.vbox.setSpacing(16)

        # 標題
        self.title = SubtitleLabel("兩兩 XOR 陣列演化模擬器")
        self.vbox.addWidget(self.title)

        self.desc = BodyLabel(
            "基於 FWHT 降維打擊演算法。請輸入以逗號分隔的非負整數陣列。"
        )
        # 將 GlobalColor 列舉轉換為明確的 QColor 物件
        self.desc.setTextColor(
            QColor(Qt.GlobalColor.darkGray), QColor(Qt.GlobalColor.lightGray)
        )
        self.vbox.addWidget(self.desc)

        # 輸入區塊
        self.input_layout = QHBoxLayout()
        self.input_layout.setSpacing(12)

        self.array_input = LineEdit()
        self.array_input.setPlaceholderText("例如: 1, 2, 3 或 1024, 2048, 4096")
        self.array_input.setClearButtonEnabled(True)  # 內建 WinUI 3 的清除按鈕

        self.run_btn = PrimaryPushButton("🚀 開始模擬")
        self.run_btn.setFixedWidth(120)
        self.run_btn.clicked.connect(self.run_simulation)

        # 支援按 Enter 鍵直接執行
        self.array_input.returnPressed.connect(self.run_simulation)

        self.input_layout.addWidget(self.array_input)
        self.input_layout.addWidget(self.run_btn)
        self.vbox.addLayout(self.input_layout)

        # 輸出終端機區塊
        self.output_console = TextEdit()
        self.output_console.setReadOnly(True)
        self.output_console.setPlaceholderText("模擬結果將顯示於此...")
        # 設定等寬字體讓報表對齊
        font = self.output_console.font()
        font.setFamily("Consolas")
        self.output_console.setFont(font)

        self.vbox.addWidget(self.output_console)

    def log(self, message):
        """將訊息附加到輸出控制台"""
        self.output_console.append(message)

        # 加上 None 檢查，消除型別警告
        scrollbar = self.output_console.verticalScrollBar()
        if scrollbar is not None:
            scrollbar.setValue(scrollbar.maximum())

    def run_simulation(self):
        input_text = self.array_input.text().strip()

        if not input_text:
            self.log("⚠️ 錯誤：請輸入陣列內容！")
            return

        try:
            # 解析使用者輸入
            arr = [int(x.strip()) for x in input_text.split(",") if x.strip()]
        except ValueError:
            self.log("⚠️ 錯誤：輸入格式無效，請確保只包含整數與逗號。")
            return

        self.log(f"\n{'=' * 50}")
        self.log(f"📥 載入陣列: {arr}")

        try:
            # 呼叫 test10.py 的核心引擎
            res = self.engine.run(arr, strategy_fn=fwht_strategy)

            # 格式化輸出
            formatted_time = format_time(res["elapsed_sec"])
            length_str = str(res["length"])

            if len(length_str) > 15:
                length_str = f"~ {length_str[0]}.{length_str[1:3]}e+{len(length_str) - 1} (長度已達 {len(length_str)} 位數)"

            self.log(f"✅ 狀態: {res['status']}")
            self.log(f"👣 執行步數: {res['step']}")
            self.log(f"⏱️ 運算耗時: {formatted_time}")
            self.log(f"📏 最終長度: {length_str}")

            if "is_length_fixed_point" in res and res["is_length_fixed_point"]:
                self.log("=> ⚠️ 偵測到 N=3 數學奇異點！")

            if "msb_history" in res and res["msb_history"]:
                self.log(f"=> 📉 偵測到勢能崩潰次數: {len(res['msb_history'])}")

        except Exception as e:
            self.log(f"❌ 執行時發生錯誤:\n{traceback.format_exc()}")


if __name__ == "__main__":
    # 啟用高 DPI 縮放支援 (讓字體與介面在 4K 螢幕上依然清晰銳利)
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    window = XorSimulatorWindow()
    window.show()
    sys.exit(app.exec())
