#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
本科毕业设计演示版：
文本输入 -> 个性化字库检索 -> G-code 排版拼接 -> 预览 -> 导出综合 G-code
"""

from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from PyQt5.QtCore import Qt, QSettings, QRectF
    from PyQt5.QtGui import QColor, QFont, QPainter, QPen
    from PyQt5.QtWidgets import (
        QApplication,
        QComboBox,
        QFileDialog,
        QFormLayout,
        QFrame,
        QGraphicsScene,
        QGraphicsView,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QPlainTextEdit,
        QDoubleSpinBox,
        QSplitter,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except Exception:
    print("PyQt5 is required. Install: pip install PyQt5")
    raise


GCODE_RE = re.compile(r"\bG(0|1)\b", re.IGNORECASE)
AXIS_RE = re.compile(r"\b([XYZF])\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)


@dataclass
class GlyphData:
    char: str
    path: Path
    strokes: List[List[Tuple[float, float]]] = field(default_factory=list)
    bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    width: float = 1.0
    height: float = 1.0


@dataclass
class PlacedGlyph:
    char: str
    strokes: List[List[Tuple[float, float]]]
    missing: bool = False


@dataclass
class LayoutResult:
    placed: List[PlacedGlyph] = field(default_factory=list)
    output_strokes: List[List[Tuple[float, float]]] = field(default_factory=list)
    travel_segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = field(default_factory=list)
    missing_chars: List[str] = field(default_factory=list)
    missing_count: int = 0
    total_chars: int = 0
    used_width: float = 0.0
    used_height: float = 0.0
    line_counts: List[int] = field(default_factory=list)
    draw_length: float = 0.0
    travel_length: float = 0.0
    pen_lifts: int = 0
    export_file: str = ""
    warnings: List[str] = field(default_factory=list)


class GCodeParser:
    @staticmethod
    def parse_file(path: Path) -> GlyphData:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        x = 0.0
        y = 0.0
        z = 5.0
        z_values: List[float] = []
        moves: List[Tuple[Tuple[float, float], Tuple[float, float], float]] = []

        for raw in lines:
            line = raw.strip()
            if not line or line.startswith(";"):
                continue
            g = GCODE_RE.search(line)
            axes = {k.upper(): float(v) for k, v in AXIS_RE.findall(line)}
            if not g:
                if "Z" in axes:
                    z = axes["Z"]
                    z_values.append(z)
                continue

            nx = axes.get("X", x)
            ny = axes.get("Y", y)
            nz = axes.get("Z", z)
            z_values.append(nz)
            if nx != x or ny != y:
                moves.append(((x, y), (nx, ny), nz))
            x, y, z = nx, ny, nz

        if z_values:
            z_min = min(z_values)
            z_max = max(z_values)
            threshold = (z_min + z_max) * 0.5
        else:
            threshold = 2.5

        strokes: List[List[Tuple[float, float]]] = []
        current: Optional[List[Tuple[float, float]]] = None
        for p0, p1, nz in moves:
            if nz <= threshold:
                if current is None:
                    current = [p0, p1]
                else:
                    if current[-1] != p0:
                        current.append(p0)
                    current.append(p1)
            else:
                if current and len(current) >= 2:
                    strokes.append(current)
                current = None

        if current and len(current) >= 2:
            strokes.append(current)
        if not strokes:
            strokes = [[(0.0, 0.0), (1.0, 0.0)]]

        xs = [p[0] for s in strokes for p in s]
        ys = [p[1] for s in strokes for p in s]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)

        return GlyphData(
            char=path.stem,
            path=path,
            strokes=strokes,
            bbox=(xmin, ymin, xmax, ymax),
            width=max(1.0, xmax - xmin),
            height=max(1.0, ymax - ymin),
        )


class CharacterLibrary:
    def __init__(self) -> None:
        self.folder: Optional[Path] = None
        self.glyphs: Dict[str, GlyphData] = {}
        self.cache: Dict[str, GlyphData] = {}

    def load_folder(self, folder: str) -> int:
        p = Path(folder)
        if not p.exists() or not p.is_dir():
            raise FileNotFoundError(f"Folder not found: {folder}")
        self.folder = p
        self.glyphs.clear()

        files = sorted([f for f in p.iterdir() if f.is_file() and f.suffix.lower() in (".gcode", ".nc", ".tap")])
        for f in files:
            if f.stem:
                self.glyphs[f.stem] = GlyphData(char=f.stem, path=f)
        return len(self.glyphs)

    def has_char(self, ch: str) -> bool:
        return ch in self.glyphs

    def get_glyph(self, ch: str) -> Optional[GlyphData]:
        base = self.glyphs.get(ch)
        if not base:
            return None
        key = str(base.path.resolve())
        if key not in self.cache:
            self.cache[key] = GCodeParser.parse_file(base.path)
        return self.cache[key]


class LayoutEngine:
    @staticmethod
    def placeholder() -> List[List[Tuple[float, float]]]:
        return [[(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)], [(0.15, 0.15), (0.85, 0.85)]]

    @staticmethod
    def poly_len(poly: List[Tuple[float, float]]) -> float:
        return sum(math.hypot(poly[i][0] - poly[i - 1][0], poly[i][1] - poly[i - 1][1]) for i in range(1, len(poly)))

    def layout(
        self,
        library: CharacterLibrary,
        text: str,
        origin_x: float,
        origin_y: float,
        font_size: float,
        char_spacing: float,
        line_spacing: float,
        align: str,
        missing_policy: str,
        page_width: float,
        page_height: float,
    ) -> LayoutResult:
        result = LayoutResult()
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        result.total_chars = len([c for c in normalized if c not in ("\n",)])
        punct_chars = {"。", "，"}

        def cell_w(ch: str, g: Optional[GlyphData], missing: bool) -> float:
            if ch == " ":
                return font_size * 0.45 + char_spacing
            # Chinese punctuation should occupy one full character slot.
            if ch in punct_chars:
                return font_size + char_spacing
            if g and not missing:
                s = font_size / max(g.height, 1e-6)
                return g.width * s + char_spacing
            return font_size * 0.75 + char_spacing

        lines: List[List[Tuple[str, Optional[GlyphData], bool]]] = []
        cur: List[Tuple[str, Optional[GlyphData], bool]] = []
        curw = 0.0

        for ch in normalized:
            if ch == "\n":
                lines.append(cur)
                cur, curw = [], 0.0
                continue

            g = library.get_glyph(ch)
            missing = g is None and ch != " "
            if missing:
                result.missing_chars.append(ch)
                result.missing_count += 1
                if missing_policy == "error":
                    result.warnings.append(f"缺失字符: {ch}")
                    return result
                if missing_policy == "skip":
                    continue

            w = cell_w(ch, g, missing)
            if cur and curw + w > max(1.0, page_width - origin_x):
                lines.append(cur)
                cur, curw = [], 0.0
            cur.append((ch, g, missing))
            curw += w

        lines.append(cur)
        result.line_counts = [len(x) for x in lines]

        # Unified paper coordinate system:
        # origin at lower-left, +X to right, +Y upward.
        # Layout stage only applies translation + uniform scale, no mirroring.
        # Paper coordinate: origin at lower-left, +X right, +Y up.
        # Line wrap must go downward => Y decreases line by line.
        y_cursor = origin_y
        min_x = origin_x
        max_x = origin_x
        min_y = origin_y
        max_y = origin_y

        for line in lines:
            if not line:
                y_cursor -= (font_size + line_spacing)
                continue

            line_w = max(0.0, sum(cell_w(ch, g, m) for ch, g, m in line) - char_spacing)
            if align == "center":
                x_cursor = origin_x + max(0.0, (page_width - origin_x - line_w) * 0.5)
            elif align == "right":
                x_cursor = max(origin_x, page_width - line_w)
            else:
                x_cursor = origin_x

            for ch, g, missing in line:
                if ch == " ":
                    x_cursor += font_size * 0.45 + char_spacing
                    continue

                if g and not missing:
                    gx0, gy0, gx1, gy1 = g.bbox
                    if ch in punct_chars:
                        # Normalize by local bbox then center in a full character cell.
                        cell_size = font_size
                        side = max(g.width, g.height, 1e-6)
                        s = cell_size / side
                        gw = g.width * s
                        gh = g.height * s
                        off_x = (cell_size - gw) * 0.5
                        off_y = (cell_size - gh) * 0.5
                    else:
                        s = font_size / max(g.height, 1e-6)
                        gw = g.width * s
                        gh = g.height * s
                        off_x = 0.0
                        off_y = 0.0
                    strokes: List[List[Tuple[float, float]]] = []
                    for st in g.strokes:
                        pts = []
                        for px, py in st:
                            tx = x_cursor + off_x + (px - gx0) * s
                            # Keep source glyph orientation; do NOT flip X/Y here.
                            ty = y_cursor + off_y + (py - gy0) * s
                            pts.append((tx, ty))
                        if len(pts) >= 2:
                            strokes.append(pts)
                    pg = PlacedGlyph(char=ch, strokes=strokes, missing=False)
                    result.placed.append(pg)
                    if ch in punct_chars:
                        adv = font_size
                    else:
                        adv = gw
                    x_cursor += adv + char_spacing
                    min_x = min(min_x, x_cursor - adv - char_spacing)
                    max_x = max(max_x, x_cursor)
                    max_y = max(max_y, y_cursor + (font_size if ch in punct_chars else gh))
                    min_y = min(min_y, y_cursor)
                else:
                    pw = font_size * 0.75
                    ph = font_size
                    strokes = []
                    for st in self.placeholder():
                        # Placeholder follows the same coordinate convention (no Y flip).
                        pts = [(x_cursor + p[0] * pw, y_cursor + p[1] * ph) for p in st]
                        if len(pts) >= 2:
                            strokes.append(pts)
                    result.placed.append(PlacedGlyph(char=ch, strokes=strokes, missing=True))
                    x_cursor += pw + char_spacing
                    min_x = min(min_x, x_cursor - pw - char_spacing)
                    max_x = max(max_x, x_cursor)
                    max_y = max(max_y, y_cursor + ph)
                    min_y = min(min_y, y_cursor)

            y_cursor -= (font_size + line_spacing)
            if y_cursor < 0:
                result.warnings.append("文本超出页面下边界")

        out: List[List[Tuple[float, float]]] = []
        travel: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
        draw_len = 0.0
        travel_len = 0.0
        prev_end: Optional[Tuple[float, float]] = None

        for pg in result.placed:
            for st in pg.strokes:
                if len(st) < 2:
                    continue
                if prev_end is not None and prev_end != st[0]:
                    travel.append((prev_end, st[0]))
                    travel_len += math.hypot(st[0][0] - prev_end[0], st[0][1] - prev_end[1])
                out.append(st)
                draw_len += self.poly_len(st)
                prev_end = st[-1]

        result.output_strokes = out
        result.travel_segments = travel
        result.draw_length = draw_len
        result.travel_length = travel_len
        result.pen_lifts = len(out)
        result.used_width = max(0.0, max_x - min_x)
        result.used_height = max(0.0, max_y - min_y)
        result.missing_chars = sorted(set(result.missing_chars), key=result.missing_chars.index)
        return result


class PreviewWidget(QGraphicsView):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.scene_ = QGraphicsScene(self)
        self.setScene(self.scene_)
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setBackgroundBrush(QColor("#0B1220"))
        self.setDragMode(QGraphicsView.ScrollHandDrag)

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        scale = 1.12 if event.angleDelta().y() > 0 else 1.0 / 1.12
        self.scale(scale, scale)

    def draw_layout(self, result: LayoutResult, page_w: float, page_h: float, show_travel: bool = True) -> None:
        self.scene_.clear()
        self.scene_.addRect(QRectF(0, 0, page_w, page_h), QPen(QColor("#64748B"), 1.0))

        draw_pen = QPen(QColor("#22C55E"), 1.3)
        missing_pen = QPen(QColor("#F59E0B"), 1.3)
        travel_pen = QPen(QColor("#60A5FA"), 1.0, Qt.DashLine)

        for pg in result.placed:
            pen = missing_pen if pg.missing else draw_pen
            for st in pg.strokes:
                for i in range(1, len(st)):
                    # Display mapping only: convert paper Y-up to screen Y-down.
                    x1, y1 = st[i - 1]
                    x2, y2 = st[i]
                    self.scene_.addLine(x1, page_h - y1, x2, page_h - y2, pen)

        if show_travel:
            for p0, p1 in result.travel_segments:
                self.scene_.addLine(p0[0], page_h - p0[1], p1[0], page_h - p1[1], travel_pen)

        self.setSceneRect(self.scene_.itemsBoundingRect().adjusted(-40, -40, 40, 40))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("写字机器人文本排版系统")
        self.resize(1800, 1080)

        self.settings = QSettings("GraduationProject", "TextGCodeComposer")
        self.library = CharacterLibrary()
        self.layout_engine = LayoutEngine()
        self.result = LayoutResult()

        # Fixed A4 in mm
        self.page_w = 210.0
        self.page_h = 297.0
        self.z_up = 5.0
        self.z_down = 0.0
        self.feed = 1800.0
        self.rapid = 4000.0

        self._build_ui()
        self._apply_theme()
        self._load_last_state()

    def _spin_mm(self, lo: float, hi: float, val: float, step: float = 0.5, dec: int = 1) -> QDoubleSpinBox:
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setValue(val)
        s.setSingleStep(step)
        s.setDecimals(dec)
        s.setSuffix(" mm")
        s.setMinimumWidth(230)
        return s

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        lay = QHBoxLayout(root)
        lay.setContentsMargins(18, 18, 18, 18)
        lay.setSpacing(16)

        splitter = QSplitter(Qt.Horizontal)
        lay.addWidget(splitter)

        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setSpacing(14)
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setSpacing(14)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([760, 1040])

        g_lib = QGroupBox("字库目录")
        f_lib = QFormLayout(g_lib)
        f_lib.setHorizontalSpacing(14)
        f_lib.setVerticalSpacing(14)
        self.edit_lib = QLineEdit("gcode_library")
        btn_browse = QPushButton("选择字库")
        btn_load = QPushButton("加载字库")
        row = QHBoxLayout()
        row.setSpacing(10)
        row.addWidget(self.edit_lib, 1)
        row.addWidget(btn_browse)
        row.addWidget(btn_load)
        self.lbl_count = QLabel("已加载字数：0")
        f_lib.addRow("字库路径", row)
        f_lib.addRow(self.lbl_count)
        left_lay.addWidget(g_lib)

        g_text = QGroupBox("文本输入")
        v_text = QVBoxLayout(g_text)
        v_text.setSpacing(10)
        self.text_input = QPlainTextEdit()
        self.text_input.setPlaceholderText("请输入汉字、标点和换行；支持粘贴。")
        self.text_input.setMinimumHeight(300)
        self.lbl_missing = QLabel("不支持字符：无")
        v_text.addWidget(self.text_input)
        v_text.addWidget(self.lbl_missing)
        left_lay.addWidget(g_text)

        g_param = QGroupBox("参数设置")
        f_param = QFormLayout(g_param)
        f_param.setHorizontalSpacing(14)
        f_param.setVerticalSpacing(14)
        self.sp_x = self._spin_mm(-1000, 3000, 0.0, 1.0, 1)
        self.sp_y = self._spin_mm(-1000, 3000, 0.0, 1.0, 1)
        self.sp_font = self._spin_mm(1, 300, 5.0, 0.5, 1)
        self.sp_char = self._spin_mm(-50, 200, 1.0, 0.5, 1)
        self.sp_line = self._spin_mm(0, 300, 2.0, 0.5, 1)
        self.cmb_align = QComboBox()
        self.cmb_align.addItems(["左对齐", "居中", "右对齐"])
        self.cmb_missing = QComboBox()
        self.cmb_missing.addItems(["跳过", "占位", "报错"])
        f_param.addRow("起始点 X (mm)", self.sp_x)
        f_param.addRow("起始点 Y (mm)", self.sp_y)
        f_param.addRow("字体大小 (mm)", self.sp_font)
        f_param.addRow("字间距 (mm)", self.sp_char)
        f_param.addRow("行间距 (mm)", self.sp_line)
        f_param.addRow("对齐方式", self.cmb_align)
        f_param.addRow("缺字处理", self.cmb_missing)
        left_lay.addWidget(g_param)

        row_btn = QHBoxLayout()
        row_btn.setSpacing(12)
        self.btn_preview = QPushButton("生成预览")
        self.btn_export = QPushButton("导出综合 G-code")
        self.btn_export.setObjectName("btnExport")
        row_btn.addWidget(self.btn_preview)
        row_btn.addWidget(self.btn_export)
        left_lay.addLayout(row_btn)
        left_lay.addStretch(1)

        self.preview = PreviewWidget()
        self.preview.setFrameShape(QFrame.NoFrame)
        right_lay.addWidget(self.preview, 7)

        g_stats = QGroupBox("统计信息")
        v_stats = QVBoxLayout(g_stats)
        self.text_stats = QTextEdit()
        self.text_stats.setReadOnly(True)
        self.text_stats.setMinimumHeight(250)
        v_stats.addWidget(self.text_stats)
        right_lay.addWidget(g_stats, 3)

        btn_browse.clicked.connect(self.on_browse)
        btn_load.clicked.connect(self.on_load_library)
        self.btn_preview.clicked.connect(self.on_preview)
        self.btn_export.clicked.connect(self.on_export)
        self.text_input.textChanged.connect(self.on_text_changed)

    def _apply_theme(self) -> None:
        self.setStyleSheet(
            """
            QWidget { background: #0F172A; color: #E2E8F0; font-size: 34px; }
            QGroupBox {
                border: 1px solid #334155;
                border-radius: 12px;
                margin-top: 14px;
                padding: 16px 14px 14px 14px;
                background: #0B1220;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 14px;
                color: #93C5FD;
                font-size: 38px;
                font-weight: 700;
                padding: 0 4px;
            }
            QPushButton {
                background: #1D4ED8;
                border: none;
                border-radius: 10px;
                padding: 14px 18px;
                color: #EFF6FF;
                font-size: 34px;
                font-weight: 600;
            }
            QPushButton:hover { background: #2563EB; }
            QPushButton#btnExport { background: #0F766E; }
            QPushButton#btnExport:hover { background: #0D9488; }
            QLineEdit, QPlainTextEdit, QTextEdit, QComboBox, QDoubleSpinBox {
                background: #111827;
                border: 1px solid #334155;
                border-radius: 10px;
                padding: 8px;
                font-size: 34px;
            }
            QLabel { font-size: 34px; }
            QSplitter::handle { background: #334155; }
            """
        )

    def _load_last_state(self) -> None:
        self.edit_lib.setText(str(self.settings.value("library_path", "gcode_library")))
        self.text_input.setPlainText(str(self.settings.value("last_text", "你好，写字机器人\n毕业设计演示")))

    def _save_last_state(self) -> None:
        self.settings.setValue("library_path", self.edit_lib.text().strip())
        self.settings.setValue("last_text", self.text_input.toPlainText())

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._save_last_state()
        super().closeEvent(event)

    def on_browse(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择 G-code 字库目录", self.edit_lib.text().strip() or ".")
        if folder:
            self.edit_lib.setText(folder)

    def on_load_library(self) -> None:
        folder = self.edit_lib.text().strip()
        try:
            n = self.library.load_folder(folder)
            self.lbl_count.setText(f"已加载字数：{n}")
            self.settings.setValue("library_path", folder)
            self.on_text_changed()
        except Exception as e:
            QMessageBox.critical(self, "字库加载失败", str(e))

    def on_text_changed(self) -> None:
        text = self.text_input.toPlainText()
        missing = []
        for ch in text:
            if ch in ("\n", "\r", " ", "\t"):
                continue
            if not self.library.has_char(ch):
                missing.append(ch)
        unique = "".join(dict.fromkeys(missing))
        self.lbl_missing.setText(f"不支持字符：{unique if unique else '无'}")

    def _build_layout(self) -> LayoutResult:
        if not self.library.glyphs:
            raise RuntimeError("请先加载字库。")
        text = self.text_input.toPlainText()
        if not text.strip():
            raise RuntimeError("请输入文本。")

        align_map = {"左对齐": "left", "居中": "center", "右对齐": "right"}
        miss_map = {"跳过": "skip", "占位": "placeholder", "报错": "error"}

        return self.layout_engine.layout(
            library=self.library,
            text=text,
            origin_x=self.sp_x.value(),
            origin_y=self.sp_y.value(),
            font_size=self.sp_font.value(),
            char_spacing=self.sp_char.value(),
            line_spacing=self.sp_line.value(),
            align=align_map.get(self.cmb_align.currentText(), "left"),
            missing_policy=miss_map.get(self.cmb_missing.currentText(), "skip"),
            page_width=self.page_w,
            page_height=self.page_h,
        )

    def _stats_text(self, r: LayoutResult) -> str:
        miss = "".join(r.missing_chars) if r.missing_chars else "无"
        lc = ", ".join(str(x) for x in r.line_counts) if r.line_counts else "0"
        return (
            f"总字数：{r.total_chars}\n"
            f"缺失字数：{r.missing_count}\n"
            f"缺失字符：{miss}\n"
            f"页面占用宽度：{r.used_width:.2f} mm\n"
            f"页面占用高度：{r.used_height:.2f} mm\n"
            f"落笔路径长度：{r.draw_length:.2f} mm\n"
            f"空走路径长度：{r.travel_length:.2f} mm\n"
            f"总路径长度：{(r.draw_length + r.travel_length):.2f} mm\n"
            f"预计抬笔次数：{r.pen_lifts}\n"
            f"每行字符数：[{lc}]\n"
            f"导出文件：{r.export_file or '尚未导出'}\n"
            f"警告：{'; '.join(r.warnings) if r.warnings else '无'}"
        )

    def on_preview(self) -> None:
        try:
            self.result = self._build_layout()
            self.preview.draw_layout(self.result, self.page_w, self.page_h, show_travel=True)
            self.text_stats.setPlainText(self._stats_text(self.result))
        except Exception as e:
            QMessageBox.warning(self, "预览失败", str(e))

    def _compose_gcode(self, r: LayoutResult) -> str:
        lines = [
            "; ===============================================",
            "; Combined G-code generated by UI.py",
            f"; Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"; Total chars: {r.total_chars}",
            f"; Missing chars: {r.missing_count}",
            f"; Missing list: {''.join(r.missing_chars) if r.missing_chars else 'None'}",
            f"; Draw length: {r.draw_length:.3f} mm",
            f"; Travel length: {r.travel_length:.3f} mm",
            "; ===============================================",
            "G21 ; mm",
            "G90 ; absolute",
            f"G0 F{self.rapid:.1f}",
            f"G1 F{self.feed:.1f}",
            f"G0 Z{self.z_up:.3f}",
        ]
        for st in r.output_strokes:
            if len(st) < 2:
                continue
            lines.append(f"G0 X{st[0][0]:.3f} Y{st[0][1]:.3f}")
            lines.append(f"G1 Z{self.z_down:.3f}")
            for p in st[1:]:
                lines.append(f"G1 X{p[0]:.3f} Y{p[1]:.3f}")
            lines.append(f"G0 Z{self.z_up:.3f}")
        lines.append("M2")
        return "\n".join(lines) + "\n"

    def on_export(self) -> None:
        try:
            if not self.result.output_strokes:
                self.result = self._build_layout()
            if not self.result.output_strokes:
                raise RuntimeError("没有可导出的轨迹。")

            name = f"combined_{datetime.now().strftime('%Y%m%d_%H%M%S')}.gcode"
            out_path, _ = QFileDialog.getSaveFileName(
                self,
                "导出综合 G-code",
                str(Path.cwd() / name),
                "G-code Files (*.gcode *.nc *.tap);;All Files (*)",
            )
            if not out_path:
                return

            Path(out_path).write_text(self._compose_gcode(self.result), encoding="utf-8")
            self.result.export_file = out_path
            self.text_stats.setPlainText(self._stats_text(self.result))
            QMessageBox.information(self, "导出成功", f"已导出:\n{out_path}")
        except Exception as e:
            QMessageBox.warning(self, "导出失败", str(e))


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("Text G-code Composer")
    app.setFont(QFont("Microsoft YaHei", 30))
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
