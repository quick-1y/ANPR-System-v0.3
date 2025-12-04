import cv2
from typing import Dict, List, Optional, Tuple

from PyQt5 import QtCore, QtGui, QtWidgets

from anpr.workers.channel_worker import ChannelWorker
from logging_manager import get_logger
from settings_manager import SettingsManager
from storage import EventDatabase

logger = get_logger(__name__)


class ChannelView(QtWidgets.QWidget):
    """Отображает поток канала с подсказками и индикатором движения."""

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

        self.video_label = QtWidgets.QLabel("Нет сигнала")
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.video_label.setStyleSheet(
            "background-color: #1c1c1c; color: #ccc; border: 1px solid #444; padding: 4px;"
        )
        self.video_label.setMinimumSize(220, 170)
        self.video_label.setScaledContents(False)
        self.video_label.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.video_label)

        self.motion_indicator = QtWidgets.QLabel("Движение")
        self.motion_indicator.setParent(self.video_label)
        self.motion_indicator.setStyleSheet(
            "background-color: rgba(220, 53, 69, 0.85); color: white;"
            "padding: 3px 6px; border-radius: 6px; font-weight: bold;"
        )
        self.motion_indicator.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.motion_indicator.hide()

        self.status_hint = QtWidgets.QLabel("")
        self.status_hint.setParent(self.video_label)
        self.status_hint.setStyleSheet(
            "background-color: rgba(0, 0, 0, 0.55); color: #ddd; padding: 2px 4px;"
        )
        self.status_hint.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.status_hint.hide()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        rect = self.video_label.contentsRect()
        margin = 8
        indicator_size = self.motion_indicator.sizeHint()
        self.motion_indicator.move(
            rect.right() - indicator_size.width() - margin, rect.top() + margin
        )
        status_size = self.status_hint.sizeHint()
        self.status_hint.move(rect.left() + margin, rect.bottom() - status_size.height() - margin)

    def set_pixmap(self, pixmap: QtGui.QPixmap) -> None:
        self.video_label.setPixmap(pixmap)

    def set_motion_active(self, active: bool) -> None:
        self.motion_indicator.setVisible(active)

    def set_status(self, text: str) -> None:
        self.status_hint.setVisible(bool(text))
        self.status_hint.setText(text)
        if text:
            self.status_hint.adjustSize()


class ROIEditor(QtWidgets.QLabel):
    """Виджет предпросмотра канала с настраиваемой областью распознавания."""

    roi_changed = QtCore.pyqtSignal(dict)

    def __init__(self) -> None:
        super().__init__("Нет кадра")
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setMinimumSize(400, 260)
        self.setStyleSheet(
            "background-color: #111; color: #888; border: 1px solid #444; padding: 6px;"
        )
        self._roi = {"x": 0, "y": 0, "width": 100, "height": 100}
        self._pixmap: Optional[QtGui.QPixmap] = None
        self._rubber_band = QtWidgets.QRubberBand(QtWidgets.QRubberBand.Rectangle, self)
        self._origin: Optional[QtCore.QPoint] = None

    def set_roi(self, roi: Dict[str, int]) -> None:
        self._roi = {
            "x": int(roi.get("x", 0)),
            "y": int(roi.get("y", 0)),
            "width": int(roi.get("width", 100)),
            "height": int(roi.get("height", 100)),
        }
        self._roi["width"] = min(self._roi["width"], max(1, 100 - self._roi["x"]))
        self._roi["height"] = min(self._roi["height"], max(1, 100 - self._roi["y"]))
        self.update()

    def setPixmap(self, pixmap: Optional[QtGui.QPixmap]) -> None:  # noqa: N802
        self._pixmap = pixmap
        if pixmap is None:
            super().setPixmap(QtGui.QPixmap())
            self.setText("Нет кадра")
            return
        scaled = self._scaled_pixmap(self.size())
        super().setPixmap(scaled)
        self.setText("")

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._pixmap:
            super().setPixmap(self._scaled_pixmap(event.size()))

    def _scaled_pixmap(self, size: QtCore.QSize) -> QtGui.QPixmap:
        assert self._pixmap is not None
        return self._pixmap.scaled(
            size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        )

    def _image_geometry(self) -> Optional[Tuple[QtCore.QPoint, QtCore.QSize]]:
        if self._pixmap is None:
            return None
        pixmap = self._scaled_pixmap(self.size())
        area = self.contentsRect()
        x = area.x() + (area.width() - pixmap.width()) // 2
        y = area.y() + (area.height() - pixmap.height()) // 2
        return QtCore.QPoint(x, y), pixmap.size()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        super().paintEvent(event)
        geom = self._image_geometry()
        if geom is None:
            return
        offset, size = geom
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        roi_rect = QtCore.QRect(
            offset.x() + int(size.width() * self._roi["x"] / 100),
            offset.y() + int(size.height() * self._roi["y"] / 100),
            int(size.width() * self._roi["width"] / 100),
            int(size.height() * self._roi["height"] / 100),
        )
        pen = QtGui.QPen(QtGui.QColor(0, 200, 0))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setBrush(QtGui.QColor(0, 200, 0, 40))
        painter.drawRect(roi_rect)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        geom = self._image_geometry()
        if geom is None:
            return
        offset, size = geom
        area_rect = QtCore.QRect(offset, size)
        if not area_rect.contains(event.pos()):
            return
        self._origin = event.pos()
        self._rubber_band.setGeometry(QtCore.QRect(self._origin, QtCore.QSize()))
        self._rubber_band.show()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if self._origin is None:
            return
        rect = QtCore.QRect(self._origin, event.pos()).normalized()
        self._rubber_band.setGeometry(rect)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if self._origin is None:
            return
        geom = self._image_geometry()
        self._rubber_band.hide()
        if geom is None:
            self._origin = None
            return
        offset, size = geom
        selection = self._rubber_band.geometry().intersected(QtCore.QRect(offset, size))
        if selection.isValid() and selection.width() > 5 and selection.height() > 5:
            x_pct = max(0, min(100, int((selection.left() - offset.x()) * 100 / size.width())))
            y_pct = max(0, min(100, int((selection.top() - offset.y()) * 100 / size.height())))
            w_pct = max(1, min(100 - x_pct, int(selection.width() * 100 / size.width())))
            h_pct = max(1, min(100 - y_pct, int(selection.height() * 100 / size.height())))
            self._roi = {"x": x_pct, "y": y_pct, "width": w_pct, "height": h_pct}
            self.roi_changed.emit(self._roi)
        self._origin = None
        self.update()


class MainWindow(QtWidgets.QMainWindow):
    """Главное окно приложения ANPR с вкладками мониторинга, событий, поиска и настроек."""

    GRID_VARIANTS = ["1x1", "1x2", "2x2", "2x3", "3x3"]

    def __init__(self, settings: Optional[SettingsManager] = None) -> None:
        super().__init__()
        self.setWindowTitle("ANPR Desktop")
        self.resize(1280, 800)

        self.settings = settings or SettingsManager()
        self.db = EventDatabase(self.settings.get_db_path())

        self.channel_workers: List[ChannelWorker] = []
        self.channel_labels: Dict[str, ChannelView] = {}

        self.tabs = QtWidgets.QTabWidget()
        self.monitor_tab = self._build_monitor_tab()
        self.events_tab = self._build_events_tab()
        self.search_tab = self._build_search_tab()
        self.settings_tab = self._build_settings_tab()

        self.tabs.addTab(self.monitor_tab, "Монитор")
        self.tabs.addTab(self.events_tab, "События")
        self.tabs.addTab(self.search_tab, "Поиск")
        self.tabs.addTab(self.settings_tab, "Настройки")

        self.setCentralWidget(self.tabs)
        self._refresh_events_table()

    # ------------------ Мониторинг ------------------
    def _build_monitor_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("Сетка:"))
        self.grid_selector = QtWidgets.QComboBox()
        self.grid_selector.addItems(self.GRID_VARIANTS)
        self.grid_selector.setCurrentText(self.settings.get_grid())
        self.grid_selector.currentTextChanged.connect(self._on_grid_changed)
        controls.addWidget(self.grid_selector)

        self.start_button = QtWidgets.QPushButton("Запустить")
        self.start_button.clicked.connect(self._start_channels)
        controls.addWidget(self.start_button)

        controls.addStretch()
        controls.addWidget(QtWidgets.QLabel("Последнее событие:"))
        self.last_event_label = QtWidgets.QLabel("—")
        controls.addWidget(self.last_event_label)

        layout.addLayout(controls)

        self.grid_widget = QtWidgets.QWidget()
        self.grid_layout = QtWidgets.QGridLayout(self.grid_widget)
        self.grid_layout.setSpacing(6)
        layout.addWidget(self.grid_widget)

        self._draw_grid()
        return widget

    @staticmethod
    def _prepare_optional_datetime(widget: QtWidgets.QDateTimeEdit) -> None:
        widget.setCalendarPopup(True)
        widget.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        min_dt = QtCore.QDateTime.fromSecsSinceEpoch(0)
        widget.setMinimumDateTime(min_dt)
        widget.setSpecialValueText("Не выбрано")
        widget.setDateTime(min_dt)

    @staticmethod
    def _get_datetime_value(widget: QtWidgets.QDateTimeEdit) -> Optional[str]:
        if widget.dateTime() == widget.minimumDateTime():
            return None
        return widget.dateTime().toString(QtCore.Qt.ISODate)

    def _draw_grid(self) -> None:
        for i in reversed(range(self.grid_layout.count())):
            item = self.grid_layout.takeAt(i)
            widget = item.widget()
            if widget:
                widget.setParent(None)

        self.channel_labels.clear()
        channels = self.settings.get_channels()
        rows, cols = map(int, self.grid_selector.currentText().split("x"))
        index = 0
        for row in range(rows):
            for col in range(cols):
                label = ChannelView(f"Канал {index+1}")
                if index < len(channels):
                    channel_name = channels[index].get("name", f"Канал {index+1}")
                    self.channel_labels[channel_name] = label
                self.grid_layout.addWidget(label, row, col)
                index += 1

    def _on_grid_changed(self, grid: str) -> None:
        self.settings.save_grid(grid)
        self._draw_grid()

    def _start_channels(self) -> None:
        self._stop_workers()
        self.channel_workers = []
        for channel_conf in self.settings.get_channels():
            worker = ChannelWorker(channel_conf, self.settings.get_db_path())
            worker.frame_ready.connect(self._update_frame)
            worker.event_ready.connect(self._handle_event)
            worker.status_ready.connect(self._handle_status)
            self.channel_workers.append(worker)
            worker.start()

    def _stop_workers(self) -> None:
        for worker in self.channel_workers:
            worker.stop()
            worker.wait(1000)
        self.channel_workers = []

    def _update_frame(self, channel_name: str, image: QtGui.QImage) -> None:
        label = self.channel_labels.get(channel_name)
        if not label:
            return
        target_size = label.video_label.contentsRect().size()
        pixmap = QtGui.QPixmap.fromImage(image).scaled(
            target_size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        )
        label.set_pixmap(pixmap)

    def _handle_event(self, event: Dict) -> None:
        self.last_event_label.setText(
            f"{event['timestamp']} | {event['channel']} | {event['plate']} | {event['confidence']:.2f}"
        )
        self._refresh_events_table()

    def _handle_status(self, channel: str, status: str) -> None:
        label = self.channel_labels.get(channel)
        if label:
            normalized = status.lower()
            if "движ" in normalized or "motion" in normalized:
                label.set_status("")
            else:
                label.set_status(status)
            label.set_motion_active("обнаружено" in normalized)

    # ------------------ События ------------------
    def _build_events_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        filters = QtWidgets.QHBoxLayout()
        filters.addWidget(QtWidgets.QLabel("Дата с:"))
        self.events_from = QtWidgets.QDateTimeEdit()
        self._prepare_optional_datetime(self.events_from)
        filters.addWidget(self.events_from)

        filters.addWidget(QtWidgets.QLabel("по:"))
        self.events_to = QtWidgets.QDateTimeEdit()
        self._prepare_optional_datetime(self.events_to)
        filters.addWidget(self.events_to)

        filters.addWidget(QtWidgets.QLabel("Канал:"))
        self.events_channel = QtWidgets.QComboBox()
        self.events_channel.addItem("Все", "")
        for channel in self.settings.get_channels():
            self.events_channel.addItem(channel.get("name", ""), channel.get("name", ""))
        filters.addWidget(self.events_channel)

        filters.addWidget(QtWidgets.QLabel("Список номеров (через запятую):"))
        self.events_plate_list = QtWidgets.QLineEdit()
        filters.addWidget(self.events_plate_list)

        apply_btn = QtWidgets.QPushButton("Применить")
        apply_btn.clicked.connect(self._refresh_events_table)
        filters.addWidget(apply_btn)

        layout.addLayout(filters)

        self.events_table = QtWidgets.QTableWidget(0, 5)
        self.events_table.setHorizontalHeaderLabels(
            ["Время", "Канал", "Номер", "Уверенность", "Источник"]
        )
        self.events_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.events_table)

        return widget

    def _refresh_events_table(self) -> None:
        start = self._get_datetime_value(self.events_from)
        end = self._get_datetime_value(self.events_to)
        channel = self.events_channel.currentData() if hasattr(self, "events_channel") else None
        plates_input = self.events_plate_list.text() if hasattr(self, "events_plate_list") else ""
        plates = [plate.strip() for plate in plates_input.split(",") if plate.strip()]

        rows = self.db.fetch_filtered(start=start or None, end=end or None, channel=channel or None, plates=plates)
        self.events_table.setRowCount(0)
        for row_data in rows:
            row_index = self.events_table.rowCount()
            self.events_table.insertRow(row_index)
            self.events_table.setItem(row_index, 0, QtWidgets.QTableWidgetItem(row_data["timestamp"]))
            self.events_table.setItem(row_index, 1, QtWidgets.QTableWidgetItem(row_data["channel"]))
            self.events_table.setItem(row_index, 2, QtWidgets.QTableWidgetItem(row_data["plate"]))
            self.events_table.setItem(
                row_index, 3, QtWidgets.QTableWidgetItem(f"{row_data['confidence'] or 0:.2f}")
            )
            self.events_table.setItem(row_index, 4, QtWidgets.QTableWidgetItem(row_data["source"]))

    # ------------------ Поиск ------------------
    def _build_search_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        form = QtWidgets.QFormLayout()
        self.search_plate = QtWidgets.QLineEdit()
        self.search_from = QtWidgets.QDateTimeEdit()
        self._prepare_optional_datetime(self.search_from)
        self.search_to = QtWidgets.QDateTimeEdit()
        self._prepare_optional_datetime(self.search_to)

        form.addRow("Номер:", self.search_plate)
        form.addRow("Дата с:", self.search_from)
        form.addRow("Дата по:", self.search_to)
        layout.addLayout(form)

        search_btn = QtWidgets.QPushButton("Искать")
        search_btn.clicked.connect(self._run_plate_search)
        layout.addWidget(search_btn)

        self.search_table = QtWidgets.QTableWidget(0, 5)
        self.search_table.setHorizontalHeaderLabels(
            ["Время", "Канал", "Номер", "Уверенность", "Источник"]
        )
        self.search_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.search_table)

        return widget

    def _run_plate_search(self) -> None:
        start = self._get_datetime_value(self.search_from)
        end = self._get_datetime_value(self.search_to)
        plate_fragment = self.search_plate.text()
        rows = self.db.search_by_plate(plate_fragment, start=start or None, end=end or None)
        self.search_table.setRowCount(0)
        for row_data in rows:
            row_index = self.search_table.rowCount()
            self.search_table.insertRow(row_index)
            self.search_table.setItem(row_index, 0, QtWidgets.QTableWidgetItem(row_data["timestamp"]))
            self.search_table.setItem(row_index, 1, QtWidgets.QTableWidgetItem(row_data["channel"]))
            self.search_table.setItem(row_index, 2, QtWidgets.QTableWidgetItem(row_data["plate"]))
            self.search_table.setItem(
                row_index, 3, QtWidgets.QTableWidgetItem(f"{row_data['confidence'] or 0:.2f}")
            )
            self.search_table.setItem(row_index, 4, QtWidgets.QTableWidgetItem(row_data["source"]))

    # ------------------ Настройки ------------------
    def _build_settings_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)

        left_panel = QtWidgets.QVBoxLayout()
        left_panel.setSpacing(6)
        self.channels_list = QtWidgets.QListWidget()
        self.channels_list.setFixedWidth(180)
        self.channels_list.currentRowChanged.connect(self._load_channel_form)
        left_panel.addWidget(self.channels_list)

        list_buttons = QtWidgets.QHBoxLayout()
        add_btn = QtWidgets.QPushButton("Добавить")
        add_btn.clicked.connect(self._add_channel)
        remove_btn = QtWidgets.QPushButton("Удалить")
        remove_btn.clicked.connect(self._remove_channel)
        list_buttons.addWidget(add_btn)
        list_buttons.addWidget(remove_btn)
        left_panel.addLayout(list_buttons)
        layout.addLayout(left_panel)

        form_container = QtWidgets.QVBoxLayout()
        self.preview = ROIEditor()
        self.preview.roi_changed.connect(self._on_roi_drawn)
        form_container.addWidget(self.preview)

        # Блок настроек канала
        channel_group = QtWidgets.QGroupBox("Канал")
        channel_form = QtWidgets.QFormLayout(channel_group)
        self.channel_name_input = QtWidgets.QLineEdit()
        self.channel_source_input = QtWidgets.QLineEdit()
        channel_form.addRow("Название:", self.channel_name_input)
        channel_form.addRow("Источник/RTSP:", self.channel_source_input)
        form_container.addWidget(channel_group)

        # Блок распознавания
        recognition_group = QtWidgets.QGroupBox("Распознавание")
        recognition_form = QtWidgets.QFormLayout(recognition_group)
        self.best_shots_input = QtWidgets.QSpinBox()
        self.best_shots_input.setRange(1, 50)
        self.best_shots_input.setToolTip("Количество бестшотов, участвующих в консенсусе трека")
        recognition_form.addRow("Бестшоты на трек:", self.best_shots_input)

        self.cooldown_input = QtWidgets.QSpinBox()
        self.cooldown_input.setRange(0, 3600)
        self.cooldown_input.setToolTip(
            "Интервал (в секундах), в течение которого не создается повторное событие для того же номера"
        )
        recognition_form.addRow("Пауза повтора (сек):", self.cooldown_input)

        self.min_conf_input = QtWidgets.QDoubleSpinBox()
        self.min_conf_input.setRange(0.0, 1.0)
        self.min_conf_input.setSingleStep(0.05)
        self.min_conf_input.setDecimals(2)
        self.min_conf_input.setToolTip(
            "Минимальная уверенность OCR (0-1) для приема результата; ниже — помечается как нечитаемое"
        )
        recognition_form.addRow("Мин. уверенность OCR:", self.min_conf_input)
        form_container.addWidget(recognition_group)

        # Блок детектора движения
        motion_group = QtWidgets.QGroupBox("Детектор движения")
        motion_form = QtWidgets.QFormLayout(motion_group)
        self.detection_mode_input = QtWidgets.QComboBox()
        self.detection_mode_input.addItem("Постоянное", "continuous")
        self.detection_mode_input.addItem("Детектор движения", "motion")
        motion_form.addRow("Обнаружение ТС:", self.detection_mode_input)

        self.detector_stride_input = QtWidgets.QSpinBox()
        self.detector_stride_input.setRange(1, 12)
        self.detector_stride_input.setToolTip(
            "Запускать YOLO на каждом N-м кадре в зоне распознавания, чтобы снизить нагрузку"
        )
        motion_form.addRow("Шаг инференса (кадр):", self.detector_stride_input)

        self.motion_threshold_input = QtWidgets.QDoubleSpinBox()
        self.motion_threshold_input.setRange(0.0, 1.0)
        self.motion_threshold_input.setDecimals(3)
        self.motion_threshold_input.setSingleStep(0.005)
        self.motion_threshold_input.setToolTip("Порог чувствительности по площади изменения внутри ROI")
        motion_form.addRow("Порог движения:", self.motion_threshold_input)

        self.motion_stride_input = QtWidgets.QSpinBox()
        self.motion_stride_input.setRange(1, 30)
        self.motion_stride_input.setToolTip("Обрабатывать каждый N-й кадр для поиска движения")
        motion_form.addRow("Частота анализа (кадр):", self.motion_stride_input)

        self.motion_activation_frames_input = QtWidgets.QSpinBox()
        self.motion_activation_frames_input.setRange(1, 60)
        self.motion_activation_frames_input.setToolTip("Сколько кадров подряд должно быть движение, чтобы включить распознавание")
        motion_form.addRow("Мин. кадров с движением:", self.motion_activation_frames_input)

        self.motion_release_frames_input = QtWidgets.QSpinBox()
        self.motion_release_frames_input.setRange(1, 120)
        self.motion_release_frames_input.setToolTip("Сколько кадров без движения нужно, чтобы остановить распознавание")
        motion_form.addRow("Мин. кадров без движения:", self.motion_release_frames_input)
        form_container.addWidget(motion_group)

        # Блок зоны распознавания
        roi_group = QtWidgets.QGroupBox("Зона распознавания")
        roi_layout = QtWidgets.QGridLayout()
        self.roi_x_input = QtWidgets.QSpinBox()
        self.roi_x_input.setRange(0, 100)
        self.roi_y_input = QtWidgets.QSpinBox()
        self.roi_y_input.setRange(0, 100)
        self.roi_w_input = QtWidgets.QSpinBox()
        self.roi_w_input.setRange(1, 100)
        self.roi_h_input = QtWidgets.QSpinBox()
        self.roi_h_input.setRange(1, 100)

        for spin in (self.roi_x_input, self.roi_y_input, self.roi_w_input, self.roi_h_input):
            spin.valueChanged.connect(self._on_roi_inputs_changed)

        roi_layout.addWidget(QtWidgets.QLabel("X (%):"), 0, 0)
        roi_layout.addWidget(self.roi_x_input, 0, 1)
        roi_layout.addWidget(QtWidgets.QLabel("Y (%):"), 1, 0)
        roi_layout.addWidget(self.roi_y_input, 1, 1)
        roi_layout.addWidget(QtWidgets.QLabel("Ширина (%):"), 2, 0)
        roi_layout.addWidget(self.roi_w_input, 2, 1)
        roi_layout.addWidget(QtWidgets.QLabel("Высота (%):"), 3, 0)
        roi_layout.addWidget(self.roi_h_input, 3, 1)
        refresh_btn = QtWidgets.QPushButton("Обновить кадр")
        refresh_btn.clicked.connect(self._refresh_preview_frame)
        roi_layout.addWidget(refresh_btn, 4, 0, 1, 2)
        roi_group.setLayout(roi_layout)
        form_container.addWidget(roi_group)

        save_btn = QtWidgets.QPushButton("Сохранить")
        save_btn.clicked.connect(self._save_channel)
        form_container.addWidget(save_btn)
        form_container.addStretch()

        layout.addLayout(form_container)

        self._reload_channels_list()
        return widget

    def _reload_channels_list(self) -> None:
        self.channels_list.clear()
        for channel in self.settings.get_channels():
            self.channels_list.addItem(channel.get("name", "Канал"))
        if self.channels_list.count():
            self.channels_list.setCurrentRow(0)

    def _load_channel_form(self, index: int) -> None:
        channels = self.settings.get_channels()
        if 0 <= index < len(channels):
            channel = channels[index]
            self.channel_name_input.setText(channel.get("name", ""))
            self.channel_source_input.setText(channel.get("source", ""))
            self.best_shots_input.setValue(int(channel.get("best_shots", 3)))
            self.cooldown_input.setValue(int(channel.get("cooldown_seconds", 5)))
            self.min_conf_input.setValue(float(channel.get("ocr_min_confidence", 0.6)))

            self.motion_threshold_input.setValue(float(channel.get("motion_threshold", 0.01)))
            self.motion_stride_input.setValue(int(channel.get("motion_frame_stride", 1)))
            self.motion_activation_frames_input.setValue(int(channel.get("motion_activation_frames", 3)))
            self.motion_release_frames_input.setValue(int(channel.get("motion_release_frames", 6)))

            mode = channel.get("detection_mode", "continuous")
            mode_index = max(0, self.detection_mode_input.findData(mode))
            self.detection_mode_input.setCurrentIndex(mode_index)
            self.detector_stride_input.setValue(int(channel.get("detector_frame_stride", 2)))

            region = channel.get("region", {})
            self.roi_x_input.setValue(int(region.get("x", 0)))
            self.roi_y_input.setValue(int(region.get("y", 0)))
            self.roi_w_input.setValue(int(region.get("width", 100)))
            self.roi_h_input.setValue(int(region.get("height", 100)))
            self.preview.set_roi(
                {
                    "x": int(region.get("x", 0)),
                    "y": int(region.get("y", 0)),
                    "width": int(region.get("width", 100)),
                    "height": int(region.get("height", 100)),
                }
            )
            self._refresh_preview_frame()

    def _add_channel(self) -> None:
        channels = self.settings.get_channels()
        new_id = max([c.get("id", 0) for c in channels] + [0]) + 1
        channels.append(
            {
                "id": new_id,
                "name": f"Канал {new_id}",
                "source": "",
                "best_shots": self.settings.get_best_shots(),
                "cooldown_seconds": self.settings.get_cooldown_seconds(),
                "ocr_min_confidence": self.settings.get_min_confidence(),
                "region": {"x": 0, "y": 0, "width": 100, "height": 100},
                "detection_mode": "continuous",
                "detector_frame_stride": 2,
                "motion_threshold": 0.01,
                "motion_frame_stride": 1,
                "motion_activation_frames": 3,
                "motion_release_frames": 6,
            }
        )
        self.settings.save_channels(channels)
        self._reload_channels_list()
        self._draw_grid()

    def _remove_channel(self) -> None:
        index = self.channels_list.currentRow()
        channels = self.settings.get_channels()
        if 0 <= index < len(channels):
            channels.pop(index)
            self.settings.save_channels(channels)
            self._reload_channels_list()
            self._draw_grid()

    def _save_channel(self) -> None:
        index = self.channels_list.currentRow()
        channels = self.settings.get_channels()
        if 0 <= index < len(channels):
            channels[index]["name"] = self.channel_name_input.text()
            channels[index]["source"] = self.channel_source_input.text()
            channels[index]["best_shots"] = int(self.best_shots_input.value())
            channels[index]["cooldown_seconds"] = int(self.cooldown_input.value())
            channels[index]["ocr_min_confidence"] = float(self.min_conf_input.value())
            channels[index]["detection_mode"] = self.detection_mode_input.currentData()
            channels[index]["detector_frame_stride"] = int(self.detector_stride_input.value())
            channels[index]["motion_threshold"] = float(self.motion_threshold_input.value())
            channels[index]["motion_frame_stride"] = int(self.motion_stride_input.value())
            channels[index]["motion_activation_frames"] = int(self.motion_activation_frames_input.value())
            channels[index]["motion_release_frames"] = int(self.motion_release_frames_input.value())

            region = {
                "x": int(self.roi_x_input.value()),
                "y": int(self.roi_y_input.value()),
                "width": int(self.roi_w_input.value()),
                "height": int(self.roi_h_input.value()),
            }
            # Корректируем область, чтобы она не выходила за пределы кадра.
            region["width"] = min(region["width"], max(1, 100 - region["x"]))
            region["height"] = min(region["height"], max(1, 100 - region["y"]))
            channels[index]["region"] = region
            self.settings.save_channels(channels)
            self._reload_channels_list()
            self._draw_grid()

    def _on_roi_drawn(self, roi: Dict[str, int]) -> None:
        self.roi_x_input.blockSignals(True)
        self.roi_y_input.blockSignals(True)
        self.roi_w_input.blockSignals(True)
        self.roi_h_input.blockSignals(True)
        self.roi_x_input.setValue(roi["x"])
        self.roi_y_input.setValue(roi["y"])
        self.roi_w_input.setValue(roi["width"])
        self.roi_h_input.setValue(roi["height"])
        self.roi_x_input.blockSignals(False)
        self.roi_y_input.blockSignals(False)
        self.roi_w_input.blockSignals(False)
        self.roi_h_input.blockSignals(False)

    def _on_roi_inputs_changed(self) -> None:
        roi = {
            "x": int(self.roi_x_input.value()),
            "y": int(self.roi_y_input.value()),
            "width": int(self.roi_w_input.value()),
            "height": int(self.roi_h_input.value()),
        }
        roi["width"] = min(roi["width"], max(1, 100 - roi["x"]))
        roi["height"] = min(roi["height"], max(1, 100 - roi["y"]))
        self.preview.set_roi(roi)

    def _refresh_preview_frame(self) -> None:
        index = self.channels_list.currentRow()
        channels = self.settings.get_channels()
        if not (0 <= index < len(channels)):
            return
        source = str(channels[index].get("source", ""))
        if not source:
            self.preview.setPixmap(None)
            return
        capture = cv2.VideoCapture(int(source) if source.isnumeric() else source)
        ret, frame = capture.read()
        capture.release()
        if not ret or frame is None:
            self.preview.setPixmap(None)
            return
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, _ = rgb_frame.shape
        bytes_per_line = 3 * width
        q_image = QtGui.QImage(
            rgb_frame.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888
        ).copy()
        self.preview.setPixmap(QtGui.QPixmap.fromImage(q_image))

    # ------------------ Жизненный цикл ------------------
    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        self._stop_workers()
        event.accept()
