"""
Main PyQt5 application for Grain Size Calculator.
Provides a desktop GUI interface for SEM image analysis.
"""

import sys
import os
import traceback
import cv2
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QGridLayout, QLabel, QPushButton, 
                           QLineEdit, QTextEdit, QProgressBar, QFileDialog,
                           QTabWidget, QScrollArea, QGroupBox, QCheckBox,
                           QSpinBox, QDoubleSpinBox, QComboBox, QMessageBox,
                           QTableWidget, QTableWidgetItem, QSplitter, QListWidget,
                           QProgressDialog, QDialog)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QTime
from PyQt5.QtGui import QPixmap, QFont, QIcon, QPalette, QImage

# Import our core modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from core.config import ConfigManager, AnalysisVariant
from core.image_processing import load_and_convert_to_grayscale
from core.ocr import create_ocr_processor
from core.exact_footer_ocr import create_exact_footer_ocr
from core.pinhole_detection import create_pinhole_detector
from .analysis_worker import AnalysisWorker


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.config_manager = ConfigManager()
        self.current_image_path = ""
        self.image_list = []  # List of image paths for batch processing
        self.current_batch_index = 0  # Current processing index
        self.results = []
        self.exact_ocr = create_exact_footer_ocr()  # Initialize exact OCR processor
        self.pinhole_detector = create_pinhole_detector()  # Initialize pinhole detector
        self.pinhole_results = None  # Store pinhole detection results
        
        self.init_ui()
        self.setup_connections()
        
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Grain Size Calculator - SEM Image Analysis")
        self.setGeometry(100, 100, 1400, 900)
        
        # Set application style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin: 5px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Create left panel for controls
        left_panel = self.create_control_panel()
        
        # Create right panel for results and visualization
        right_panel = self.create_results_panel()
        
        # Create splitter to allow resizing
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)  # Left panel
        splitter.setStretchFactor(1, 2)  # Right panel gets more space
        
        main_layout.addWidget(splitter)
        
        # Create status bar
        self.statusBar().showMessage("Ready")
        
        # Create device info label
        self.device_label = QLabel("Detecting processing device...")
        self.device_label.setStyleSheet("color: #666; font-weight: bold;")
        self.statusBar().addPermanentWidget(self.device_label)
        
        # Create progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress_bar)
        
        # Initialize device detection
        QTimer.singleShot(1000, self.detect_device_info)
        
        # Set initial button states
        self.analyze_button.setEnabled(False)
        self.detect_pinholes_button.setEnabled(False)
    
    def create_control_panel(self):
        """Create the left control panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Image selection group
        image_group = QGroupBox("📂 Image Selection")
        image_layout = QVBoxLayout(image_group)
        
        # Single image selection
        single_layout = QHBoxLayout()
        self.image_path_edit = QLineEdit()
        self.image_path_edit.setPlaceholderText("Select SEM image file...")
        
        self.browse_button = QPushButton("Browse Single Image")
        self.browse_button.clicked.connect(self.browse_image)
        
        single_layout.addWidget(self.image_path_edit)
        single_layout.addWidget(self.browse_button)
        
        # Multiple images selection
        multi_layout = QHBoxLayout()
        self.browse_multi_button = QPushButton("📁 Browse Multiple Images")
        self.browse_multi_button.clicked.connect(self.browse_multiple_images)
        
        self.clear_images_button = QPushButton("🗑️ Clear All")
        self.clear_images_button.clicked.connect(self.clear_image_list)
        
        multi_layout.addWidget(self.browse_multi_button)
        multi_layout.addWidget(self.clear_images_button)
        
        # Image list display
        self.image_list_widget = QListWidget()
        self.image_list_widget.setMaximumHeight(120)
        self.image_list_widget.itemClicked.connect(self.on_image_list_selection)
        
        # Batch processing controls
        batch_layout = QHBoxLayout()
        self.automate_check = QCheckBox("🤖 Automate Processing")
        self.automate_check.setChecked(False)
        self.automate_check.setToolTip("Automatically detect pinholes and grains for all images without user input")
        
        self.process_all_button = QPushButton("🚀 Process All Images")
        self.process_all_button.clicked.connect(self.process_all_images)
        self.process_all_button.setEnabled(False)
        
        batch_layout.addWidget(self.automate_check)
        batch_layout.addWidget(self.process_all_button)
        
        image_layout.addWidget(QLabel("Single Image:"))
        image_layout.addLayout(single_layout)
        image_layout.addWidget(QLabel("Multiple Images:"))
        image_layout.addLayout(multi_layout)
        image_layout.addWidget(QLabel("Image Queue:"))
        image_layout.addWidget(self.image_list_widget)
        image_layout.addLayout(batch_layout)
        image_layout.addWidget(self.image_path_edit)
        image_layout.addWidget(self.browse_button)
        
        # OCR and Scale group
        ocr_group = QGroupBox("📐 Scale Detection")
        ocr_layout = QGridLayout(ocr_group)
        
        self.auto_ocr_check = QCheckBox("🔍 Auto-detect Frame Width from SEM footer")
        self.auto_ocr_check.setChecked(True)
        self.auto_ocr_check.setToolTip("Automatically extract Frame Width from SEM image footer using OCR")
        
        self.frame_width_spinbox = QDoubleSpinBox()
        self.frame_width_spinbox.setRange(0.1, 1000.0)
        self.frame_width_spinbox.setValue(21.8)
        self.frame_width_spinbox.setSuffix(" µm")
        self.frame_width_spinbox.setEnabled(False)
        self.frame_width_spinbox.setToolTip("Manual Frame Width input (enabled when auto-detection is off)")
        
        self.detect_scale_button = QPushButton("🔍 Detect Scale from Footer")
        self.detect_scale_button.clicked.connect(self.detect_scale)
        self.detect_scale_button.setToolTip("Manually trigger scale detection from SEM footer")
        
        # Status label for detected values
        self.scale_status_label = QLabel("Status: Auto-detection enabled")
        self.scale_status_label.setStyleSheet("color: #666666; font-style: italic;")
        
        ocr_layout.addWidget(self.auto_ocr_check, 0, 0, 1, 2)
        ocr_layout.addWidget(QLabel("📏 Frame Width:"), 1, 0)
        ocr_layout.addWidget(self.frame_width_spinbox, 1, 1)
        ocr_layout.addWidget(self.detect_scale_button, 2, 0, 1, 2)
        ocr_layout.addWidget(self.scale_status_label, 3, 0, 1, 2)
        
        # Connect checkbox to enable/disable manual input
        self.auto_ocr_check.toggled.connect(self.on_auto_ocr_toggled)
        
        # Pinhole Detection group
        pinhole_group = QGroupBox("🕳️ Pinhole Detection")
        pinhole_layout = QGridLayout(pinhole_group)
        
        self.detect_pinholes_button = QPushButton("🔍 Detect Pinholes")
        self.detect_pinholes_button.clicked.connect(self.detect_pinholes)
        self.detect_pinholes_button.setToolTip("Detect pinholes in the SEM image")
        
        self.pinhole_count_spinbox = QSpinBox()
        self.pinhole_count_spinbox.setRange(0, 9999)
        self.pinhole_count_spinbox.setValue(0)
        self.pinhole_count_spinbox.setSuffix(" pinholes")
        self.pinhole_count_spinbox.setEnabled(False)
        
        self.pinhole_status_label = QLabel("Status: Ready for detection")
        self.pinhole_status_label.setStyleSheet("color: #666666; font-style: italic;")
        
        self.show_pinhole_preview_button = QPushButton("🖼️ Show Preview")
        self.show_pinhole_preview_button.clicked.connect(self.show_pinhole_preview)
        self.show_pinhole_preview_button.setEnabled(False)
        
        self.confirm_pinholes_button = QPushButton("✅ Confirm Count")
        self.confirm_pinholes_button.clicked.connect(self.confirm_pinhole_count)
        self.confirm_pinholes_button.setEnabled(False)
        
        pinhole_layout.addWidget(self.detect_pinholes_button, 0, 0, 1, 2)
        pinhole_layout.addWidget(QLabel("🔢 Pinhole Count:"), 1, 0)
        pinhole_layout.addWidget(self.pinhole_count_spinbox, 1, 1)
        pinhole_layout.addWidget(self.show_pinhole_preview_button, 2, 0)
        pinhole_layout.addWidget(self.confirm_pinholes_button, 2, 1)
        pinhole_layout.addWidget(self.pinhole_status_label, 3, 0, 1, 2)
        
        # Processing options group
        processing_group = QGroupBox("Processing Options")
        processing_layout = QGridLayout(processing_group)
        
        self.min_area_spinbox = QSpinBox()
        self.min_area_spinbox.setRange(1, 1000)
        self.min_area_spinbox.setValue(50)
        self.min_area_spinbox.setSuffix(" px")
        
        self.apply_cap_check = QCheckBox("Apply Feret diameter cap")
        self.apply_cap_check.setChecked(True)
        
        self.feret_cap_spinbox = QDoubleSpinBox()
        self.feret_cap_spinbox.setRange(0.1, 100.0)
        self.feret_cap_spinbox.setValue(5.0)
        self.feret_cap_spinbox.setSuffix(" µm")
        
        self.save_overlays_check = QCheckBox("Save overlay images")
        self.save_overlays_check.setChecked(True)
        
        self.annotate_check = QCheckBox("Annotate measurements")
        self.annotate_check.setChecked(True)
        
        processing_layout.addWidget(QLabel("Min grain area:"), 0, 0)
        processing_layout.addWidget(self.min_area_spinbox, 0, 1)
        processing_layout.addWidget(self.apply_cap_check, 1, 0, 1, 2)
        processing_layout.addWidget(QLabel("Feret cap:"), 2, 0)
        processing_layout.addWidget(self.feret_cap_spinbox, 2, 1)
        processing_layout.addWidget(self.save_overlays_check, 3, 0, 1, 2)
        processing_layout.addWidget(self.annotate_check, 4, 0, 1, 2)
        
        # Connect cap checkbox to enable/disable spinbox
        self.apply_cap_check.toggled.connect(self.feret_cap_spinbox.setEnabled)
        
        # Analysis control group
        analysis_group = QGroupBox("🚀 Analysis Control")
        analysis_layout = QVBoxLayout(analysis_group)
        
        self.analyze_button = QPushButton("🔬 Start Analysis")
        self.analyze_button.clicked.connect(self.start_analysis)
        self.analyze_button.setStyleSheet("QPushButton { font-size: 14px; padding: 12px; }")
        
        self.stop_button = QPushButton("⏹️ Stop Analysis")
        self.stop_button.clicked.connect(self.stop_analysis)
        self.stop_button.setEnabled(False)
        
        # Info label about outputs
        output_info_label = QLabel("📁 Results saved to: [filename]_output folder")
        output_info_label.setStyleSheet("color: #666666; font-style: italic; font-size: 10px;")
        output_info_label.setWordWrap(True)
        
        analysis_layout.addWidget(self.analyze_button)
        analysis_layout.addWidget(self.stop_button)
        analysis_layout.addWidget(output_info_label)
        
        # Add all groups to main layout
        layout.addWidget(image_group)
        layout.addWidget(ocr_group)
        layout.addWidget(pinhole_group)
        layout.addWidget(processing_group)
        layout.addWidget(analysis_group)
        layout.addStretch()  # Push everything to top
        
        return panel
    
    def create_results_panel(self):
        """Create the right results panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Create tab widget for different result views
        self.results_tabs = QTabWidget()
        
        # Log tab
        self.log_tab = QTextEdit()
        self.log_tab.setReadOnly(True)
        self.log_tab.setFont(QFont("Consolas", 9))
        self.results_tabs.addTab(self.log_tab, "Analysis Log")
        
        # Results table tab
        self.results_table_tab = QTableWidget()
        self.results_tabs.addTab(self.results_table_tab, "Results Summary")
        
        # Image preview tab
        self.image_preview_tab = QScrollArea()
        self.image_preview_label = QLabel()
        self.image_preview_label.setAlignment(Qt.AlignCenter)
        self.image_preview_label.setText("No image loaded")
        self.image_preview_tab.setWidget(self.image_preview_label)
        self.results_tabs.addTab(self.image_preview_tab, "Image Preview")
        
        layout.addWidget(self.results_tabs)
        
        # Export controls
        export_group = QGroupBox("Export Results")
        export_layout = QHBoxLayout(export_group)
        
        self.export_csv_button = QPushButton("Export CSV")
        self.export_csv_button.clicked.connect(self.export_csv)
        self.export_csv_button.setEnabled(False)
        
        self.export_json_button = QPushButton("Export JSON")
        self.export_json_button.clicked.connect(self.export_json)
        self.export_json_button.setEnabled(False)
        
        self.open_output_button = QPushButton("Open Output Folder")
        self.open_output_button.clicked.connect(self.open_output_folder)
        
        export_layout.addWidget(self.export_csv_button)
        export_layout.addWidget(self.export_json_button)
        export_layout.addWidget(self.open_output_button)
        
        layout.addWidget(export_group)
        
        return panel
    
    def setup_connections(self):
        """Setup signal connections."""
        self.image_path_edit.textChanged.connect(self.on_image_path_changed)
    
    def browse_image(self):
        """Open file dialog to browse for image."""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self, 
            "Select SEM Image", 
            "", 
            "Image Files (*.tiff *.tif *.png *.jpg *.jpeg *.bmp);;All Files (*)"
        )
        
        if file_path:
            self.image_path_edit.setText(file_path)
    
    def on_image_path_changed(self, path):
        """Handle image path change."""
        self.current_image_path = path
        if path and os.path.exists(path):
            self.load_image_preview(path)
            self.log_message(f"Image loaded: {os.path.basename(path)}")
            
            # Enable analysis and pinhole detection buttons
            self.analyze_button.setEnabled(True)
            self.detect_pinholes_button.setEnabled(True)
            
            # Automatically detect scale if auto-OCR is enabled
            if self.auto_ocr_check.isChecked():
                self.auto_detect_scale_from_footer()
        else:
            self.image_preview_label.setText("No image loaded")
            self.analyze_button.setEnabled(False)
            self.detect_pinholes_button.setEnabled(False)
    
    def on_auto_ocr_toggled(self, checked):
        """Handle auto-OCR checkbox toggle."""
        self.frame_width_spinbox.setEnabled(not checked)
        if checked:
            self.scale_status_label.setText("Status: Auto-detection enabled")
            self.scale_status_label.setStyleSheet("color: #4CAF50; font-style: italic;")
        else:
            self.scale_status_label.setText("Status: Manual input mode")
            self.scale_status_label.setStyleSheet("color: #FF9800; font-style: italic;")
    
    def browse_multiple_images(self):
        """Browse and select multiple SEM images."""
        file_dialog = QFileDialog()
        file_paths, _ = file_dialog.getOpenFileNames(
            self,
            "Select Multiple SEM Images",
            "",
            "Image Files (*.tiff *.tif *.png *.jpg *.jpeg *.bmp);;All Files (*)"
        )
        
        if file_paths:
            # Add new files to the list (avoid duplicates)
            for file_path in file_paths:
                if file_path not in self.image_list:
                    self.image_list.append(file_path)
            
            self.update_image_list_display()
            self.process_all_button.setEnabled(len(self.image_list) > 0)
            self.log_message(f"📁 Added {len(file_paths)} images to queue (Total: {len(self.image_list)})")
    
    def clear_image_list(self):
        """Clear the image list."""
        self.image_list.clear()
        self.update_image_list_display()
        self.process_all_button.setEnabled(False)
        self.log_message("🗑️ Cleared image queue")
    
    def update_image_list_display(self):
        """Update the image list widget display."""
        self.image_list_widget.clear()
        for i, image_path in enumerate(self.image_list):
            filename = os.path.basename(image_path)
            self.image_list_widget.addItem(f"{i+1}. {filename}")
    
    def on_image_list_selection(self, item):
        """Handle selection from image list."""
        if item:
            # Extract index from item text
            index = int(item.text().split('.')[0]) - 1
            if 0 <= index < len(self.image_list):
                selected_path = self.image_list[index]
                self.image_path_edit.setText(selected_path)
                self.log_message(f"📂 Selected: {os.path.basename(selected_path)}")
    
    def process_all_images(self):
        """Process all images in the queue."""
        if not self.image_list:
            QMessageBox.warning(self, "Warning", "No images in queue to process.")
            return
        
        automate_enabled = self.automate_check.isChecked()
        
        if automate_enabled:
            # Automated processing
            reply = QMessageBox.question(
                self,
                "Automated Batch Processing",
                f"🤖 Ready to automatically process {len(self.image_list)} images\n\n"
                f"This will:\n"
                f"• Detect scale for each image\n"
                f"• Detect and save pinholes\n"
                f"• Analyze grain sizes\n"
                f"• Save all results automatically\n\n"
                f"Continue with automated processing?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
        else:
            # Manual processing
            reply = QMessageBox.question(
                self,
                "Manual Batch Processing",
                f"📋 Ready to process {len(self.image_list)} images with manual confirmation\n\n"
                f"You will be prompted for each step:\n"
                f"• Scale detection confirmation\n"
                f"• Pinhole detection review\n"
                f"• Grain analysis confirmation\n\n"
                f"Continue with manual processing?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
        
        if reply == QMessageBox.Yes:
            self.start_batch_processing()
    
    def start_batch_processing(self):
        """Start processing all images in the queue."""
        if not self.image_list:
            return
        
        self.current_batch_index = 0
        self.process_all_button.setEnabled(False)
        self.analyze_button.setEnabled(False)
        
        self.log_message(f"🚀 Starting batch processing of {len(self.image_list)} images...")
        self.process_next_image()
    
    def process_next_image(self):
        """Process the next image in the queue."""
        if self.current_batch_index >= len(self.image_list):
            # All images processed
            self.finish_batch_processing()
            return
        
        current_image = self.image_list[self.current_batch_index]
        self.current_image_path = current_image
        self.image_path_edit.setText(current_image)
        
        progress = ((self.current_batch_index + 1) / len(self.image_list)) * 100
        self.progress_bar.setValue(int(progress))
        
        self.log_message(f"📸 Processing image {self.current_batch_index + 1}/{len(self.image_list)}: {os.path.basename(current_image)}")
        
        if self.automate_check.isChecked():
            # Automated processing
            self.process_image_automated()
        else:
            # Manual processing - load image and let user proceed manually
            self.load_image_preview(current_image)
            QMessageBox.information(
                self,
                "Manual Processing",
                f"📸 Loaded image {self.current_batch_index + 1}/{len(self.image_list)}\n\n"
                f"{os.path.basename(current_image)}\n\n"
                f"Please process this image manually, then click 'Next Image' to continue."
            )
    
    def process_image_automated(self):
        """Process current image automatically (detect scale, pinholes, and grains)."""
        try:
            # Step 1: Auto-detect scale
            self.log_message("🔍 Step 1: Auto-detecting scale...")
            self.auto_detect_scale_from_footer()
            
            # Check if we have a valid frame width
            if self.frame_width_spinbox.value() <= 0:
                self.log_message("❌ Could not auto-detect scale, skipping image")
                self.current_batch_index += 1
                QTimer.singleShot(1000, self.process_next_image)
                return
            
            # Step 2: Auto-detect pinholes
            self.log_message("🕳️ Step 2: Auto-detecting pinholes...")
            self.detect_pinholes_automated()
            
            # Step 3: Auto-analyze grains (after a short delay)
            QTimer.singleShot(2000, self.analyze_grains_automated)
            
        except Exception as e:
            self.log_message(f"❌ Error in automated processing: {str(e)}")
            self.current_batch_index += 1
            QTimer.singleShot(1000, self.process_next_image)
    
    def detect_pinholes_automated(self):
        """Detect pinholes automatically without user interaction."""
        try:
            results = self.pinhole_detector.detect_pinholes(
                self.current_image_path,
                self.frame_width_spinbox.value()
            )
            
            if results['success']:
                count = results['count']
                self.pinhole_results = results
                
                # Save pinhole data automatically
                base_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
                output_dir = os.path.join(os.path.dirname(self.current_image_path), f"{base_name}_output")
                os.makedirs(output_dir, exist_ok=True)
                
                pinhole_csv_path = os.path.join(output_dir, f"{base_name}_pinholes.csv")
                
                success, message = self.pinhole_detector.save_pinhole_csv(
                    results['pinholes'],
                    pinhole_csv_path,
                    None  # Use detected count as-is
                )
                
                if success:
                    # Save preview image
                    if results['preview_image'] is not None:
                        preview_path = os.path.join(output_dir, f"{base_name}_pinhole_preview.png")
                        cv2.imwrite(preview_path, results['preview_image'])
                    
                    self.log_message(f"✅ Pinholes detected and saved: {count} pinholes")
                else:
                    self.log_message(f"❌ Error saving pinholes: {message}")
            else:
                self.log_message(f"⚠️ Pinhole detection failed: {results['message']}")
                
        except Exception as e:
            self.log_message(f"❌ Error in automated pinhole detection: {str(e)}")
    
    def analyze_grains_automated(self):
        """Analyze grains automatically without user interaction."""
        try:
            self.log_message("🔬 Step 3: Auto-analyzing grains...")
            
            # Prepare analysis parameters
            analysis_params = {
                'image_path': self.current_image_path,
                'frame_width_um': self.frame_width_spinbox.value(),
                'min_area_px': self.min_area_spinbox.value(),
                'apply_feret_cap': self.apply_cap_check.isChecked(),
                'feret_cap_um': self.feret_cap_spinbox.value(),
                'save_overlays': self.save_overlays_check.isChecked(),
                'annotate_measurements': self.annotate_check.isChecked(),
                'variants': self.config_manager.variants.copy()
            }
            
            # Create and start analysis worker thread
            self.analysis_worker = AnalysisWorker(analysis_params)
            self.analysis_worker.progress_updated.connect(self.update_progress)
            self.analysis_worker.log_message.connect(self.log_message)
            self.analysis_worker.analysis_completed.connect(self.on_automated_analysis_completed)
            self.analysis_worker.error_occurred.connect(self.on_automated_analysis_error)
            
            self.analysis_worker.start()
            
        except Exception as e:
            self.log_message(f"❌ Error starting automated grain analysis: {str(e)}")
            self.current_batch_index += 1
            QTimer.singleShot(1000, self.process_next_image)
    
    def on_automated_analysis_completed(self, results):
        """Handle automated analysis completion."""
        self.log_message("✅ Grain analysis completed automatically")
        
        # Move to next image
        self.current_batch_index += 1
        QTimer.singleShot(1000, self.process_next_image)
    
    def on_automated_analysis_error(self, error_message):
        """Handle automated analysis error."""
        self.log_message(f"❌ Automated grain analysis failed: {error_message}")
        
        # Move to next image
        self.current_batch_index += 1
        QTimer.singleShot(1000, self.process_next_image)
    
    def finish_batch_processing(self):
        """Finish batch processing and reset UI."""
        self.process_all_button.setEnabled(True)
        self.analyze_button.setEnabled(True)
        self.progress_bar.setValue(0)
        
        self.log_message(f"🎉 Batch processing completed! Processed {len(self.image_list)} images.")
        
        QMessageBox.information(
            self,
            "Batch Processing Complete",
            f"🎉 Successfully processed {len(self.image_list)} images!\n\n"
            f"📁 Results saved to individual [filename]_output folders\n"
            f"📊 Check logs for detailed processing information"
        )
    
    def load_image_preview(self, image_path):
        """Load and display image preview."""
        try:
            # Load image for preview
            gray, H, W = load_and_convert_to_grayscale(image_path)
            
            # Convert to QPixmap for display
            # Simple preview - just show the original image
            import cv2
            preview_img = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
            
            # Resize for preview if too large
            max_preview_size = 800
            if max(H, W) > max_preview_size:
                scale = max_preview_size / max(H, W)
                new_size = (int(W * scale), int(H * scale))
                preview_img = cv2.resize(preview_img, new_size)
            
            # Convert to RGB for Qt
            preview_rgb = cv2.cvtColor(preview_img, cv2.COLOR_GRAY2RGB)
            
            # Create QPixmap
            h, w, ch = preview_rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(preview_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            qt_pixmap = QPixmap.fromImage(qt_image)
            
            self.image_preview_label.setPixmap(qt_pixmap)
            
        except Exception as e:
            self.log_message(f"Error loading image preview: {str(e)}")
            self.image_preview_label.setText("Error loading image")
    
    def auto_detect_scale_from_footer(self):
        """Automatically detect scale from SEM footer using exact OCR."""
        if not self.current_image_path or not os.path.exists(self.current_image_path):
            return
        
        try:
            self.log_message("🔍 Auto-detecting scale from SEM footer...")
            
            # Use exact footer OCR
            results = self.exact_ocr.analyze_sem_footer_exact(self.current_image_path)
            
            if 'error' in results:
                self.log_message(f"❌ OCR Error: {results['error']}")
                self.ask_user_for_manual_scale()
                return
            
            metadata = results.get('metadata', {})
            
            if 'fw_um' in metadata:
                frame_width = metadata['fw_um']
                self.frame_width_spinbox.setValue(frame_width)
                self.log_message(f"✅ Auto-detected Frame Width: {frame_width} μm")
                
                # Update status label
                self.scale_status_label.setText(f"Status: ✅ Detected {frame_width} μm")
                self.scale_status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
                
                # Show additional detected info
                if 'magnification' in metadata:
                    self.log_message(f"🔍 Magnification: {metadata['magnification']:,}x")
                if 'voltage_kv' in metadata:
                    self.log_message(f"⚡ Voltage: {metadata['voltage_kv']} kV")
                if 'working_distance_mm' in metadata:
                    self.log_message(f"📏 Working Distance: {metadata['working_distance_mm']} mm")
                
                # Show pixel size calculation
                if results.get('um_per_pixel'):
                    self.log_message(f"📐 Calculated pixel size: {results['um_per_pixel']:.8f} μm/pixel")
                
                # Show success message
                QMessageBox.information(
                    self, 
                    "Auto-Detection Success", 
                    f"Successfully detected Frame Width: {frame_width} μm\n\n"
                    f"Please verify this value is correct before proceeding with analysis."
                )
            else:
                self.log_message("❌ Could not detect Frame Width from footer")
                self.scale_status_label.setText("Status: ❌ Detection failed")
                self.scale_status_label.setStyleSheet("color: #F44336; font-weight: bold;")
                self.ask_user_for_manual_scale()
                
        except Exception as e:
            self.log_message(f"❌ Auto-detection error: {str(e)}")
            self.scale_status_label.setText("Status: ❌ Error occurred")
            self.scale_status_label.setStyleSheet("color: #F44336; font-weight: bold;")
            self.ask_user_for_manual_scale()
    
    def ask_user_for_manual_scale(self):
        """Ask user to provide manual scale information."""
        reply = QMessageBox.question(
            self, 
            "Manual Scale Input Required", 
            "Automatic scale detection failed.\n\n"
            "Would you like to:\n"
            "• Click 'Yes' to enter Frame Width manually\n"
            "• Click 'No' to try OCR detection again",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        
        if reply == QMessageBox.Yes:
            # Enable manual input
            self.auto_ocr_check.setChecked(False)
            self.frame_width_spinbox.setEnabled(True)
            self.frame_width_spinbox.setFocus()
            self.log_message("📝 Please enter Frame Width manually")
        else:
            # Try OCR again
            self.detect_scale()
    
    def detect_scale(self):
        """Detect scale information from the image using exact footer OCR."""
        if not self.current_image_path or not os.path.exists(self.current_image_path):
            QMessageBox.warning(self, "Warning", "Please select a valid image file first.")
            return
        
        try:
            self.log_message("🔍 Detecting scale information using exact footer OCR...")
            
            # Use exact footer OCR (same as auto-detection)
            results = self.exact_ocr.analyze_sem_footer_exact(self.current_image_path)
            
            if 'error' in results:
                self.log_message(f"❌ OCR Error: {results['error']}")
                QMessageBox.critical(self, "OCR Error", f"Error during scale detection: {results['error']}")
                return
            
            metadata = results.get('metadata', {})
            
            if 'fw_um' in metadata:
                frame_width = metadata['fw_um']
                self.frame_width_spinbox.setValue(frame_width)
                self.log_message(f"✅ Detected Frame Width: {frame_width} μm")
                
                # Show additional detected info
                info_lines = [f"Frame Width: {frame_width} μm"]
                if 'magnification' in metadata:
                    mag = metadata['magnification']
                    self.log_message(f"🔍 Magnification: {mag:,}x")
                    info_lines.append(f"Magnification: {mag:,}x")
                if 'voltage_kv' in metadata:
                    voltage = metadata['voltage_kv']
                    self.log_message(f"⚡ Voltage: {voltage} kV")
                    info_lines.append(f"Voltage: {voltage} kV")
                if 'working_distance_mm' in metadata:
                    wd = metadata['working_distance_mm']
                    self.log_message(f"📏 Working Distance: {wd} mm")
                    info_lines.append(f"Working Distance: {wd} mm")
                
                # Show pixel size calculation
                if results.get('um_per_pixel'):
                    pixel_size = results['um_per_pixel']
                    self.log_message(f"📐 Calculated pixel size: {pixel_size:.8f} μm/pixel")
                    info_lines.append(f"Pixel size: {pixel_size:.6f} μm/pixel")
                
                # Show extracted text for debugging
                if results.get('ocr_text'):
                    self.log_message(f"📝 OCR Text: {results['ocr_text'][:100]}...")
                
                # Show success dialog
                QMessageBox.information(
                    self, 
                    "Scale Detection Success", 
                    "Successfully detected SEM metadata:\n\n" + "\n".join(info_lines) +
                    "\n\nPlease verify these values are correct."
                )
                    
            else:
                self.log_message("❌ Could not detect Frame Width from image footer")
                QMessageBox.information(
                    self, 
                    "Detection Failed", 
                    "Could not detect Frame Width automatically.\n\n"
                    "Please enter the value manually or check if the image has a readable SEM footer."
                )
                
        except Exception as e:
            error_msg = f"Error during scale detection: {str(e)}"
            self.log_message(error_msg)
            QMessageBox.critical(self, "OCR Error", error_msg)
    
    def detect_pinholes(self):
        """Detect pinholes in the current image."""
        if not self.current_image_path or not os.path.exists(self.current_image_path):
            QMessageBox.warning(self, "Warning", "Please select a valid image file first.")
            return
        
        if self.frame_width_spinbox.value() <= 0:
            QMessageBox.warning(self, "Warning", "Please set a valid Frame Width first.")
            return
        
        try:
            self.log_message("🕳️ Detecting pinholes...")
            self.detect_pinholes_button.setEnabled(False)
            
            # Detect pinholes
            results = self.pinhole_detector.detect_pinholes(
                self.current_image_path,
                self.frame_width_spinbox.value()
            )
            
            if results['success']:
                count = results['count']
                self.pinhole_results = results
                self.pinhole_count_spinbox.setValue(count)
                self.pinhole_count_spinbox.setEnabled(True)
                self.show_pinhole_preview_button.setEnabled(True)
                self.confirm_pinholes_button.setEnabled(True)
                
                self.pinhole_status_label.setText(f"Status: ✅ Detected {count} pinholes")
                self.pinhole_status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
                
                self.log_message(f"✅ Detected {count} pinholes")
                
                # Show confirmation dialog
                reply = QMessageBox.question(
                    self,
                    "Pinholes Detected",
                    f"🕳️ Detected {count} pinholes\n\n"
                    f"Would you like to:\n"
                    f"• View the preview image?\n"
                    f"• Modify the count if needed?\n"
                    f"• Confirm and save the results?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes
                )
                
                if reply == QMessageBox.Yes:
                    self.show_pinhole_preview()
                    
            else:
                self.pinhole_status_label.setText(f"Status: ❌ Detection failed")
                self.pinhole_status_label.setStyleSheet("color: #F44336; font-weight: bold;")
                self.log_message(f"❌ Pinhole detection failed: {results['message']}")
                QMessageBox.warning(self, "Detection Failed", f"Pinhole detection failed:\n{results['message']}")
                
        except Exception as e:
            error_msg = f"Error during pinhole detection: {str(e)}"
            self.log_message(error_msg)
            QMessageBox.critical(self, "Pinhole Detection Error", error_msg)
        finally:
            self.detect_pinholes_button.setEnabled(True)
    
    def show_pinhole_preview(self):
        """Show preview of detected pinholes."""
        if not self.pinhole_results or not self.pinhole_results['success']:
            QMessageBox.warning(self, "Warning", "No pinhole detection results available.")
            return
        
        try:
            # Create preview dialog
            from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel
            
            dialog = QDialog(self)
            dialog.setWindowTitle("Pinhole Detection Preview")
            dialog.setMinimumSize(800, 600)
            
            layout = QVBoxLayout(dialog)
            
            # Info label
            count = self.pinhole_results['count']
            info_label = QLabel(f"🕳️ Detected {count} pinholes (green circles and labels)")
            info_label.setStyleSheet("font-weight: bold; font-size: 14px; margin: 10px;")
            layout.addWidget(info_label)
            
            # Preview image
            preview_img = self.pinhole_results['preview_image']
            if preview_img is not None:
                # Convert BGR to RGB for Qt
                preview_rgb = cv2.cvtColor(preview_img, cv2.COLOR_BGR2RGB)
                
                # Resize if too large
                h, w = preview_rgb.shape[:2]
                max_size = 700
                if max(h, w) > max_size:
                    scale = max_size / max(h, w)
                    new_w, new_h = int(w * scale), int(h * scale)
                    preview_rgb = cv2.resize(preview_rgb, (new_w, new_h))
                
                # Create QPixmap
                h, w, ch = preview_rgb.shape
                bytes_per_line = ch * w
                qt_image = QImage(preview_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                qt_pixmap = QPixmap.fromImage(qt_image)
                
                preview_label = QLabel()
                preview_label.setPixmap(qt_pixmap)
                preview_label.setAlignment(Qt.AlignCenter)
                layout.addWidget(preview_label)
            
            # Buttons
            button_layout = QHBoxLayout()
            close_button = QPushButton("Close")
            close_button.clicked.connect(dialog.accept)
            button_layout.addWidget(close_button)
            layout.addLayout(button_layout)
            
            dialog.exec_()
            
        except Exception as e:
            self.log_message(f"Error showing pinhole preview: {str(e)}")
            QMessageBox.critical(self, "Preview Error", f"Error showing preview: {str(e)}")
    
    def confirm_pinhole_count(self):
        """Confirm and save the pinhole count."""
        if not self.pinhole_results:
            QMessageBox.warning(self, "Warning", "No pinhole detection results available.")
            return
        
        try:
            detected_count = self.pinhole_results['count']
            user_count = self.pinhole_count_spinbox.value()
            
            # Determine if user modified the count
            count_modified = (user_count != detected_count)
            final_count = user_count
            
            # Save pinhole data
            base_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
            output_dir = os.path.join(os.path.dirname(self.current_image_path), f"{base_name}_output")
            os.makedirs(output_dir, exist_ok=True)
            
            pinhole_csv_path = os.path.join(output_dir, f"{base_name}_pinholes.csv")
            
            success, message = self.pinhole_detector.save_pinhole_csv(
                self.pinhole_results['pinholes'],
                pinhole_csv_path,
                final_count if count_modified else None
            )
            
            if success:
                # Save preview image
                if self.pinhole_results['preview_image'] is not None:
                    preview_path = os.path.join(output_dir, f"{base_name}_pinhole_preview.png")
                    cv2.imwrite(preview_path, self.pinhole_results['preview_image'])
                    self.log_message(f"📷 Pinhole preview saved: {base_name}_pinhole_preview.png")
                
                self.pinhole_status_label.setText(f"Status: ✅ Saved {final_count} pinholes")
                self.pinhole_status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
                
                status_msg = f"✅ Pinhole data saved successfully!"
                if count_modified:
                    status_msg += f"\n🔧 Count modified: {detected_count} → {final_count}"
                else:
                    status_msg += f"\n📊 Final count: {final_count}"
                
                self.log_message(f"💾 {message}")
                
                QMessageBox.information(
                    self,
                    "Pinholes Saved",
                    status_msg + f"\n📁 Saved to: {base_name}_output/"
                )
                
            else:
                self.log_message(f"❌ {message}")
                QMessageBox.critical(self, "Save Error", message)
                
        except Exception as e:
            error_msg = f"Error saving pinhole data: {str(e)}"
            self.log_message(error_msg)
            QMessageBox.critical(self, "Save Error", error_msg)
    
    def start_analysis(self):
        """Start the grain analysis process."""
        if not self.current_image_path or not os.path.exists(self.current_image_path):
            QMessageBox.warning(self, "Warning", "Please select a valid image file first.")
            return
        
        # Prepare analysis parameters (using all variants but hidden from UI)
        analysis_params = {
            'image_path': self.current_image_path,
            'frame_width_um': self.frame_width_spinbox.value(),
            'min_area_px': self.min_area_spinbox.value(),
            'apply_feret_cap': self.apply_cap_check.isChecked(),
            'feret_cap_um': self.feret_cap_spinbox.value(),
            'save_overlays': self.save_overlays_check.isChecked(),
            'annotate_measurements': self.annotate_check.isChecked(),
            'variants': self.config_manager.variants.copy()  # All variants run in background
        }
        
        # Create and start analysis worker thread
        self.analysis_worker = AnalysisWorker(analysis_params)
        self.analysis_worker.progress_updated.connect(self.update_progress)
        self.analysis_worker.log_message.connect(self.log_message)
        self.analysis_worker.analysis_completed.connect(self.on_analysis_completed)
        self.analysis_worker.error_occurred.connect(self.on_analysis_error)
        
        # Update UI state
        self.analyze_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        self.log_message("Starting grain analysis...")
        self.analysis_worker.start()
    
    def stop_analysis(self):
        """Stop the current analysis."""
        if hasattr(self, 'analysis_worker') and self.analysis_worker.isRunning():
            self.analysis_worker.stop()
            self.log_message("Analysis stopped by user.")
        
        self.reset_ui_after_analysis()
    
    def update_progress(self, value, message=""):
        """Update progress bar and status."""
        self.progress_bar.setValue(value)
        if message:
            self.statusBar().showMessage(message)
    
    def log_message(self, message):
        """Add message to log."""
        current_time = QTime.currentTime().toString("hh:mm:ss")
        self.log_tab.append(f"[{current_time}] {message}")
        self.log_tab.ensureCursorVisible()
    
    def auto_detect_scale_from_footer(self):
        """Auto-detect scale from image footer."""
        if not self.current_image_path:
            return
        
        try:
            from ..core.exact_footer_ocr import ExactFooterOCR
            ocr = ExactFooterOCR()
            frame_width = ocr.extract_frame_width(self.current_image_path)
            
            if frame_width:
                self.frame_width_spinbox.setValue(frame_width)
                self.scale_status_label.setText(f"✅ Detected: {frame_width} μm")
                self.scale_status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
                self.log_message(f"🎯 Auto-detected scale: {frame_width} μm")
            else:
                self.scale_status_label.setText("❌ Auto-detection failed")
                self.scale_status_label.setStyleSheet("color: #f44336; font-weight: bold;")
                self.log_message("⚠️ Could not auto-detect scale from footer")
        except Exception as e:
            self.scale_status_label.setText("❌ Detection error")
            self.scale_status_label.setStyleSheet("color: #f44336; font-weight: bold;")
            self.log_message(f"❌ Scale detection error: {str(e)}")

    def load_image_preview(self, image_path):
        """Load image preview without full processing."""
        try:
            self.current_image_path = image_path
            # Update UI with current image
            self.image_path_edit.setText(image_path)
            
            # If auto-OCR is enabled, try to detect scale
            if self.auto_ocr_check.isChecked():
                self.auto_detect_scale_from_footer()
                
        except Exception as e:
            self.log_message(f"❌ Error loading image preview: {str(e)}")

    def detect_pinholes(self):
        """Detect pinholes in the current image."""
        if not self.current_image_path:
            QMessageBox.warning(self, "Warning", "Please select an image first.")
            return
        
        if not self.frame_width_spinbox.value():
            QMessageBox.warning(self, "Warning", "Please set the frame width first.")
            return
        
        try:
            self.log_message("🕳️ Starting pinhole detection...")
            
            # Show progress
            progress = QProgressDialog("Detecting pinholes...", None, 0, 0, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
            QApplication.processEvents()
            
            # Detect pinholes
            results = self.pinhole_detector.detect_pinholes(
                self.current_image_path,
                self.frame_width_spinbox.value()
            )
            
            progress.close()
            
            if results['success']:
                self.pinhole_results = results
                self.show_pinhole_preview_dialog(results)
            else:
                QMessageBox.warning(self, "Detection Failed", 
                                  f"Pinhole detection failed:\n{results['message']}")
                
        except Exception as e:
            if 'progress' in locals():
                progress.close()
            QMessageBox.critical(self, "Error", f"Error during pinhole detection:\n{str(e)}")
            self.log_message(f"❌ Pinhole detection error: {str(e)}")
    
    def show_pinhole_preview_dialog(self, results):
        """Show pinhole detection preview dialog with results parameter."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Pinhole Detection Results")
        dialog.setModal(True)
        dialog.resize(800, 600)
        
        layout = QVBoxLayout(dialog)
        
        # Info section
        info_text = f"""
Detected Pinholes: {results['count']}
Confidence Threshold: {results.get('confidence_threshold', 0.5):.2f}
Processing Time: {results.get('processing_time', 0):.1f}s
        """.strip()
        
        info_label = QLabel(info_text)
        info_label.setStyleSheet("font-family: monospace; background: #f0f0f0; padding: 10px; border-radius: 5px;")
        layout.addWidget(info_label)
        
        # Preview image
        if results.get('preview_image') is not None:
            preview_label = QLabel()
            preview_label.setAlignment(Qt.AlignCenter)
            
            # Convert OpenCV image to QPixmap
            height, width, channel = results['preview_image'].shape
            bytes_per_line = 3 * width
            q_image = QImage(results['preview_image'].data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(q_image)
            
            # Scale to fit dialog
            scaled_pixmap = pixmap.scaled(750, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            preview_label.setPixmap(scaled_pixmap)
            
            scroll_area = QScrollArea()
            scroll_area.setWidget(preview_label)
            scroll_area.setWidgetResizable(True)
            layout.addWidget(scroll_area)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        save_button = QPushButton("Save Results")
        save_button.clicked.connect(lambda: self.save_pinhole_results(dialog))
        
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(dialog.reject)
        
        button_layout.addWidget(save_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)
        
        dialog.exec_()
    
    def save_pinhole_results(self, dialog):
        """Save pinhole detection results."""
        if not self.pinhole_results:
            return
        
        try:
            # Create output directory
            base_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
            output_dir = os.path.join(os.path.dirname(self.current_image_path), f"{base_name}_output")
            os.makedirs(output_dir, exist_ok=True)
            
            # Save CSV
            csv_path = os.path.join(output_dir, f"{base_name}_pinholes.csv")
            success, message = self.pinhole_detector.save_pinhole_csv(
                self.pinhole_results['pinholes'],
                csv_path,
                self.pinhole_results['count']
            )
            
            if success:
                # Save preview image
                if self.pinhole_results.get('preview_image') is not None:
                    preview_path = os.path.join(output_dir, f"{base_name}_pinhole_preview.png")
                    cv2.imwrite(preview_path, self.pinhole_results['preview_image'])
                
                self.log_message(f"✅ Pinhole results saved to: {output_dir}")
                QMessageBox.information(dialog, "Success", f"Results saved successfully!\n\nLocation: {output_dir}")
                dialog.accept()
            else:
                QMessageBox.warning(dialog, "Save Error", f"Failed to save results:\n{message}")
                
        except Exception as e:
            QMessageBox.critical(dialog, "Error", f"Error saving results:\n{str(e)}")
            self.log_message(f"❌ Error saving pinhole results: {str(e)}")
    
    def confirm_pinhole_count(self):
        """Confirm and save pinhole count from manual detection."""
        if not self.pinhole_results:
            QMessageBox.warning(self, "Warning", "No pinhole detection results available.")
            return
        
        # Show save dialog immediately
        self.save_pinhole_results(self)
    

    
    def on_analysis_completed(self, results):
        """Handle completed analysis."""
        self.results = results
        self.log_message("Analysis completed successfully!")
        
        # Update results table
        self.update_results_table(results)
        
        # Enable export buttons
        self.export_csv_button.setEnabled(True)
        self.export_json_button.setEnabled(True)
        
        self.reset_ui_after_analysis()
        
        # Get the specific output folder name
        base_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
        output_folder = f"{base_name}_output"
        
        # Show completion message
        QMessageBox.information(
            self, 
            "Analysis Complete", 
            f"✅ Grain analysis completed successfully!\n\n"
            f"📊 Processed {len(results.get('variant_results', []))} analysis variants\n"
            f"📁 Results saved to: {output_folder}/"
        )
    
    def on_analysis_error(self, error_message):
        """Handle analysis error."""
        self.log_message(f"Analysis error: {error_message}")
        self.reset_ui_after_analysis()
        QMessageBox.critical(self, "Analysis Error", f"Analysis failed:\n{error_message}")
    
    def reset_ui_after_analysis(self):
        """Reset UI state after analysis completion or error."""
        self.analyze_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.statusBar().showMessage("Ready")
    
    def update_results_table(self, results):
        """Update the results table with analysis results."""
        variant_results = results.get('variant_results', [])
        
        if not variant_results:
            return
        
        # Setup table
        self.results_table_tab.setRowCount(len(variant_results))
        self.results_table_tab.setColumnCount(6)
        self.results_table_tab.setHorizontalHeaderLabels([
            "Variant", "Grains Used", "Mean Chord (µm)", 
            "Median Chord (µm)", "Mean Area (µm²)", "Median Area (µm²)"
        ])
        
        # Populate table
        for row, result in enumerate(variant_results):
            chord_stats = result.get('chord_statistics')
            area_stats = result.get('area_statistics')
            
            self.results_table_tab.setItem(row, 0, QTableWidgetItem(result.get('variant_name', '')))
            self.results_table_tab.setItem(row, 1, QTableWidgetItem(str(result.get('grains_used', 0))))
            
            if chord_stats:
                self.results_table_tab.setItem(row, 2, QTableWidgetItem(f"{chord_stats.mean:.3f}"))
                self.results_table_tab.setItem(row, 3, QTableWidgetItem(f"{chord_stats.median:.3f}"))
            
            if area_stats:
                self.results_table_tab.setItem(row, 4, QTableWidgetItem(f"{area_stats.mean:.3f}"))
                self.results_table_tab.setItem(row, 5, QTableWidgetItem(f"{area_stats.median:.3f}"))
        
        self.results_table_tab.resizeColumnsToContents()
    
    def export_csv(self):
        """Export results to CSV format."""
        if not self.results:
            return
        
        try:
            from core.results import ResultsExporter
            
            # Get base output path using new folder structure
            base_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
            output_dir = os.path.join(os.path.dirname(self.current_image_path), f"{base_name}_output")
            base_path = os.path.join(output_dir, base_name)
            
            # Create exporter and export
            exporter = ResultsExporter(base_path)
            exported_files = exporter.export_all_formats(
                self.results['variant_results'],
                self.results['combined_results'],
                self.results.get('processing_config', {})
            )
            
            self.log_message(f"Results exported to: {base_name}_output/")
            for format_name, file_path in exported_files.items():
                self.log_message(f"  {format_name}: {os.path.basename(file_path)}")
            
        except Exception as e:
            error_msg = f"Error exporting CSV: {str(e)}"
            self.log_message(error_msg)
            QMessageBox.critical(self, "Export Error", error_msg)
    
    def export_json(self):
        """Export results to JSON format."""
        if not self.results:
            return
        
        try:
            import json
            
            # Get output path using new folder structure
            base_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
            output_dir = os.path.join(os.path.dirname(self.current_image_path), f"{base_name}_output")
            os.makedirs(output_dir, exist_ok=True)
            
            json_path = os.path.join(output_dir, f"{base_name}_detailed_results.json")
            
            # Save JSON
            with open(json_path, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            
            self.log_message(f"JSON results exported to: {base_name}_output/")
            
        except Exception as e:
            error_msg = f"Error exporting JSON: {str(e)}"
            self.log_message(error_msg)
            QMessageBox.critical(self, "Export Error", error_msg)
    
    def open_output_folder(self):
        """Open the output folder in file explorer."""
        try:
            if self.current_image_path:
                base_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
                output_dir = os.path.join(os.path.dirname(self.current_image_path), f"{base_name}_output")
                if os.path.exists(output_dir):
                    os.startfile(output_dir)  # Windows
                else:
                    os.makedirs(output_dir, exist_ok=True)
                    os.startfile(output_dir)
            else:
                QMessageBox.information(self, "Info", "Please load an image first to determine output location.")
        except Exception as e:
            self.log_message(f"Error opening output folder: {str(e)}")
    
    def detect_device_info(self):
        """Detect and display device information for processing."""
        try:
            from core.sam_analysis import GrainAnalyzer
            
            # Create a temporary analyzer to get device info
            temp_analyzer = GrainAnalyzer()
            device_info = temp_analyzer.get_device_info()
            
            # Format device info for display
            device_type = device_info.get("device_type", "Unknown")
            device = device_info.get("device", "Unknown")
            
            if device_type == "GPU":
                gpu_name = device_info.get("gpu_name", "Unknown GPU")
                gpu_memory = device_info.get("gpu_memory", 0)
                device_text = f"🚀 GPU: {gpu_name} ({gpu_memory:.1f}GB)"
            else:
                cpu_cores = device_info.get("cpu_cores", 0)
                cpu_threads = device_info.get("cpu_threads", 0)
                device_text = f"🖥️ CPU: {cpu_cores} cores, {cpu_threads} threads"
            
            self.device_label.setText(device_text)
            self.log_message(f"Processing device detected: {device_text}")
            
        except Exception as e:
            self.device_label.setText("⚠️ Device detection failed")
            self.log_message(f"Device detection error: {str(e)}")


def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Grain Size Calculator")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("Rayleigh Solar Tech")
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Start event loop
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()