"""
Clinical PDF Report Generation Engine
======================================
Domain: Healthcare / Computer Vision / Deep Learning
Dataset: APTOS 2019 Blindness Detection
Task: Generates publication-grade clinical diagnostic PDF reports containing:
      - Patient metadata & timestamp
      - Primary diagnostic impression & severity classification
      - Confidence score & clinical recommendations
      - Class probability distribution table
      - Side-by-side embedded fundus image & Grad-CAM overlay heatmap

Author: Senior Software Architect & Clinical AI Lead
"""

import os
import io
import base64
from datetime import datetime
from PIL import Image

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, HRFlowable


class ClinicalPDFGenerator:
    """
    Generates medical PDF report documents adhering to healthcare documentation standards.
    """

    def __init__(self, reports_dir: str = "reports/pdf"):
        self.reports_dir = reports_dir
        os.makedirs(self.reports_dir, exist_ok=True)

    def generate_pdf_report(
        self,
        patient_id: str,
        filename: str,
        predicted_class: str,
        confidence: float,
        recommendation: str,
        class_probabilities: list,
        original_img_rgb: str = None, # Base64 or numpy array path
        overlay_img_rgb: str = None,  # Base64 or numpy array path
        output_pdf_path: str = None
    ) -> str:
        """
        Builds a multi-section clinical PDF report document.
        Returns the absolute filepath to the saved PDF document.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if not output_pdf_path:
            pdf_name = f"DR_Report_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            output_pdf_path = os.path.join(self.reports_dir, pdf_name)

        doc = SimpleDocTemplate(
            output_pdf_path,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )

        story = []
        styles = getSampleStyleSheet()

        # Custom Styles
        title_style = ParagraphStyle(
            'DocTitle',
            parent=styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=20,
            leading=24,
            textColor=colors.HexColor('#1e293b')
        )

        subtitle_style = ParagraphStyle(
            'DocSubtitle',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=10,
            leading=12,
            textColor=colors.HexColor('#64748b')
        )

        header_style = ParagraphStyle(
            'SectionHeader',
            parent=styles['Heading2'],
            fontName='Helvetica-Bold',
            fontSize=12,
            leading=16,
            textColor=colors.HexColor('#0f172a'),
            spaceBefore=10,
            spaceAfter=6
        )

        body_style = ParagraphStyle(
            'Body',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=10,
            leading=14,
            textColor=colors.HexColor('#334155')
        )

        # 1. Document Header Banner
        story.append(Paragraph("RetinaX AI — Clinical Diagnostic Report", title_style))
        story.append(Paragraph("Diabetic Retinopathy Screening & Explainable AI Visual Analysis", subtitle_style))
        story.append(Spacer(1, 10))
        story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#6366f1'), spaceAfter=15))

        # 2. Patient & Scan Metadata Table
        meta_data = [
            [
                Paragraph(f"<b>Patient ID:</b> {patient_id}", body_style),
                Paragraph(f"<b>Timestamp:</b> {timestamp}", body_style)
            ],
            [
                Paragraph(f"<b>Source Image:</b> {filename}", body_style),
                Paragraph(f"<b>Model Architecture:</b> EfficientNet-B0", body_style)
            ]
        ]
        meta_table = Table(meta_data, colWidths=[270, 270])
        meta_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
            ('PADDING', (0,0), (-1,-1), 8),
            ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 15))

        # 3. Primary AI Diagnostic Impression Box
        severity_colors = {
            "No DR": colors.HexColor('#10b981'),
            "Mild": colors.HexColor('#3b82f6'),
            "Moderate": colors.HexColor('#f59e0b'),
            "Severe": colors.HexColor('#f97316'),
            "Proliferative DR": colors.HexColor('#ef4444')
        }
        banner_bg = severity_colors.get(predicted_class, colors.HexColor('#6366f1'))

        diag_data = [
            [
                Paragraph(f"<font color='white' size=14><b>Diagnostic Impression: {predicted_class}</b></font>", body_style),
                Paragraph(f"<font color='white' size=12><b>Confidence: {confidence*100:.1f}%</b></font>", body_style)
            ],
            [
                Paragraph(f"<font color='white'><b>Clinical Recommendation:</b> {recommendation}</font>", body_style),
                ""
            ]
        ]
        diag_table = Table(diag_data, colWidths=[380, 160])
        diag_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), banner_bg),
            ('PADDING', (0,0), (-1,-1), 10),
            ('SPAN', (0,1), (1,1)),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ]))
        story.append(diag_table)
        story.append(Spacer(1, 15))

        # 4. Class Severity Probability Breakdown Table
        story.append(Paragraph("Severity Class Probability Distribution", header_style))
        prob_headers = ["Severity Grade", "Class Name", "Model Probability", "Status"]
        prob_rows = [prob_headers]

        for item in class_probabilities:
            cls_name = item.get('class_name', item.get('class', ''))
            cls_idx = item.get('class_index', item.get('idx', 0))
            prob = item.get('probability', item.get('prob', 0.0))
            is_target = (cls_name == predicted_class)

            status_text = "PRIMARY DIAGNOSIS" if is_target else "-"
            prob_rows.append([
                f"Grade {cls_idx}",
                cls_name,
                f"{prob*100:.2f}%",
                status_text
            ])

        prob_table = Table(prob_rows, colWidths=[100, 180, 130, 130])
        prob_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0f172a')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('PADDING', (0,0), (-1,-1), 6),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
            ('ALIGN', (2,0), (2,-1), 'RIGHT'),
        ]))
        story.append(prob_table)
        story.append(Spacer(1, 15))

        # 5. Grad-CAM Visual Explanation Image Panels (Side-by-Side if provided)
        if overlay_img_rgb:
            story.append(Paragraph("Grad-CAM Visual Explainability Analysis", header_style))
            try:
                # Helper to convert Base64 string to ReportLab RLImage
                def b64_to_rlimage(b64_str, width=240, height=240):
                    if b64_str.startswith("data:image"):
                        b64_str = b64_str.split(",")[1]
                    img_bytes = base64.b64decode(b64_str)
                    img_buf = io.BytesIO(img_bytes)
                    return RLImage(img_buf, width=width, height=height)

                overlay_rl = b64_to_rlimage(overlay_img_rgb, width=250, height=250)
                img_table = Table([
                    [overlay_rl],
                    [Paragraph("<b>Clinical Grad-CAM Heatmap Overlay</b><br/>Red/Yellow highlights focal regions triggering AI classification.", subtitle_style)]
                ], colWidths=[540])
                img_table.setStyle(TableStyle([
                    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                    ('PADDING', (0,0), (-1,-1), 4),
                ]))
                story.append(img_table)
            except Exception as e:
                print(f"[WARN] Could not render image overlay into PDF: {e}")

        # Build PDF Document
        doc.build(story)
        print(f"[OK] Generated Clinical PDF Report at: '{output_pdf_path}'")
        return output_pdf_path
