"""
PDF Report Generator
Generate professional PDF reports for evaluations
"""
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from io import BytesIO
from datetime import datetime


def generate_evaluation_pdf(evaluation_data: dict, interview_data: dict, candidate_data: dict):
    """
    Generate a professional PDF evaluation report.
    
    Returns:
        BytesIO: PDF file buffer
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    
    # Container for PDF elements
    elements = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#6366f1'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#1f2937'),
        spaceAfter=12,
        spaceBefore=20,
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=11,
        textColor=colors.HexColor('#374151'),
        spaceAfter=12,
        leading=16
    )
    
    # Header
    elements.append(Paragraph("⚡ InnovAIte Interview Guardian", title_style))
    elements.append(Paragraph("AI-Assisted Explainable Evaluation Report", styles['Normal']))
    elements.append(Spacer(1, 0.3*inch))
    
    # Candidate Info
    elements.append(Paragraph("Candidate Information", heading_style))
    candidate_table_data = [
        ['Name:', candidate_data.get('name', 'N/A')],
        ['Email:', candidate_data.get('email', 'N/A')],
        ['Interview:', interview_data.get('title', 'N/A')],
        ['Job Role:', interview_data.get('job_title', 'N/A')],
        ['Date:', datetime.fromisoformat(evaluation_data['created_at'].replace('Z', '+00:00')).strftime('%B %d, %Y')],
    ]
    candidate_table = Table(candidate_table_data, colWidths=[2*inch, 4*inch])
    candidate_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f3f4f6')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#1f2937')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e5e7eb')),
    ]))
    elements.append(candidate_table)
    elements.append(Spacer(1, 0.3*inch))
    
    # Overall Score
    elements.append(Paragraph("Overall Performance", heading_style))
    score = evaluation_data['overall_score']
    recommendation = evaluation_data['recommendation'].replace('_', ' ').upper()
    
    score_color = colors.HexColor('#10b981') if score >= 70 else colors.HexColor('#f59e0b') if score >= 50 else colors.HexColor('#ef4444')
    
    score_table_data = [
        ['Overall Score', f"{score}/100"],
        ['Recommendation', recommendation],
    ]
    score_table = Table(score_table_data, colWidths=[3*inch, 3*inch])
    score_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#6366f1')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 14),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e5e7eb')),
        ('BACKGROUND', (1, 1), (1, 1), score_color),
        ('TEXTCOLOR', (1, 1), (1, 1), colors.white),
    ]))
    elements.append(score_table)
    elements.append(Spacer(1, 0.2*inch))
    
    # AI Summary
    elements.append(Paragraph("AI Executive Summary", heading_style))
    elements.append(Paragraph(evaluation_data['summary'], body_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # Criterion Breakdown
    elements.append(Paragraph("Detailed Criterion Analysis", heading_style))
    
    criterion_data = [['Criterion', 'Score', 'Weight', 'Explanation']]
    for cr in evaluation_data['criterion_results']:
        criterion_data.append([
            cr['criterion'].replace('_', ' ').title(),
            f"{cr['score']}/10",
            f"{cr['weight']}x",
            cr['explanation'][:100] + '...' if len(cr['explanation']) > 100 else cr['explanation']
        ])
    
    criterion_table = Table(criterion_data, colWidths=[1.5*inch, 0.8*inch, 0.7*inch, 3*inch])
    criterion_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#6366f1')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e5e7eb')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    elements.append(criterion_table)
    elements.append(Spacer(1, 0.3*inch))
    
    # Strengths & Weaknesses
    elements.append(Paragraph("Key Strengths", heading_style))
    for strength in evaluation_data['strengths']:
        elements.append(Paragraph(f"✅ {strength}", body_style))
    
    elements.append(Spacer(1, 0.2*inch))
    elements.append(Paragraph("Areas for Improvement", heading_style))
    for weakness in evaluation_data['weaknesses']:
        elements.append(Paragraph(f"📍 {weakness}", body_style))
    
    # Footer
    elements.append(Spacer(1, 0.5*inch))
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.HexColor('#6b7280'),
        alignment=TA_CENTER
    )
    elements.append(Paragraph(
        f"Generated by InnovAIte Interview Guardian | {datetime.utcnow().strftime('%B %d, %Y %H:%M UTC')}",
        footer_style
    ))
    elements.append(Paragraph("© 2024 InnovAIte | AI-Powered Recruitment Platform", footer_style))
    
    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    
    return buffer
