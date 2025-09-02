"""
PDF report generator using ReportLab.
"""

import io
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from decimal import Decimal

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Table,
    TableStyle,
    Paragraph,
    Spacer,
    PageBreak,
    Image,
    KeepTogether,
    ListFlowable,
    ListItem
)
from reportlab.graphics.shapes import Drawing, Line, Rect, Circle
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.graphics.charts.spider import SpiderChart
from reportlab.graphics.widgets.markers import makeMarker

from codeant_agent.domain.dora import (
    ExecutiveReport,
    ExecutiveSummary,
    BusinessMetrics,
    KeyMetric,
    StrategicRecommendation,
    Language,
    TrendDirection,
    BusinessImpact,
    RiskLevel
)
import logging

logger = logging.getLogger(__name__)


class PDFReportGenerator:
    """Generator for PDF executive reports."""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        
        # Color scheme
        self.colors = {
            'primary': colors.HexColor('#2563eb'),      # Blue
            'secondary': colors.HexColor('#64748b'),    # Gray
            'success': colors.HexColor('#16a34a'),      # Green
            'warning': colors.HexColor('#f59e0b'),      # Amber
            'danger': colors.HexColor('#dc2626'),       # Red
            'info': colors.HexColor('#0891b2'),         # Cyan
            'background': colors.HexColor('#f8fafc'),   # Light gray
            'text': colors.HexColor('#1e293b'),         # Dark gray
            'border': colors.HexColor('#e2e8f0')        # Light border
        }
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            textColor=colors.HexColor('#1e293b'),
            spaceAfter=20,
            alignment=1  # Center
        ))
        
        # Subtitle style
        self.styles.add(ParagraphStyle(
            name='CustomSubtitle',
            parent=self.styles['Heading2'],
            fontSize=18,
            textColor=colors.HexColor('#2563eb'),
            spaceAfter=12,
            spaceBefore=12
        ))
        
        # Section header style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading3'],
            fontSize=14,
            textColor=colors.HexColor('#1e293b'),
            spaceAfter=8,
            spaceBefore=16
        ))
        
        # Executive summary style
        self.styles.add(ParagraphStyle(
            name='ExecutiveSummary',
            parent=self.styles['Normal'],
            fontSize=11,
            leading=16,
            textColor=colors.HexColor('#475569'),
            alignment=4  # Justify
        ))
        
        # Metric value style
        self.styles.add(ParagraphStyle(
            name='MetricValue',
            parent=self.styles['Normal'],
            fontSize=20,
            textColor=colors.HexColor('#2563eb'),
            alignment=1
        ))
        
        # Recommendation style
        self.styles.add(ParagraphStyle(
            name='Recommendation',
            parent=self.styles['Normal'],
            fontSize=10,
            leading=14,
            textColor=colors.HexColor('#334155'),
            leftIndent=20,
            bulletIndent=10
        ))
    
    async def generate_pdf(
        self,
        report: ExecutiveReport,
        organization_name: str,
        logo_path: Optional[str] = None
    ) -> bytes:
        """Generate PDF report."""
        try:
            # Create PDF in memory
            buffer = io.BytesIO()
            
            # Create document
            doc = SimpleDocTemplate(
                buffer,
                pagesize=letter,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72,
                title=f"Executive Report - {organization_name}",
                author="CodeAnt AI"
            )
            
            # Build story (content)
            story = []
            
            # Add cover page
            story.extend(self._create_cover_page(
                report,
                organization_name,
                logo_path
            ))
            story.append(PageBreak())
            
            # Add executive summary
            story.extend(self._create_executive_summary(
                report.executive_summary,
                report.language
            ))
            story.append(PageBreak())
            
            # Add key metrics dashboard
            story.extend(self._create_metrics_dashboard(
                report.business_metrics,
                report.language
            ))
            story.append(PageBreak())
            
            # Add DORA metrics section
            if report.business_metrics.dora_metrics:
                story.extend(self._create_dora_section(
                    report.business_metrics.dora_metrics,
                    report.language
                ))
                story.append(PageBreak())
            
            # Add financial impact section
            story.extend(self._create_financial_section(
                report.business_metrics,
                report.language
            ))
            story.append(PageBreak())
            
            # Add recommendations section
            story.extend(self._create_recommendations_section(
                report.recommendations,
                report.language
            ))
            
            # Build PDF
            doc.build(story, onFirstPage=self._add_header_footer, onLaterPages=self._add_header_footer)
            
            # Get PDF bytes
            pdf_bytes = buffer.getvalue()
            buffer.close()
            
            return pdf_bytes
            
        except Exception as e:
            logger.error(f"Failed to generate PDF: {e}")
            raise
    
    def _create_cover_page(
        self,
        report: ExecutiveReport,
        organization_name: str,
        logo_path: Optional[str]
    ) -> List[Any]:
        """Create cover page elements."""
        elements = []
        
        # Add logo if provided
        if logo_path:
            try:
                logo = Image(logo_path, width=2*inch, height=1*inch)
                elements.append(logo)
                elements.append(Spacer(1, 0.5*inch))
            except Exception as e:
                logger.warning(f"Failed to add logo: {e}")
        
        # Add title
        title_text = self._translate("Executive Code Quality Report", report.language)
        elements.append(Paragraph(title_text, self.styles['CustomTitle']))
        
        # Add organization name
        elements.append(Spacer(1, 0.3*inch))
        elements.append(Paragraph(organization_name, self.styles['CustomSubtitle']))
        
        # Add date range
        elements.append(Spacer(1, 0.5*inch))
        date_format = "%d/%m/%Y" if report.language == Language.SPANISH else "%m/%d/%Y"
        date_range = f"{report.time_range.start_date.strftime(date_format)} - {report.time_range.end_date.strftime(date_format)}"
        elements.append(Paragraph(date_range, self.styles['Normal']))
        
        # Add generation date
        elements.append(Spacer(1, 2*inch))
        generated_text = self._translate("Generated on", report.language)
        generation_date = f"{generated_text}: {report.generated_at.strftime(date_format + ' %H:%M UTC')}"
        elements.append(Paragraph(generation_date, self.styles['Normal']))
        
        return elements
    
    def _create_executive_summary(
        self,
        summary: ExecutiveSummary,
        language: Language
    ) -> List[Any]:
        """Create executive summary section."""
        elements = []
        
        # Section title
        title = self._translate("Executive Summary", language)
        elements.append(Paragraph(title, self.styles['CustomSubtitle']))
        
        # Summary text
        elements.append(Paragraph(summary.summary_text, self.styles['ExecutiveSummary']))
        elements.append(Spacer(1, 0.3*inch))
        
        # Key metrics table
        if summary.key_metrics:
            elements.append(Paragraph(
                self._translate("Key Performance Indicators", language),
                self.styles['SectionHeader']
            ))
            elements.append(self._create_key_metrics_table(summary.key_metrics, language))
            elements.append(Spacer(1, 0.3*inch))
        
        # Critical issues
        if summary.critical_issues:
            elements.append(Paragraph(
                self._translate("Critical Issues Requiring Attention", language),
                self.styles['SectionHeader']
            ))
            for issue in summary.critical_issues[:3]:  # Top 3 issues
                elements.append(self._create_issue_box(issue, language))
                elements.append(Spacer(1, 0.1*inch))
        
        # Next steps
        if summary.next_steps:
            elements.append(Spacer(1, 0.2*inch))
            elements.append(Paragraph(
                self._translate("Immediate Next Steps", language),
                self.styles['SectionHeader']
            ))
            
            step_items = []
            for step in summary.next_steps[:5]:  # Top 5 steps
                step_text = f"<b>{step.step_number}.</b> {step.description} ({step.responsible_party} - {step.timeline})"
                step_items.append(ListItem(
                    Paragraph(step_text, self.styles['Normal']),
                    leftIndent=20,
                    bulletColor=self.colors['primary']
                ))
            
            elements.append(ListFlowable(step_items, bulletType='bullet'))
        
        return elements
    
    def _create_metrics_dashboard(
        self,
        business_metrics: BusinessMetrics,
        language: Language
    ) -> List[Any]:
        """Create metrics dashboard section."""
        elements = []
        
        # Section title
        title = self._translate("Performance Dashboard", language)
        elements.append(Paragraph(title, self.styles['CustomSubtitle']))
        
        # Create metrics grid
        metrics_data = [
            [
                self._create_metric_card(
                    self._translate("Code Quality", language),
                    f"{business_metrics.overall_quality_score:.1f}",
                    "/100",
                    business_metrics.quality_trend,
                    self._get_quality_color(business_metrics.overall_quality_score)
                ),
                self._create_metric_card(
                    self._translate("Technical Debt", language),
                    f"{business_metrics.technical_debt.total_hours:.0f}",
                    self._translate("hours", language),
                    TrendDirection.STABLE,
                    self.colors['warning']
                )
            ],
            [
                self._create_metric_card(
                    self._translate("Security Risk", language),
                    self._translate_risk_level(business_metrics.security_metrics.overall_risk_level, language),
                    "",
                    TrendDirection.STABLE,
                    self._get_risk_color(business_metrics.security_metrics.overall_risk_level)
                ),
                self._create_metric_card(
                    self._translate("Team Productivity", language),
                    f"{business_metrics.team_productivity.overall_score:.0f}%",
                    "",
                    TrendDirection.IMPROVING if business_metrics.team_productivity.overall_score > 70 else TrendDirection.DEGRADING,
                    self.colors['info']
                )
            ]
        ]
        
        # Create table
        metrics_table = Table(metrics_data, colWidths=[3.5*inch, 3.5*inch])
        metrics_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 1, self.colors['border']),
            ('BACKGROUND', (0, 0), (-1, -1), self.colors['background']),
            ('TOPPADDING', (0, 0), (-1, -1), 20),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 20),
        ]))
        
        elements.append(metrics_table)
        elements.append(Spacer(1, 0.3*inch))
        
        # Add quality trend chart
        elements.append(self._create_quality_trend_chart(business_metrics, language))
        
        return elements
    
    def _create_dora_section(
        self,
        dora_metrics: Any,
        language: Language
    ) -> List[Any]:
        """Create DORA metrics section."""
        elements = []
        
        # Section title
        title = self._translate("Software Delivery Performance (DORA)", language)
        elements.append(Paragraph(title, self.styles['CustomSubtitle']))
        
        # Overall performance
        performance_text = self._translate(
            f"Your team's overall DORA performance is <b>{dora_metrics.performance_rating.overall_category.value}</b> "
            f"with a score of <b>{dora_metrics.performance_rating.overall_score:.1f}/4.0</b>",
            language
        )
        elements.append(Paragraph(performance_text, self.styles['Normal']))
        elements.append(Spacer(1, 0.2*inch))
        
        # DORA metrics table
        dora_data = [
            [
                self._translate("Metric", language),
                self._translate("Current Performance", language),
                self._translate("Category", language),
                self._translate("Industry Elite", language)
            ],
            [
                self._translate("Deployment Frequency", language),
                f"{dora_metrics.deployment_frequency.deployments_per_day:.2f}/day",
                self._translate(dora_metrics.deployment_frequency.performance_category.value, language),
                ">1/day"
            ],
            [
                self._translate("Lead Time for Changes", language),
                f"{dora_metrics.lead_time_for_changes.stats.median:.1f} hours",
                self._translate(dora_metrics.lead_time_for_changes.performance_category.value, language),
                "<1 hour"
            ],
            [
                self._translate("Change Failure Rate", language),
                f"{dora_metrics.change_failure_rate.failure_rate_percentage:.1f}%",
                self._translate(dora_metrics.change_failure_rate.performance_category.value, language),
                "<15%"
            ],
            [
                self._translate("Time to Recovery", language),
                f"{dora_metrics.time_to_recovery.stats.median_hours:.1f} hours",
                self._translate(dora_metrics.time_to_recovery.performance_category.value, language),
                "<1 hour"
            ]
        ]
        
        dora_table = Table(dora_data, colWidths=[2.5*inch, 1.8*inch, 1.2*inch, 1.5*inch])
        dora_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), self.colors['primary']),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, self.colors['border'])
        ]))
        
        elements.append(dora_table)
        elements.append(Spacer(1, 0.3*inch))
        
        # Add spider chart for DORA metrics
        elements.append(self._create_dora_spider_chart(dora_metrics))
        
        return elements
    
    def _create_financial_section(
        self,
        business_metrics: BusinessMetrics,
        language: Language
    ) -> List[Any]:
        """Create financial impact section."""
        elements = []
        
        # Section title
        title = self._translate("Financial Impact Analysis", language)
        elements.append(Paragraph(title, self.styles['CustomSubtitle']))
        
        # Technical debt cost
        debt_text = self._translate(
            f"Current technical debt represents <b>{self._format_currency(business_metrics.technical_debt.estimated_cost)}</b> "
            f"in remediation costs, with a monthly carrying cost of <b>{self._format_currency(business_metrics.technical_debt.monthly_interest)}</b>",
            language
        )
        elements.append(Paragraph(debt_text, self.styles['Normal']))
        elements.append(Spacer(1, 0.2*inch))
        
        # ROI scenarios table
        if business_metrics.roi_analysis.investment_scenarios:
            elements.append(Paragraph(
                self._translate("Investment Scenarios", language),
                self.styles['SectionHeader']
            ))
            
            roi_data = [
                [
                    self._translate("Scenario", language),
                    self._translate("Investment", language),
                    self._translate("Expected Return", language),
                    self._translate("Payback Period", language),
                    self._translate("3-Year ROI", language)
                ]
            ]
            
            for scenario in business_metrics.roi_analysis.investment_scenarios[:3]:
                roi_percentage = (scenario.expected_return * 3 - scenario.investment_amount) / scenario.investment_amount * 100
                roi_data.append([
                    scenario.scenario_name,
                    self._format_currency(scenario.investment_amount),
                    self._format_currency(scenario.expected_return) + "/year",
                    f"{scenario.payback_period_months} months",
                    f"{roi_percentage:.0f}%"
                ])
            
            roi_table = Table(roi_data, colWidths=[1.8*inch, 1.3*inch, 1.5*inch, 1.2*inch, 1.2*inch])
            roi_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), self.colors['secondary']),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, self.colors['border'])
            ]))
            
            elements.append(roi_table)
        
        elements.append(Spacer(1, 0.3*inch))
        
        # Add cost breakdown pie chart
        elements.append(self._create_cost_breakdown_chart(business_metrics, language))
        
        return elements
    
    def _create_recommendations_section(
        self,
        recommendations: List[StrategicRecommendation],
        language: Language
    ) -> List[Any]:
        """Create recommendations section."""
        elements = []
        
        # Section title
        title = self._translate("Strategic Recommendations", language)
        elements.append(Paragraph(title, self.styles['CustomSubtitle']))
        
        # Introduction
        intro_text = self._translate(
            "Based on the analysis, we recommend the following strategic initiatives to improve "
            "code quality, reduce technical debt, and enhance team productivity:",
            language
        )
        elements.append(Paragraph(intro_text, self.styles['Normal']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Recommendations
        for i, recommendation in enumerate(recommendations[:5], 1):
            elements.append(self._create_recommendation_box(recommendation, i, language))
            elements.append(Spacer(1, 0.2*inch))
        
        return elements
    
    def _create_metric_card(
        self,
        title: str,
        value: str,
        unit: str,
        trend: TrendDirection,
        color: Any
    ) -> Table:
        """Create a metric card for the dashboard."""
        # Create drawing for trend arrow
        trend_drawing = Drawing(30, 30)
        if trend == TrendDirection.IMPROVING:
            arrow = self._create_arrow(15, 20, 15, 10, self.colors['success'])
            trend_drawing.add(arrow)
        elif trend == TrendDirection.DEGRADING:
            arrow = self._create_arrow(15, 10, 15, 20, self.colors['danger'])
            trend_drawing.add(arrow)
        else:
            line = Line(10, 15, 20, 15, strokeColor=self.colors['secondary'], strokeWidth=2)
            trend_drawing.add(line)
        
        # Create table for metric card
        data = [
            [Paragraph(title, self.styles['Normal'])],
            [Paragraph(f"<font color='{color}' size='20'><b>{value}</b></font>{unit}", self.styles['Normal'])],
            [trend_drawing]
        ]
        
        table = Table(data, colWidths=[3*inch], rowHeights=[0.3*inch, 0.5*inch, 0.3*inch])
        table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BACKGROUND', (0, 0), (-1, -1), colors.white),
            ('BOX', (0, 0), (-1, -1), 1, self.colors['border']),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ]))
        
        return table
    
    def _create_key_metrics_table(
        self,
        metrics: List[KeyMetric],
        language: Language
    ) -> Table:
        """Create key metrics table."""
        data = [
            [
                self._translate("Metric", language),
                self._translate("Current", language),
                self._translate("Target", language),
                self._translate("Trend", language),
                self._translate("Impact", language)
            ]
        ]
        
        for metric in metrics:
            trend_symbol = self._get_trend_symbol(metric.trend)
            impact_text = self._translate_impact(metric.business_impact, language)
            impact_color = self._get_impact_color(metric.business_impact)
            
            data.append([
                metric.name,
                metric.value,
                metric.target_value or "-",
                trend_symbol,
                Paragraph(f"<font color='{impact_color}'>{impact_text}</font>", self.styles['Normal'])
            ])
        
        table = Table(data, colWidths=[2*inch, 1.2*inch, 1.2*inch, 0.8*inch, 1.8*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), self.colors['primary']),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, self.colors['border']),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, self.colors['background']])
        ]))
        
        return table
    
    def _create_issue_box(self, issue: Any, language: Language) -> Table:
        """Create a box for critical issue display."""
        # Issue severity indicator
        severity_color = self._get_risk_color(RiskLevel.HIGH)
        
        data = [
            [
                Paragraph(f"<b>{issue.issue_type}</b>", self.styles['Normal']),
                Paragraph(f"<font color='{severity_color}'><b>{issue.resolution_timeline}</b></font>", self.styles['Normal'])
            ],
            [
                Paragraph(issue.description, self.styles['Normal']),
                ""
            ],
            [
                Paragraph(f"<b>{self._translate('Business Impact', language)}:</b> {issue.business_impact}", self.styles['Normal']),
                Paragraph(f"<b>{self._translate('Investment', language)}:</b> {self._format_currency(issue.required_investment)}", self.styles['Normal'])
            ]
        ]
        
        table = Table(data, colWidths=[4.5*inch, 2.5*inch])
        table.setStyle(TableStyle([
            ('SPAN', (0, 1), (1, 1)),
            ('BACKGROUND', (0, 0), (-1, -1), colors.white),
            ('BOX', (0, 0), (-1, -1), 2, severity_color),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
            ('VALIGN', (0, 0), (-1, -1), 'TOP')
        ]))
        
        return table
    
    def _create_recommendation_box(
        self,
        recommendation: StrategicRecommendation,
        number: int,
        language: Language
    ) -> KeepTogether:
        """Create a recommendation box."""
        elements = []
        
        # Title with number
        title_text = f"<font color='{self.colors['primary']}'><b>{number}.</b></font> <b>{recommendation.title}</b>"
        elements.append(Paragraph(title_text, self.styles['SectionHeader']))
        
        # Description
        elements.append(Paragraph(recommendation.description, self.styles['Normal']))
        elements.append(Spacer(1, 0.1*inch))
        
        # Key details table
        details_data = [
            [
                self._translate("Investment", language),
                self._format_currency(recommendation.estimated_investment.cost_estimate)
            ],
            [
                self._translate("Expected ROI", language),
                f"{recommendation.expected_roi:.0f}%"
            ],
            [
                self._translate("Timeline", language),
                f"{recommendation.timeline.estimated_duration_weeks} weeks"
            ],
            [
                self._translate("Risk Level", language),
                self._translate_risk_level(recommendation.risk_level, language)
            ]
        ]
        
        details_table = Table(details_data, colWidths=[1.5*inch, 2*inch])
        details_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3)
        ]))
        
        elements.append(details_table)
        elements.append(Spacer(1, 0.1*inch))
        
        # Business justification
        elements.append(Paragraph(
            f"<b>{self._translate('Business Justification', language)}:</b> {recommendation.business_justification}",
            self.styles['Normal']
        ))
        
        # Success metrics
        if recommendation.success_metrics:
            elements.append(Spacer(1, 0.1*inch))
            elements.append(Paragraph(
                f"<b>{self._translate('Success Metrics', language)}:</b>",
                self.styles['Normal']
            ))
            
            metrics_items = []
            for metric in recommendation.success_metrics[:3]:
                metrics_items.append(ListItem(
                    Paragraph(metric, self.styles['Recommendation']),
                    leftIndent=30,
                    bulletColor=self.colors['success']
                ))
            
            elements.append(ListFlowable(metrics_items, bulletType='bullet'))
        
        # Keep together
        return KeepTogether(elements)
    
    def _create_quality_trend_chart(
        self,
        business_metrics: BusinessMetrics,
        language: Language
    ) -> Drawing:
        """Create quality trend line chart."""
        drawing = Drawing(400, 200)
        
        # Create line chart
        lc = HorizontalLineChart()
        lc.x = 50
        lc.y = 50
        lc.height = 125
        lc.width = 300
        
        # Sample data (would come from historical data)
        data = [[65, 68, 70, 72, 73, 75.5]]  # Quality scores over time
        lc.data = data
        
        # Styling
        lc.lines[0].strokeColor = self.colors['primary']
        lc.lines[0].strokeWidth = 2
        lc.lines[0].symbol = makeMarker('Circle')
        lc.lines[0].symbol.strokeColor = self.colors['primary']
        lc.lines[0].symbol.fillColor = colors.white
        lc.lines[0].symbol.size = 5
        
        # Axes
        lc.valueAxis.valueMin = 0
        lc.valueAxis.valueMax = 100
        lc.valueAxis.valueStep = 20
        lc.valueAxis.labelTextFormat = '%d'
        
        lc.categoryAxis.categoryNames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
        lc.categoryAxis.labels.boxAnchor = 'n'
        
        # Title
        title = self._translate("Quality Score Trend", language)
        drawing.add(lc)
        drawing.add(self._create_chart_title(title, 200, 190))
        
        return drawing
    
    def _create_dora_spider_chart(self, dora_metrics: Any) -> Drawing:
        """Create DORA metrics spider chart."""
        drawing = Drawing(400, 300)
        
        # Create spider chart
        spider = SpiderChart()
        spider.x = 100
        spider.y = 75
        spider.width = 200
        spider.height = 200
        
        # Data
        data = [[
            dora_metrics.performance_rating.deployment_score,
            dora_metrics.performance_rating.lead_time_score,
            dora_metrics.performance_rating.failure_rate_score,
            dora_metrics.performance_rating.recovery_time_score
        ]]
        spider.data = data
        
        # Labels
        spider.labels = [
            'Deployment\nFrequency',
            'Lead Time',
            'Failure Rate',
            'Recovery Time'
        ]
        
        # Styling
        spider.strands[0].strokeColor = self.colors['primary']
        spider.strands[0].fillColor = colors.toColor('rgba(37, 99, 235, 0.3)')
        spider.strands[0].strokeWidth = 2
        
        # Scale
        spider.strandLabels.format = '%d'
        spider.scale = 4  # Max score
        
        drawing.add(spider)
        drawing.add(self._create_chart_title("DORA Performance", 200, 290))
        
        return drawing
    
    def _create_cost_breakdown_chart(
        self,
        business_metrics: BusinessMetrics,
        language: Language
    ) -> Drawing:
        """Create cost breakdown pie chart."""
        drawing = Drawing(400, 250)
        
        # Create pie chart
        pie = Pie()
        pie.x = 150
        pie.y = 50
        pie.width = 150
        pie.height = 150
        
        # Data
        direct_cost = float(business_metrics.technical_debt.estimated_cost)
        annual_interest = float(business_metrics.technical_debt.monthly_interest * 12)
        productivity_loss = direct_cost * 0.25  # Estimate
        
        pie.data = [direct_cost, annual_interest, productivity_loss]
        pie.labels = [
            f"{self._translate('Direct Cost', language)}\n{self._format_currency(Decimal(str(direct_cost)))}",
            f"{self._translate('Annual Interest', language)}\n{self._format_currency(Decimal(str(annual_interest)))}",
            f"{self._translate('Productivity Loss', language)}\n{self._format_currency(Decimal(str(productivity_loss)))}"
        ]
        
        # Colors
        pie.slices.strokeWidth = 0.5
        pie.slices[0].fillColor = self.colors['primary']
        pie.slices[1].fillColor = self.colors['warning']
        pie.slices[2].fillColor = self.colors['danger']
        
        # Labels
        pie.slices.labelRadius = 1.2
        pie.slices.fontName = 'Helvetica'
        pie.slices.fontSize = 9
        
        drawing.add(pie)
        drawing.add(self._create_chart_title(
            self._translate("Technical Debt Cost Breakdown", language),
            200, 240
        ))
        
        return drawing
    
    def _create_arrow(self, x1: float, y1: float, x2: float, y2: float, color: Any) -> Drawing:
        """Create an arrow shape."""
        group = Drawing(30, 30)
        
        # Line
        line = Line(x1, y1, x2, y2, strokeColor=color, strokeWidth=2)
        group.add(line)
        
        # Arrowhead
        if y2 < y1:  # Up arrow
            points = [x2-3, y2+3, x2, y2, x2+3, y2+3]
        else:  # Down arrow
            points = [x2-3, y2-3, x2, y2, x2+3, y2-3]
        
        from reportlab.graphics.shapes import Polygon
        arrowhead = Polygon(points, fillColor=color, strokeColor=color)
        group.add(arrowhead)
        
        return group
    
    def _create_chart_title(self, title: str, x: float, y: float) -> Drawing:
        """Create chart title."""
        from reportlab.graphics.shapes import String
        title_string = String(x, y, title, textAnchor='middle')
        title_string.fontName = 'Helvetica-Bold'
        title_string.fontSize = 12
        title_string.fillColor = self.colors['text']
        return title_string
    
    def _add_header_footer(self, canvas, doc):
        """Add header and footer to pages."""
        canvas.saveState()
        
        # Header
        canvas.setFont('Helvetica', 9)
        canvas.setFillColor(self.colors['secondary'])
        canvas.drawString(inch, 10.5*inch, "CodeAnt Executive Report")
        canvas.drawRightString(7.5*inch, 10.5*inch, datetime.now().strftime("%B %Y"))
        
        # Footer
        canvas.drawString(inch, 0.75*inch, f"Page {doc.page}")
        canvas.drawCentredString(4.25*inch, 0.75*inch, "Confidential")
        canvas.drawRightString(7.5*inch, 0.75*inch, "© CodeAnt AI")
        
        # Line separators
        canvas.setStrokeColor(self.colors['border'])
        canvas.line(inch, 10.3*inch, 7.5*inch, 10.3*inch)
        canvas.line(inch, inch, 7.5*inch, inch)
        
        canvas.restoreState()
    
    # Helper methods
    def _translate(self, text: str, language: Language) -> str:
        """Translate text to target language."""
        # Simplified translation - would use proper i18n
        if language == Language.SPANISH:
            translations = {
                "Executive Code Quality Report": "Reporte Ejecutivo de Calidad de Código",
                "Executive Summary": "Resumen Ejecutivo",
                "Generated on": "Generado el",
                "Key Performance Indicators": "Indicadores Clave de Rendimiento",
                "Critical Issues Requiring Attention": "Problemas Críticos que Requieren Atención",
                "Immediate Next Steps": "Próximos Pasos Inmediatos",
                "Performance Dashboard": "Panel de Rendimiento",
                "Code Quality": "Calidad del Código",
                "Technical Debt": "Deuda Técnica",
                "Security Risk": "Riesgo de Seguridad",
                "Team Productivity": "Productividad del Equipo",
                "hours": "horas",
                "Software Delivery Performance (DORA)": "Rendimiento de Entrega de Software (DORA)",
                "Metric": "Métrica",
                "Current Performance": "Rendimiento Actual",
                "Category": "Categoría",
                "Industry Elite": "Élite de la Industria",
                "Deployment Frequency": "Frecuencia de Despliegue",
                "Lead Time for Changes": "Tiempo de Entrega de Cambios",
                "Change Failure Rate": "Tasa de Fallo de Cambios",
                "Time to Recovery": "Tiempo de Recuperación",
                "Financial Impact Analysis": "Análisis de Impacto Financiero",
                "Investment Scenarios": "Escenarios de Inversión",
                "Scenario": "Escenario",
                "Investment": "Inversión",
                "Expected Return": "Retorno Esperado",
                "Payback Period": "Período de Recuperación",
                "3-Year ROI": "ROI a 3 Años",
                "Strategic Recommendations": "Recomendaciones Estratégicas",
                "Business Impact": "Impacto en el Negocio",
                "Business Justification": "Justificación de Negocio",
                "Success Metrics": "Métricas de Éxito",
                "Timeline": "Cronograma",
                "Risk Level": "Nivel de Riesgo",
                "Direct Cost": "Costo Directo",
                "Annual Interest": "Interés Anual",
                "Productivity Loss": "Pérdida de Productividad",
                "Quality Score Trend": "Tendencia de Puntuación de Calidad",
                "Technical Debt Cost Breakdown": "Desglose de Costos de Deuda Técnica"
            }
            return translations.get(text, text)
        
        return text
    
    def _translate_risk_level(self, risk_level: RiskLevel, language: Language) -> str:
        """Translate risk level."""
        if language == Language.SPANISH:
            translations = {
                RiskLevel.CRITICAL: "Crítico",
                RiskLevel.HIGH: "Alto",
                RiskLevel.MEDIUM: "Medio",
                RiskLevel.LOW: "Bajo",
                RiskLevel.MINIMAL: "Mínimo"
            }
            return translations.get(risk_level, risk_level.value)
        
        return risk_level.value.capitalize()
    
    def _translate_impact(self, impact: BusinessImpact, language: Language) -> str:
        """Translate business impact."""
        if language == Language.SPANISH:
            translations = {
                BusinessImpact.POSITIVE: "Positivo",
                BusinessImpact.NEGATIVE: "Negativo",
                BusinessImpact.NEUTRAL: "Neutral",
                BusinessImpact.CRITICAL: "Crítico"
            }
            return translations.get(impact, impact.value)
        
        return impact.value.capitalize()
    
    def _format_currency(self, amount: Decimal) -> str:
        """Format currency amount."""
        return f"${amount:,.0f}"
    
    def _get_quality_color(self, score: float) -> Any:
        """Get color based on quality score."""
        if score >= 85:
            return self.colors['success']
        elif score >= 70:
            return self.colors['info']
        elif score >= 60:
            return self.colors['warning']
        else:
            return self.colors['danger']
    
    def _get_risk_color(self, risk_level: RiskLevel) -> Any:
        """Get color based on risk level."""
        risk_colors = {
            RiskLevel.CRITICAL: self.colors['danger'],
            RiskLevel.HIGH: colors.orange,
            RiskLevel.MEDIUM: self.colors['warning'],
            RiskLevel.LOW: self.colors['info'],
            RiskLevel.MINIMAL: self.colors['success']
        }
        return risk_colors.get(risk_level, self.colors['secondary'])
    
    def _get_impact_color(self, impact: BusinessImpact) -> Any:
        """Get color based on business impact."""
        impact_colors = {
            BusinessImpact.POSITIVE: self.colors['success'],
            BusinessImpact.NEGATIVE: self.colors['danger'],
            BusinessImpact.NEUTRAL: self.colors['secondary'],
            BusinessImpact.CRITICAL: self.colors['danger']
        }
        return impact_colors.get(impact, self.colors['secondary'])
    
    def _get_trend_symbol(self, trend: TrendDirection) -> str:
        """Get symbol for trend direction."""
        symbols = {
            TrendDirection.IMPROVING: "↑",
            TrendDirection.DEGRADING: "↓",
            TrendDirection.STABLE: "→",
            TrendDirection.VOLATILE: "↕"
        }
        return symbols.get(trend, "→")
