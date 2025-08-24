"""
Cost calculator for PDF processing and AI services.
Provides detailed cost analysis for different processing methods.
"""
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)
logger.propagate = False  # Prevent duplicate logging


class ServiceType(Enum):
    """Types of services used in PDF processing"""
    PDF_EXTRACTION = "pdf_extraction"
    BOUNDARY_EXTRACTION = "boundary_extraction"
    CAD_FORMATTING = "cad_formatting"
    OPENAI_API = "openai_api"


@dataclass
class CostBreakdown:
    """Detailed cost breakdown for a processing operation"""
    service_type: ServiceType
    service_name: str
    units_processed: int
    unit_type: str  # pages, tokens, characters, etc.
    cost_per_unit: float
    total_cost: float
    processing_time: float
    notes: str = ""


@dataclass
class TotalCostAnalysis:
    """Complete cost analysis for PDF processing workflow"""
    pdf_extraction_cost: float
    boundary_extraction_cost: float
    cad_formatting_cost: float
    total_cost: float
    breakdowns: List[CostBreakdown]
    estimated_time: float
    cost_comparison: Dict[str, float]  # Different method comparisons
    
    def format_summary(self) -> str:
        """Format cost analysis as readable summary"""
        summary = []
        summary.append("💰 COST ANALYSIS SUMMARY")
        summary.append("=" * 40)
        summary.append(f"PDF Text Extraction: ${self.pdf_extraction_cost:.4f}")
        summary.append(f"Boundary Analysis: ${self.boundary_extraction_cost:.4f}")
        summary.append(f"CAD Formatting: ${self.cad_formatting_cost:.4f}")
        summary.append("-" * 40)
        summary.append(f"TOTAL COST: ${self.total_cost:.4f}")
        summary.append(f"Estimated Time: {self.estimated_time:.1f} seconds")
        summary.append("")
        
        if self.cost_comparison:
            summary.append("📊 METHOD COMPARISON:")
            for method, cost in self.cost_comparison.items():
                summary.append(f"  {method}: ${cost:.4f}")
        
        return "\n".join(summary)


class CostCalculator:
    """
    Calculator for PDF processing costs across different services.
    Provides detailed cost analysis and comparisons.
    """
    
    # Current pricing for various services (as of 2024)
    PRICING = {
        # PDF Extraction Services
        "pdfminer_tesseract": {
            "cost_per_page": 0.0,  # Free
            "setup_cost": 0.0,
            "description": "Free OCR using PDFMiner + Tesseract"
        },
        "llamaextract": {
            "cost_per_page": 0.003,  # $0.003 per page
            "setup_cost": 0.0,
            "description": "LlamaIndex premium OCR service"
        },
        "google_document_ai": {
            "cost_per_page": 0.0015,  # $0.0015 per page
            "setup_cost": 0.0,
            "description": "Google Cloud Document AI"
        },
        
        # OpenAI API Pricing (GPT-4o as example)
        "openai_gpt4o": {
            "input_cost_per_1k_tokens": 0.005,  # $0.005 per 1K input tokens
            "output_cost_per_1k_tokens": 0.015,  # $0.015 per 1K output tokens
            "description": "OpenAI GPT-4o API"
        },
        "openai_gpt4o_mini": {
            "input_cost_per_1k_tokens": 0.00015,  # $0.00015 per 1K input tokens
            "output_cost_per_1k_tokens": 0.0006,  # $0.0006 per 1K output tokens
            "description": "OpenAI GPT-4o-mini API"
        },
        "openai_o1_preview": {
            "input_cost_per_1k_tokens": 0.015,  # $0.015 per 1K input tokens
            "output_cost_per_1k_tokens": 0.060,  # $0.060 per 1K output tokens
            "description": "OpenAI o1-preview (reasoning model)"
        }
    }
    
    def __init__(self):
        """Initialize cost calculator"""
        logger.info("Cost Calculator initialized")
    
    def calculate_pdf_extraction_cost(self, page_count: int, method: str) -> CostBreakdown:
        """Calculate cost for PDF text extraction"""
        method_key = method.lower().replace("+", "_").replace(" ", "_")
        
        if method_key in self.PRICING:
            pricing = self.PRICING[method_key]
            cost_per_page = pricing["cost_per_page"]
            setup_cost = pricing["setup_cost"]
            total_cost = (page_count * cost_per_page) + setup_cost
            
            return CostBreakdown(
                service_type=ServiceType.PDF_EXTRACTION,
                service_name=pricing["description"],
                units_processed=page_count,
                unit_type="pages",
                cost_per_unit=cost_per_page,
                total_cost=total_cost,
                processing_time=self._estimate_extraction_time(page_count, method),
                notes=f"Method: {method}"
            )
        else:
            # Default estimation for unknown methods
            return CostBreakdown(
                service_type=ServiceType.PDF_EXTRACTION,
                service_name=f"Unknown method: {method}",
                units_processed=page_count,
                unit_type="pages",
                cost_per_unit=0.002,  # Default estimate
                total_cost=page_count * 0.002,
                processing_time=page_count * 5,  # 5 seconds per page estimate
                notes="Estimated cost for unknown method"
            )
    
    def calculate_openai_cost(self, input_text: str, estimated_output_tokens: int, model: str) -> CostBreakdown:
        """Calculate OpenAI API cost for text processing"""
        # Estimate input tokens (rough approximation: 1 token ≈ 0.75 words)
        input_tokens = len(input_text.split()) * 1.3
        
        # Map model name to pricing key
        model_key = self._map_model_to_pricing_key(model)
        
        if model_key in self.PRICING:
            pricing = self.PRICING[model_key]
            input_cost = (input_tokens / 1000) * pricing["input_cost_per_1k_tokens"]
            output_cost = (estimated_output_tokens / 1000) * pricing["output_cost_per_1k_tokens"]
            total_cost = input_cost + output_cost
            
            return CostBreakdown(
                service_type=ServiceType.OPENAI_API,
                service_name=pricing["description"],
                units_processed=int(input_tokens + estimated_output_tokens),
                unit_type="tokens",
                cost_per_unit=(input_cost + output_cost) / (input_tokens + estimated_output_tokens) * 1000,
                total_cost=total_cost,
                processing_time=self._estimate_openai_time(input_tokens + estimated_output_tokens),
                notes=f"Input: {input_tokens:.0f} tokens, Output: {estimated_output_tokens} tokens"
            )
        else:
            # Default GPT-4o pricing for unknown models
            input_cost = (input_tokens / 1000) * 0.005
            output_cost = (estimated_output_tokens / 1000) * 0.015
            total_cost = input_cost + output_cost
            
            return CostBreakdown(
                service_type=ServiceType.OPENAI_API,
                service_name=f"OpenAI {model} (estimated)",
                units_processed=int(input_tokens + estimated_output_tokens),
                unit_type="tokens",
                cost_per_unit=0.01,  # Average estimate
                total_cost=total_cost,
                processing_time=self._estimate_openai_time(input_tokens + estimated_output_tokens),
                notes=f"Estimated pricing for {model}"
            )
    
    def calculate_boundary_extraction_cost(self, text_length: int, model: str) -> CostBreakdown:
        """Calculate cost for boundary information extraction"""
        # Boundary extraction typically requires analyzing the full text
        # and producing structured output
        estimated_output_tokens = 1000  # Typical output for boundary extraction
        
        return self.calculate_openai_cost(
            input_text="x" * text_length,  # Dummy text for length calculation
            estimated_output_tokens=estimated_output_tokens,
            model=model
        )
    
    def calculate_cad_formatting_cost(self, boundary_text_length: int, model: str) -> CostBreakdown:
        """Calculate cost for CAD formatting conversion"""
        # CAD formatting requires processing boundary text and producing
        # structured survey calls
        estimated_output_tokens = 2000  # Typical output for CAD formatting
        
        cost_breakdown = self.calculate_openai_cost(
            input_text="x" * boundary_text_length,
            estimated_output_tokens=estimated_output_tokens,
            model=model
        )
        
        cost_breakdown.service_type = ServiceType.CAD_FORMATTING
        cost_breakdown.notes = "Converting boundary data to CAD format"
        
        return cost_breakdown
    
    def analyze_complete_workflow(self, 
                                page_count: int,
                                extraction_method: str,
                                extracted_text_length: int,
                                openai_model: str) -> TotalCostAnalysis:
        """
        Analyze complete cost for the entire PDF processing workflow
        
        Args:
            page_count: Number of pages in PDF
            extraction_method: PDF extraction method used
            extracted_text_length: Length of extracted text
            openai_model: OpenAI model used for processing
            
        Returns:
            Complete cost analysis
        """
        logger.info(f"Calculating complete workflow cost for {page_count} pages")
        
        # Calculate individual costs
        pdf_cost = self.calculate_pdf_extraction_cost(page_count, extraction_method)
        boundary_cost = self.calculate_boundary_extraction_cost(extracted_text_length, openai_model)
        cad_cost = self.calculate_cad_formatting_cost(extracted_text_length // 2, openai_model)  # Assume boundary text is ~half of extracted text
        
        # Calculate totals
        total_cost = pdf_cost.total_cost + boundary_cost.total_cost + cad_cost.total_cost
        total_time = pdf_cost.processing_time + boundary_cost.processing_time + cad_cost.processing_time
        
        # Create comparison with different methods
        cost_comparison = self._generate_method_comparison(page_count, extracted_text_length)
        
        return TotalCostAnalysis(
            pdf_extraction_cost=pdf_cost.total_cost,
            boundary_extraction_cost=boundary_cost.total_cost,
            cad_formatting_cost=cad_cost.total_cost,
            total_cost=total_cost,
            breakdowns=[pdf_cost, boundary_cost, cad_cost],
            estimated_time=total_time,
            cost_comparison=cost_comparison
        )
    
    def compare_extraction_methods(self, page_count: int) -> Dict[str, Tuple[float, str]]:
        """Compare costs of different PDF extraction methods"""
        methods = ["pdfminer_tesseract", "llamaextract", "google_document_ai"]
        comparison = {}
        
        for method in methods:
            cost_breakdown = self.calculate_pdf_extraction_cost(page_count, method)
            comparison[method] = (
                cost_breakdown.total_cost,
                cost_breakdown.service_name
            )
        
        return comparison
    
    def estimate_monthly_cost(self, 
                            pdfs_per_month: int,
                            avg_pages_per_pdf: int,
                            extraction_method: str,
                            openai_model: str) -> Dict[str, float]:
        """Estimate monthly costs for regular usage"""
        
        # Calculate per-PDF cost
        sample_analysis = self.analyze_complete_workflow(
            page_count=avg_pages_per_pdf,
            extraction_method=extraction_method,
            extracted_text_length=2000 * avg_pages_per_pdf,  # Estimate 2000 chars per page
            openai_model=openai_model
        )
        
        per_pdf_cost = sample_analysis.total_cost
        monthly_cost = per_pdf_cost * pdfs_per_month
        
        return {
            "per_pdf_cost": per_pdf_cost,
            "monthly_cost": monthly_cost,
            "yearly_cost": monthly_cost * 12,
            "pdf_extraction_monthly": sample_analysis.pdf_extraction_cost * pdfs_per_month,
            "ai_processing_monthly": (sample_analysis.boundary_extraction_cost + sample_analysis.cad_formatting_cost) * pdfs_per_month
        }
    
    def _map_model_to_pricing_key(self, model: str) -> str:
        """Map OpenAI model name to pricing key"""
        model_lower = model.lower()
        
        if "gpt-4o-mini" in model_lower:
            return "openai_gpt4o_mini"
        elif "gpt-4o" in model_lower:
            return "openai_gpt4o"
        elif "o1-preview" in model_lower:
            return "openai_o1_preview"
        else:
            return "openai_gpt4o"  # Default to GPT-4o pricing
    
    def _estimate_extraction_time(self, page_count: int, method: str) -> float:
        """Estimate processing time for PDF extraction"""
        method_lower = method.lower()
        
        if "tesseract" in method_lower:
            return page_count * 8.0  # 8 seconds per page for OCR
        elif "llamaextract" in method_lower:
            return page_count * 3.0  # 3 seconds per page for premium OCR
        elif "google" in method_lower:
            return page_count * 2.0  # 2 seconds per page for Google AI
        else:
            return page_count * 5.0  # Default estimate
    
    def _estimate_openai_time(self, total_tokens: int) -> float:
        """Estimate OpenAI API processing time"""
        # Rough estimate: 1000 tokens per second
        return max(total_tokens / 1000, 2.0)  # Minimum 2 seconds
    
    def _generate_method_comparison(self, page_count: int, text_length: int) -> Dict[str, float]:
        """Generate cost comparison for different processing methods"""
        methods = ["pdfminer_tesseract", "llamaextract", "google_document_ai"]
        models = ["gpt-4o", "gpt-4o-mini"]
        
        comparison = {}
        
        for method in methods:
            for model in models:
                try:
                    analysis = self.analyze_complete_workflow(
                        page_count=page_count,
                        extraction_method=method,
                        extracted_text_length=text_length,
                        openai_model=model
                    )
                    
                    method_name = method.replace("_", " ").title()
                    model_name = model.upper()
                    key = f"{method_name} + {model_name}"
                    
                    comparison[key] = analysis.total_cost
                    
                except Exception as e:
                    logger.warning(f"Failed to calculate cost for {method} + {model}: {e}")
        
        return comparison
    
    def get_cost_optimization_tips(self, analysis: TotalCostAnalysis) -> List[str]:
        """Provide cost optimization recommendations"""
        tips = []
        
        # PDF extraction optimization
        if analysis.pdf_extraction_cost > 0:
            tips.append("💡 Consider using free PDFMiner+Tesseract for simple text-based PDFs")
        
        # OpenAI model optimization
        if any("gpt-4o" in breakdown.service_name and "mini" not in breakdown.service_name 
               for breakdown in analysis.breakdowns):
            tips.append("💡 Try GPT-4o-mini for 85% cost savings with similar quality")
        
        # Batch processing
        if analysis.total_cost > 0.05:
            tips.append("💡 Process multiple PDFs in batches to optimize API usage")
        
        # Text preprocessing
        tips.append("💡 Use higher quality PDF extraction to improve AI processing accuracy")
        
        # Usage monitoring
        if analysis.total_cost > 0.10:
            tips.append("💡 Monitor usage regularly and set up cost alerts")
        
        return tips


def format_cost_display(cost: float) -> str:
    """Format cost for display with appropriate precision"""
    if cost == 0:
        return "Free"
    elif cost < 0.001:
        return f"${cost:.6f}"
    elif cost < 0.01:
        return f"${cost:.4f}"
    else:
        return f"${cost:.2f}"


def test_cost_calculator():
    """Test the cost calculator with sample data"""
    calculator = CostCalculator()
    
    # Test PDF extraction costs
    print("PDF Extraction Cost Comparison (10 pages):")
    comparison = calculator.compare_extraction_methods(10)
    for method, (cost, description) in comparison.items():
        print(f"  {description}: {format_cost_display(cost)}")
    
    print("\nComplete Workflow Analysis:")
    analysis = calculator.analyze_complete_workflow(
        page_count=10,
        extraction_method="llamaextract",
        extracted_text_length=20000,
        openai_model="gpt-4o"
    )
    
    print(analysis.format_summary())
    
    # Test monthly cost estimation
    print("\nMonthly Cost Estimation (20 PDFs, 5 pages each):")
    monthly = calculator.estimate_monthly_cost(
        pdfs_per_month=20,
        avg_pages_per_pdf=5,
        extraction_method="llamaextract",
        openai_model="gpt-4o-mini"
    )
    
    for key, value in monthly.items():
        print(f"  {key}: {format_cost_display(value)}")


if __name__ == "__main__":
    test_cost_calculator()
