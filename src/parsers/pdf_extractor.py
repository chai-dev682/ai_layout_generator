"""
PDF text extraction using multiple methods for different PDF types.
Supports pdfminer+tesseract, llamaextract, and Google Document AI.
"""
import os
import io
import base64
import logging
import tempfile
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
import time

# PDF processing libraries
import PyPDF2
import pdfplumber
import pytesseract
from PIL import Image
import fitz  # PyMuPDF for better PDF handling

# Cloud services
try:
    from llama_index.readers.file import PDFReader
    from llama_index.core import VectorStoreIndex
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    LLAMAINDEX_AVAILABLE = False
    
try:
    from google.cloud import documentai
    DOCUMENTAI_AVAILABLE = True
except ImportError:
    DOCUMENTAI_AVAILABLE = False

logger = logging.getLogger(__name__)
logger.propagate = False  # Prevent duplicate logging


class PDFExtractionMethod(Enum):
    """Available PDF extraction methods"""
    PDFMINER_TESSERACT = "pdfminer+tesseract"
    LLAMAEXTRACT = "llamaextract"
    GOOGLE_DOCUMENT_AI = "google_document_ai"


@dataclass
class PDFExtractionResult:
    """Result of PDF text extraction"""
    method: PDFExtractionMethod
    extracted_text: str
    confidence_score: float
    processing_time: float
    page_count: int
    metadata: Dict[str, Any]
    cost_estimate: float  # In USD
    warnings: List[str]
    
    @property
    def success(self) -> bool:
        """Whether extraction was successful"""
        return bool(self.extracted_text.strip())


class PDFExtractor:
    """
    Unified PDF text extraction supporting multiple methods:
    1. PDFMiner + Tesseract OCR (free, good for simple PDFs)
    2. LlamaExtract (premium, good for complex PDFs)  
    3. Google Document AI (premium, best for complex layouts)
    """
    
    def __init__(self, 
                 llamaindex_api_key: Optional[str] = None,
                 google_credentials_path: Optional[str] = None,
                 google_project_id: Optional[str] = None,
                 google_location: str = "us",
                 google_processor_id: Optional[str] = None):
        """
        Initialize PDF extractor with API credentials
        
        Args:
            llamaindex_api_key: LlamaIndex API key for LlamaExtract
            google_credentials_path: Path to Google Cloud credentials JSON
            google_project_id: Google Cloud project ID
            google_location: Google Cloud location (us, eu, etc.)
            google_processor_id: Document AI processor ID
        """
        self.llamaindex_api_key = llamaindex_api_key or os.getenv("LLAMAINDEX_API_KEY")
        self.google_credentials_path = google_credentials_path or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        self.google_project_id = google_project_id or os.getenv("GOOGLE_CLOUD_PROJECT_ID")
        self.google_location = google_location
        self.google_processor_id = google_processor_id or os.getenv("GOOGLE_DOCUMENT_AI_PROCESSOR_ID")
        
        # Check availability of services
        self._check_service_availability()
        
        logger.info("PDF Extractor initialized")
        logger.info(f"LlamaExtract available: {self.llamaextract_available}")
        logger.info(f"Google Document AI available: {self.google_ai_available}")
    
    def _check_service_availability(self):
        """Check which services are available"""
        # LlamaExtract availability
        self.llamaextract_available = (
            LLAMAINDEX_AVAILABLE and 
            self.llamaindex_api_key is not None
        )
        
        # Google Document AI availability
        self.google_ai_available = (
            DOCUMENTAI_AVAILABLE and 
            self.google_credentials_path and 
            self.google_project_id and 
            self.google_processor_id
        )
        
        if not self.llamaextract_available:
            logger.warning("LlamaExtract not available - missing API key or llama-index package")
        
        if not self.google_ai_available:
            logger.warning("Google Document AI not available - missing credentials or configuration")
    
    def extract_text(self, pdf_file_path: str, method: PDFExtractionMethod) -> PDFExtractionResult:
        """
        Extract text from PDF using specified method
        
        Args:
            pdf_file_path: Path to PDF file
            method: Extraction method to use
            
        Returns:
            PDFExtractionResult with extracted text and metadata
        """
        start_time = time.time()
        logger.info(f"Starting PDF extraction using {method.value}")
        logger.info(f"PDF file: {pdf_file_path}")
        
        try:
            # Get PDF metadata
            metadata = self._get_pdf_metadata(pdf_file_path)
            page_count = metadata.get("page_count", 0)
            
            logger.info(f"PDF has {page_count} pages")
            
            # Route to appropriate extraction method
            if method == PDFExtractionMethod.PDFMINER_TESSERACT:
                result = self._extract_with_pdfminer_tesseract(pdf_file_path, metadata)
            elif method == PDFExtractionMethod.LLAMAEXTRACT:
                if not self.llamaextract_available:
                    raise ValueError("LlamaExtract not available - check API key and dependencies")
                result = self._extract_with_llamaextract(pdf_file_path, metadata)
            elif method == PDFExtractionMethod.GOOGLE_DOCUMENT_AI:
                if not self.google_ai_available:
                    raise ValueError("Google Document AI not available - check credentials and configuration")
                result = self._extract_with_google_ai(pdf_file_path, metadata)
            else:
                raise ValueError(f"Unknown extraction method: {method}")
            
            processing_time = time.time() - start_time
            result.processing_time = processing_time
            
            logger.info(f"PDF extraction completed in {processing_time:.2f} seconds")
            logger.info(f"Extracted text length: {len(result.extracted_text)} characters")
            logger.info(f"Confidence score: {result.confidence_score:.2f}")
            logger.info(f"Estimated cost: ${result.cost_estimate:.4f}")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"PDF extraction failed after {processing_time:.2f} seconds: {str(e)}")
            
            # Return failed result
            return PDFExtractionResult(
                method=method,
                extracted_text="",
                confidence_score=0.0,
                processing_time=processing_time,
                page_count=0,
                metadata={},
                cost_estimate=0.0,
                warnings=[f"Extraction failed: {str(e)}"]
            )
    
    def _get_pdf_metadata(self, pdf_file_path: str) -> Dict[str, Any]:
        """Get PDF metadata using PyMuPDF"""
        try:
            doc = fitz.open(pdf_file_path)
            metadata = {
                "page_count": doc.page_count,
                "file_size": os.path.getsize(pdf_file_path),
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "subject": doc.metadata.get("subject", ""),
                "creator": doc.metadata.get("creator", ""),
                "producer": doc.metadata.get("producer", ""),
            }
            doc.close()
            return metadata
        except Exception as e:
            logger.warning(f"Failed to get PDF metadata: {e}")
            return {"page_count": 0, "file_size": 0}
    
    def _extract_with_pdfminer_tesseract(self, pdf_file_path: str, metadata: Dict[str, Any]) -> PDFExtractionResult:
        """Extract text using PDFMiner + Tesseract OCR (free method)"""
        logger.info("Using PDFMiner + Tesseract OCR extraction")
        
        extracted_text = ""
        warnings = []
        confidence_scores = []
        
        try:
            # First try pdfplumber for text extraction
            logger.info("Step 1: Attempting text extraction with pdfplumber")
            with pdfplumber.open(pdf_file_path) as pdf:
                text_pages = []
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text_pages.append(page_text)
                        logger.info(f"Page {page_num + 1}: Extracted {len(page_text)} characters")
                    else:
                        logger.info(f"Page {page_num + 1}: No text found, will need OCR")
                
                if text_pages:
                    extracted_text = "\n\n".join(text_pages)
                    confidence_scores.append(0.8)  # High confidence for direct text extraction
            
            # If no text extracted or very little text, use OCR
            if not extracted_text.strip() or len(extracted_text.strip()) < 100:
                logger.info("Step 2: Using Tesseract OCR for image-based PDF")
                warnings.append("PDF appears to be image-based, using OCR")
                
                # Convert PDF pages to images and OCR
                doc = fitz.open(pdf_file_path)
                ocr_pages = []
                
                for page_num in range(doc.page_count):
                    page = doc[page_num]
                    
                    # Convert page to image
                    mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better OCR
                    pix = page.get_pixmap(matrix=mat)
                    img_data = pix.tobytes("png")
                    
                    # OCR the image
                    image = Image.open(io.BytesIO(img_data))
                    
                    # Get OCR result with confidence
                    ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
                    
                    # Extract text and calculate confidence
                    page_words = []
                    page_confidences = []
                    
                    for i in range(len(ocr_data['text'])):
                        word = ocr_data['text'][i].strip()
                        conf = ocr_data['conf'][i]
                        
                        if word and conf > 30:  # Filter out low-confidence words
                            page_words.append(word)
                            page_confidences.append(conf)
                    
                    if page_words:
                        page_text = ' '.join(page_words)
                        ocr_pages.append(page_text)
                        
                        avg_confidence = sum(page_confidences) / len(page_confidences) / 100.0
                        confidence_scores.append(avg_confidence)
                        
                        logger.info(f"Page {page_num + 1}: OCR extracted {len(page_text)} characters, confidence: {avg_confidence:.2f}")
                    else:
                        logger.warning(f"Page {page_num + 1}: No text extracted with OCR")
                
                doc.close()
                
                if ocr_pages:
                    extracted_text = "\n\n".join(ocr_pages)
                else:
                    warnings.append("OCR failed to extract readable text")
            
            # Calculate overall confidence
            overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
            
            # Estimate cost (free method)
            cost_estimate = 0.0
            
            return PDFExtractionResult(
                method=PDFExtractionMethod.PDFMINER_TESSERACT,
                extracted_text=extracted_text,
                confidence_score=overall_confidence,
                processing_time=0.0,  # Will be set by caller
                page_count=metadata.get("page_count", 0),
                metadata=metadata,
                cost_estimate=cost_estimate,
                warnings=warnings
            )
            
        except Exception as e:
            logger.error(f"PDFMiner + Tesseract extraction failed: {e}")
            raise
    
    def _extract_with_llamaextract(self, pdf_file_path: str, metadata: Dict[str, Any]) -> PDFExtractionResult:
        """Extract text using LlamaExtract (premium OCR)"""
        logger.info("Using LlamaExtract (premium OCR)")
        
        warnings = []
        
        try:
            # Set up LlamaIndex with API key
            os.environ["LLAMA_CLOUD_API_KEY"] = self.llamaindex_api_key
            
            # Use LlamaIndex PDF reader with premium parsing
            reader = PDFReader()
            documents = reader.load_data(file_path=pdf_file_path)
            
            # Extract text from documents
            extracted_text = ""
            for doc in documents:
                extracted_text += doc.text + "\n\n"
            
            # LlamaExtract typically has high confidence for OCR
            confidence_score = 0.9
            
            # Estimate cost for LlamaExtract (approximate pricing)
            # LlamaExtract charges per page processed
            page_count = metadata.get("page_count", 1)
            cost_per_page = 0.003  # Approximate cost per page
            cost_estimate = page_count * cost_per_page
            
            if not extracted_text.strip():
                warnings.append("LlamaExtract returned empty text")
                confidence_score = 0.0
            
            return PDFExtractionResult(
                method=PDFExtractionMethod.LLAMAEXTRACT,
                extracted_text=extracted_text.strip(),
                confidence_score=confidence_score,
                processing_time=0.0,
                page_count=page_count,
                metadata=metadata,
                cost_estimate=cost_estimate,
                warnings=warnings
            )
            
        except Exception as e:
            logger.error(f"LlamaExtract extraction failed: {e}")
            raise
    
    def _extract_with_google_ai(self, pdf_file_path: str, metadata: Dict[str, Any]) -> PDFExtractionResult:
        """Extract text using Google Document AI"""
        logger.info("Using Google Document AI")
        
        warnings = []
        
        try:
            # Initialize Document AI client
            client = documentai.DocumentProcessorServiceClient()
            
            # The full resource name of the processor
            name = client.processor_path(
                self.google_project_id, 
                self.google_location, 
                self.google_processor_id
            )
            
            # Read the file into memory
            with open(pdf_file_path, "rb") as pdf_file:
                pdf_content = pdf_file.read()
            
            # Configure the process request
            request = documentai.ProcessRequest(
                name=name,
                raw_document=documentai.RawDocument(
                    content=pdf_content,
                    mime_type="application/pdf"
                )
            )
            
            # Process the document
            logger.info("Sending document to Google Document AI...")
            result = client.process_document(request=request)
            document = result.document
            
            # Extract text
            extracted_text = document.text
            
            # Calculate confidence based on OCR confidence if available
            confidence_scores = []
            if document.pages:
                for page in document.pages:
                    if hasattr(page, 'tokens'):
                        for token in page.tokens:
                            if hasattr(token, 'detection_confidence'):
                                confidence_scores.append(token.detection_confidence)
            
            overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.85
            
            # Estimate cost for Google Document AI
            # Google charges per page processed
            page_count = metadata.get("page_count", 1)
            cost_per_page = 0.0015  # Approximate cost per page for Document AI
            cost_estimate = page_count * cost_per_page
            
            if not extracted_text.strip():
                warnings.append("Google Document AI returned empty text")
                overall_confidence = 0.0
            
            return PDFExtractionResult(
                method=PDFExtractionMethod.GOOGLE_DOCUMENT_AI,
                extracted_text=extracted_text,
                confidence_score=overall_confidence,
                processing_time=0.0,
                page_count=page_count,
                metadata=metadata,
                cost_estimate=cost_estimate,
                warnings=warnings
            )
            
        except Exception as e:
            logger.error(f"Google Document AI extraction failed: {e}")
            raise
    
    def get_available_methods(self) -> List[PDFExtractionMethod]:
        """Get list of available extraction methods based on configuration"""
        methods = [PDFExtractionMethod.PDFMINER_TESSERACT]  # Always available
        
        if self.llamaextract_available:
            methods.append(PDFExtractionMethod.LLAMAEXTRACT)
        
        if self.google_ai_available:
            methods.append(PDFExtractionMethod.GOOGLE_DOCUMENT_AI)
        
        return methods
    
    def get_method_info(self, method: PDFExtractionMethod) -> Dict[str, Any]:
        """Get information about a specific extraction method"""
        method_info = {
            PDFExtractionMethod.PDFMINER_TESSERACT: {
                "name": "PDFMiner + Tesseract OCR",
                "description": "Free method using PDFMiner for text extraction and Tesseract for OCR",
                "cost": "Free",
                "best_for": "Simple PDFs with clear text or basic scanned documents",
                "pros": ["No cost", "Good for simple documents", "Reliable for text-based PDFs"],
                "cons": ["Lower accuracy on complex layouts", "Slower OCR processing", "Limited formatting preservation"],
                "available": True
            },
            PDFExtractionMethod.LLAMAEXTRACT: {
                "name": "LlamaExtract Premium OCR",
                "description": "Premium OCR service with advanced document understanding",
                "cost": "~$0.003 per page",
                "best_for": "Medium complexity PDFs with mixed text and images",
                "pros": ["High accuracy OCR", "Good layout understanding", "Handles mixed content well"],
                "cons": ["Costs money per page", "Requires API key", "Internet connection required"],
                "available": self.llamaextract_available
            },
            PDFExtractionMethod.GOOGLE_DOCUMENT_AI: {
                "name": "Google Document AI",
                "description": "Enterprise-grade document processing with advanced AI",
                "cost": "~$0.0015 per page",
                "best_for": "Complex PDFs with tables, forms, and structured layouts",
                "pros": ["Highest accuracy", "Excellent layout preservation", "Handles complex structures", "Form field extraction"],
                "cons": ["Requires Google Cloud setup", "Most expensive option", "Complex configuration"],
                "available": self.google_ai_available
            }
        }
        
        return method_info.get(method, {})
    
    def estimate_cost(self, pdf_file_path: str, method: PDFExtractionMethod) -> float:
        """Estimate cost for processing a PDF with given method"""
        try:
            metadata = self._get_pdf_metadata(pdf_file_path)
            page_count = metadata.get("page_count", 1)
            
            if method == PDFExtractionMethod.PDFMINER_TESSERACT:
                return 0.0  # Free
            elif method == PDFExtractionMethod.LLAMAEXTRACT:
                return page_count * 0.003  # $0.003 per page
            elif method == PDFExtractionMethod.GOOGLE_DOCUMENT_AI:
                return page_count * 0.0015  # $0.0015 per page
            else:
                return 0.0
        except Exception:
            return 0.0


def test_pdf_extractor():
    """Test the PDF extractor with a sample PDF"""
    # This would be used for testing
    extractor = PDFExtractor()
    
    print("Available methods:")
    for method in extractor.get_available_methods():
        info = extractor.get_method_info(method)
        print(f"- {info['name']}: {info['description']}")
        print(f"  Cost: {info['cost']}")
        print(f"  Available: {info['available']}")
        print()


if __name__ == "__main__":
    test_pdf_extractor()
