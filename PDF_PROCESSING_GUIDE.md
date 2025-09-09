# PDF-Based Site Survey Processing Guide

## Overview

The Deed Parser System now supports PDF-based input processing for site survey documents. This feature allows users to upload PDF site surveys and automatically extract boundary information for CAD drawing.

## 🔄 Processing Workflow

The PDF processing follows a 6-step workflow:

### 1. **PDF Upload** 📤
- Upload site survey PDF documents
- System analyzes file and shows available extraction methods
- Supports files up to 10MB

### 2. **Text Extraction** 🔍
Choose from 3 extraction methods:

#### **PDFMiner + Tesseract OCR** (Free)
- **Cost:** Free
- **Best for:** Simple PDFs with clear text or basic scanned documents
- **Pros:** No cost, reliable for text-based PDFs
- **Cons:** Lower accuracy on complex layouts, slower processing
- **Processing time:** ~8 seconds per page

#### **LlamaExtract Premium OCR** ($0.003/page)
- **Cost:** ~$0.003 per page
- **Best for:** Medium complexity PDFs with mixed text and images
- **Pros:** High accuracy OCR, good layout understanding
- **Cons:** Requires API key, costs money per page
- **Processing time:** ~3 seconds per page

#### **Google Document AI** ($0.0015/page)
- **Cost:** ~$0.0015 per page  
- **Best for:** Complex PDFs with tables, forms, structured layouts
- **Pros:** Highest accuracy, excellent layout preservation, handles complex structures
- **Cons:** Requires Google Cloud setup, complex configuration
- **Processing time:** ~2 seconds per page

### 3. **Boundary Information Extraction** 🎯
- AI analyzes extracted text for site boundary data
- Identifies different types of boundary information:
  - Line/Curve tables
  - Property descriptions
  - Legal descriptions
  - Survey notes
  - Coordinate tables
  - Bearing/distance tables
- Uses OpenAI GPT for intelligent extraction
- **Cost:** ~$0.005-0.015 per 1K tokens (depends on model)

### 4. **CAD Table Formatting** 📐
- Processes **ALL** extracted boundary information (not just primary data)
- Combines line/curve tables, legal descriptions, survey notes, and coordinate data
- Cross-references multiple data sources for complete boundary extraction
- Creates comprehensive line/curve table format suitable for CAD drawing
- **Cost:** ~$0.005-0.015 per 1K tokens (depends on model)

### 5. **Review & Edit** 📊
- Use existing review and edit functionality
- Manually review and adjust extracted survey calls
- Edit coordinates, bearings, distances
- Validate geometry and closure

### 6. **Visualization** 🗺️
- Use existing visualization functionality
- Generate interactive SVG boundary drawings
- Export to various formats (SVG, CSV, JSON)

## 💰 Cost Analysis

### Typical Costs (10-page PDF):

| Method | PDF Extraction | AI Processing | Total |
|--------|----------------|---------------|-------|
| PDFMiner + GPT-4o-mini | Free | ~$0.02 | **~$0.02** |
| PDFMiner + GPT-4o | Free | ~$0.08 | **~$0.08** |
| LlamaExtract + GPT-4o-mini | $0.03 | ~$0.02 | **~$0.05** |
| LlamaExtract + GPT-4o | $0.03 | ~$0.08 | **~$0.11** |
| Google AI + GPT-4o-mini | $0.015 | ~$0.02 | **~$0.035** |
| Google AI + GPT-4o | $0.015 | ~$0.08 | **~$0.095** |

### Monthly Cost Examples:
- **Light usage** (5 PDFs/month, 5 pages each): $0.25-2.75/month
- **Regular usage** (20 PDFs/month, 5 pages each): $1.00-11.00/month  
- **Heavy usage** (100 PDFs/month, 10 pages each): $20.00-220.00/month

## 🔧 Setup Requirements

### Required for All Methods:
- OpenAI API key (for boundary extraction and CAD formatting)

### For LlamaExtract:
- LlamaIndex API key
- Install: `pip install llama-index llama-index-readers-file`

### For Google Document AI:
1. Create Google Cloud Project
2. Enable Document AI API
3. Create Document AI Processor (type: "Form Parser" or "Document OCR")
4. Set up service account authentication
5. Install: `pip install google-cloud-documentai`
6. Set environment variables:
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account.json"
   export GOOGLE_CLOUD_PROJECT_ID="your-project-id"
   export GOOGLE_DOCUMENT_AI_PROCESSOR_ID="your-processor-id"
   ```

## 📄 Supported PDF Types

### Ideal PDFs:
- Site survey documents with boundary descriptions
- Plat maps with line/curve tables
- Legal descriptions with metes and bounds
- Survey reports with coordinate data

### PDF Format Support:
- Text-based PDFs (best results)
- Scanned/image PDFs (requires OCR)
- Mixed text/image PDFs
- Tables and structured layouts
- Hand-drawn surveys (limited support)

## 🎯 Best Practices

### For Best Results:
1. **Use high-quality PDFs** - Clear, well-scanned documents work better
2. **Choose appropriate extraction method** - Match complexity to method capability
3. **Review extracted data** - Always verify AI-extracted boundary information
4. **Start with free method** - Try PDFMiner+Tesseract first for simple PDFs
5. **Monitor costs** - Use the built-in cost tracking for budget management

### Common Issues:
- **Poor OCR quality**: Try premium extraction methods
- **Missing boundary data**: Check if PDF contains actual survey information
- **Incorrect parsing**: Review and edit in the Review & Edit tab
- **High costs**: Consider using GPT-4o-mini instead of GPT-4o

## 🔄 Integration with Existing Workflow

The PDF processing integrates seamlessly with existing functionality:

- **Same Review & Edit tab** - Edit extracted survey calls
- **Same Visualization** - Generate boundary drawings
- **Same Export options** - SVG, CSV, JSON exports
- **Same Geometry calculations** - Closure analysis, area calculations
- **Multi-tract support** - Handle multiple property tracts

## 📊 Quality Indicators

The system provides confidence scores for:
- **PDF extraction quality** - Based on OCR confidence
- **Boundary data extraction** - AI confidence in found data
- **Survey call parsing** - Confidence in individual calls
- **Overall processing** - Combined quality assessment

## 🚀 Performance

### Processing Times (typical):
- **Upload & Analysis:** < 5 seconds
- **PDF Extraction:** 2-8 seconds per page (method dependent)
- **Boundary Extraction:** 10-30 seconds (text length dependent)
- **CAD Formatting:** 10-20 seconds
- **Total for 10-page PDF:** 1-3 minutes

### Scalability:
- Handles PDFs up to 10MB
- Supports 1-100+ pages
- Batch processing capable
- API rate limiting respected

## 🔒 Security & Privacy

- **API Keys:** Stored securely in session (not persisted)
- **PDF Files:** Temporarily stored, automatically cleaned up
- **Extracted Text:** Not logged or stored permanently
- **Cloud Processing:** LlamaExtract and Google AI process data in cloud
- **Local Processing:** PDFMiner+Tesseract processes locally

## 📈 Future Enhancements

Planned improvements:
- Batch PDF processing
- Custom extraction templates
- Advanced table recognition
- Integration with CAD software
- PDF annotation and markup
- Historical cost tracking
- API endpoint for automation

## 🆘 Troubleshooting

### Common Issues:

**"No boundary data found"**
- Check if PDF contains actual survey information
- Try different extraction method
- Verify PDF is not corrupted

**"Text extraction failed"**
- PDF may be password protected
- File may be corrupted
- Try different extraction method

**"High API costs"**
- Switch to GPT-4o-mini model
- Use free PDFMiner+Tesseract for simple PDFs
- Process fewer pages at once

**"Poor extraction quality"**
- Try premium extraction methods (LlamaExtract/Google AI)
- Ensure PDF has good image quality
- Check if text is selectable in PDF

**"Google Document AI setup issues"**
- Verify Google Cloud project setup
- Check service account permissions
- Ensure Document AI API is enabled
- Verify processor ID is correct

## 💡 Cost Optimization Tips

1. **Start Free:** Always try PDFMiner+Tesseract first
2. **Use GPT-4o-mini:** 85% cost savings vs GPT-4o with similar quality
3. **Batch Processing:** Process multiple PDFs together when possible
4. **Quality vs Cost:** Balance extraction method with budget
5. **Monitor Usage:** Track costs using built-in cost analysis
6. **Optimize PDFs:** Use high-quality, text-searchable PDFs when possible

---

For technical support or questions, please refer to the application logs or contact the development team.
